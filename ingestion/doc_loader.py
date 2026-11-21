"""
Universal document loader for the data/docs/ folder.

Drop any supported file into data/docs/ (or a subfolder) and it will be
picked up on the next ingestion run.

Supported formats
-----------------
  .pdf   — text + tables (pymupdf markdown mode) + image descriptions
  .docx  — text + tables (python-docx pipe tables) + image descriptions
  .md    — delegated to chunk_markdown_file
  .txt   — delegated to _chunk_fallback

Images embedded in PDFs and docx files are base64-encoded and sent to
Claude Haiku for a brief description, then inlined as:

    [Image N: <description>]

so they become searchable text in the vector index even though the
embedding model itself is text-only.
"""

import base64
import os
import re
import shutil
import zipfile
from pathlib import Path

from anthropic import AsyncAnthropic

from ingestion.chunkers import Document, _chunk_fallback, chunk_markdown_file
from ingestion.embed_and_store import _CHUNK_INJECTION_RE

# Claude vision model — haiku is fast and cheap for image descriptions.
_VISION_MODEL = "claude-haiku-4-5-20251001"
_VISION_PROMPT = (
    "Describe this image concisely for a technical documentation search index. "
    "Focus on what it shows: diagrams, architecture charts, screenshots, UI elements, "
    "data tables, graphs, or formulas. Keep the description under 100 words."
)

# Extensions this loader handles. Anything else is silently skipped.
_SUPPORTED = {".pdf", ".docx", ".md", ".txt"}

# ---------------------------------------------------------------------------
# File safety checks
# ---------------------------------------------------------------------------

# Max file size: 50 MB. Larger files are likely zip bombs or accidental drops.
_MAX_FILE_BYTES = 50 * 1024 * 1024

# Magic bytes for each binary format.
# A file claiming to be .pdf must start with %PDF-
# A .docx is a ZIP archive — must start with PK\x03\x04
_MAGIC: dict[str, bytes] = {
    ".pdf":  b"%PDF-",
    ".docx": b"PK\x03\x04",
}


def _validate_file(filepath: Path) -> None:
    """
    Raise ValueError if the file fails any structural check.
    Called before any parsing so corrupt or disguised files never reach
    the parser libraries.

    Checks (in order):
      1. Not empty
      2. Under 50 MB size cap (zip bomb / accidental large file)
      3. Magic bytes match the declared extension
      4. DOCX: ZIP CRC checksums verified via zipfile.testzip()
    """
    size = filepath.stat().st_size
    if size == 0:
        raise ValueError("empty file")
    if size > _MAX_FILE_BYTES:
        raise ValueError(f"file too large ({size / 1024 / 1024:.1f} MB > 50 MB limit)")

    expected_magic = _MAGIC.get(filepath.suffix.lower())
    if expected_magic:
        header = filepath.read_bytes()[:len(expected_magic)]
        if header != expected_magic:
            raise ValueError(
                f"magic bytes mismatch — file does not appear to be a valid "
                f"{filepath.suffix} (got {header!r})"
            )

    # DOCX is a ZIP archive — testzip() verifies every entry's CRC checksum.
    # Returns the name of the first bad file, or None if all entries are intact.
    if filepath.suffix.lower() == ".docx":
        try:
            with zipfile.ZipFile(filepath) as zf:
                bad_entry = zf.testzip()
                if bad_entry:
                    raise ValueError(f"ZIP CRC error in entry: {bad_entry!r}")
        except zipfile.BadZipFile as exc:
            raise ValueError(f"not a valid ZIP/DOCX archive: {exc}") from exc


def _check_injection(docs: list[Document], filepath: Path) -> None:
    """
    Scan all chunks produced from a file for prompt injection patterns.
    Raises ValueError if any chunk matches, so the caller can quarantine
    the source file.

    Why raise instead of warn: embed_and_store already warns about suspicious
    chunks from repos (we don't control those). For the docs folder we control
    what goes in, so a match here means someone deliberately added a poisoned
    file — it should be quarantined, not silently stored.
    """
    hits = [doc for doc in docs if _CHUNK_INJECTION_RE.search(doc.content)]
    if hits:
        raise ValueError(
            f"injection pattern detected in {len(hits)} chunk(s): "
            + ", ".join(f"chunk {i}" for i, _ in enumerate(hits))
        )


# ---------------------------------------------------------------------------
# Image description via Claude Haiku
# ---------------------------------------------------------------------------

async def _describe_image(image_bytes: bytes, ext: str) -> str:
    """
    Send raw image bytes to Claude Haiku vision and return a short description.
    ext is the file extension without the dot (e.g. "png", "jpg").
    """
    # Claude vision accepts: jpeg, png, gif, webp
    ext = ext.lower().replace("jpg", "jpeg")
    if ext not in ("jpeg", "png", "gif", "webp"):
        return f"[unsupported image format: {ext}]"

    client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    b64 = base64.standard_b64encode(image_bytes).decode()

    response = await client.messages.create(
        model=_VISION_MODEL,
        max_tokens=150,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": f"image/{ext}",
                        "data": b64,
                    },
                },
                {"type": "text", "text": _VISION_PROMPT},
            ],
        }],
    )
    return response.content[0].text.strip()


# ---------------------------------------------------------------------------
# PDF loader
# ---------------------------------------------------------------------------


async def _load_pdf(filepath: Path) -> str:
    """
    Extract content from a PDF file.

    - Text and tables: pymupdf4llm.to_markdown() — the official high-level API
      for pymupdf 2.x; produces proper markdown pipe tables and respects actual
      document headings (better chunk boundaries than artificial ## Page N).
    - Cross-page tables: handled internally by pymupdf4llm.
    - Images: extracted separately via the low-level pymupdf API, described by
      Claude Haiku, and appended to the markdown as [Image N: description].

    Returns a markdown string ready for chunk_markdown_file.
    """
    import pymupdf
    import pymupdf4llm

    # to_markdown handles text + tables across the full document reliably
    md_text = pymupdf4llm.to_markdown(str(filepath))

    # Extract and describe images from every page
    doc = pymupdf.open(str(filepath))
    image_lines: list[str] = []
    img_counter = 0
    for page in doc:
        for img_ref in page.get_images():
            xref = img_ref[0]
            try:
                img_info = doc.extract_image(xref)
                img_counter += 1
                description = await _describe_image(img_info["image"], img_info["ext"])
                image_lines.append(f"[Image {img_counter}: {description}]")
            except Exception:
                pass  # skip unreadable images without crashing the whole file

    if image_lines:
        md_text += "\n\n" + "\n".join(image_lines)

    return md_text


# ---------------------------------------------------------------------------
# DOCX loader
# ---------------------------------------------------------------------------

async def _load_docx(filepath: Path) -> str:
    """
    Extract content from a Word document.

    - Paragraphs → plain text lines
    - Tables → markdown pipe tables (| col | col | with separator row)
    - Inline images → extracted from the relationship store, described via
      Claude Haiku, inlined as [Image N: description]

    Returns a markdown-ish string suitable for chunk_markdown_file.
    """
    import docx
    from docx.oxml.ns import qn

    doc = docx.Document(str(filepath))
    parts: list[str] = []
    image_counter = 0

    for block in doc.element.body:
        # Strip namespace prefix from tag name
        tag = block.tag.split("}")[-1] if "}" in block.tag else block.tag

        if tag == "p":
            # Normal paragraph — also scan for inline images
            para = docx.text.paragraph.Paragraph(block, doc)
            text = para.text.strip()
            if text:
                parts.append(text)

            # Scan for blip references (inline images)
            for elem in block.iter():
                if elem.tag.endswith("}blip"):
                    r_embed = elem.get(qn("r:embed"))
                    if not r_embed:
                        continue
                    rel = doc.part.rels.get(r_embed)
                    if rel is None:
                        continue
                    img_bytes = rel.target_part.blob
                    ext = rel.target_ref.split(".")[-1]
                    image_counter += 1
                    try:
                        description = await _describe_image(img_bytes, ext)
                        parts.append(f"[Image {image_counter}: {description}]")
                    except Exception:
                        pass

        elif tag == "tbl":
            # Table → markdown pipe table
            tbl = docx.table.Table(block, doc)
            rows: list[str] = []
            for row_idx, row in enumerate(tbl.rows):
                cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
                rows.append("| " + " | ".join(cells) + " |")
                if row_idx == 0:
                    # Header separator
                    rows.append("| " + " | ".join(["---"] * len(cells)) + " |")
            parts.append("\n".join(rows))

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def load_docs_folder(
    folder: str = "data/docs",
    repo: str = "",
) -> list[Document]:
    """
    Walk a directory and load all supported documents into Document chunks.

    Routing:
      .pdf  → _load_pdf  → chunk_markdown_file (## Page N as section headings)
      .docx → _load_docx → chunk_markdown_file
      .md   → chunk_markdown_file directly
      .txt  → _chunk_fallback (fixed-size lines)

    Every resulting Document gets:
      metadata["repo"]     = repo (blank string if not set)
      metadata["language"] = "doc"  (not set by markdown/fallback chunkers)

    Files that fail to parse are skipped with a warning — one bad file should
    never block the rest of the ingestion run.
    """
    folder_path = Path(folder)
    if not folder_path.exists():
        return []

    quarantine = folder_path / "_quarantine"
    documents: list[Document] = []

    for filepath in sorted(folder_path.rglob("*")):
        if not filepath.is_file():
            continue
        # Skip files already in quarantine
        if "_quarantine" in filepath.parts:
            continue
        if filepath.suffix.lower() not in _SUPPORTED:
            continue

        ext = filepath.suffix.lower()
        str_path = str(filepath)

        try:
            # Validate before any parsing — catches empty, oversized, and
            # disguised files (e.g. a shell script renamed to .pdf)
            _validate_file(filepath)

            if ext == ".pdf":
                markdown_text = await _load_pdf(filepath)
                docs = chunk_markdown_file(markdown_text, str_path)

            elif ext == ".docx":
                markdown_text = await _load_docx(filepath)
                docs = chunk_markdown_file(markdown_text, str_path)

            elif ext == ".md":
                docs = chunk_markdown_file(filepath.read_text(errors="replace"), str_path)

            elif ext == ".txt":
                docs = _chunk_fallback(
                    filepath.read_text(errors="replace"), str_path, repo
                )

            else:
                continue

            # Scan chunks for injection patterns before accepting the file.
            # Raises ValueError → triggers quarantine below if anything is found.
            _check_injection(docs, filepath)

            for doc in docs:
                doc.metadata["repo"] = repo
                doc.metadata.setdefault("language", "doc")

            documents.extend(docs)
            print(f"  [docs] {len(docs):>4} chunks  ← {filepath.name}")

        except Exception as exc:
            # Move the bad file to data/docs/_quarantine/ so it doesn't block
            # future runs and is preserved for investigation.
            quarantine.mkdir(exist_ok=True)
            dest = quarantine / filepath.name
            shutil.move(str(filepath), str(dest))
            print(f"  [docs] QUARANTINED {filepath.name} → _quarantine/ — {exc!r}")

    return documents