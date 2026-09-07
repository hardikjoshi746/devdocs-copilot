from dataclasses import dataclass
from typing import Literal
import ast


@dataclass
class Document:
    id: str            # unique identifier: filepath::ClassName::method_name
    content: str       # raw source text of this chunk
    type: Literal["code", "doc", "issue"]
    source: str        # file path or issue URL — where this chunk came from
    parent_id: str | None  # id of the containing chunk (class for methods, None for top-level)
    metadata: dict     # structured fields for filtering: function_name, class_name, start_line


def chunk_python_file(source: str, filepath: str) -> list[Document]:
    """
    Parse a Python source file and split it into Document chunks at
    function/class boundaries using the AST.

    Why AST instead of splitting every N tokens?
    Token-splitting can cut mid-function, producing incomplete chunks whose
    embeddings are meaningless. AST knows where each function/class actually
    begins and ends — every chunk is a complete semantic unit.

    Parent-child structure:
    - ClassDef  → parent chunk  (parent_id=None)
    - FunctionDef inside class → child chunk (parent_id=class chunk id)
    - Top-level FunctionDef → standalone chunk (parent_id=None)

    The retrieval evaluator uses parent_id to fetch broader context when a
    child chunk scores borderline (EXPAND path).
    """
    documents = []
    # ast.parse() converts source text into a syntax tree.
    # tree.body contains only the top-level nodes (classes, top-level functions,
    # imports) — not everything nested inside them.
    tree = ast.parse(source)

    for curr in tree.body:
        if isinstance(curr, ast.ClassDef):
            # Build a stable unique id for this class chunk
            class_id = f"{filepath}::{curr.name}"
            # ast.get_source_segment() extracts the exact source lines for a node —
            # no manual line slicing needed. Falls back to "" if it can't extract.
            class_content = ast.get_source_segment(source, curr) or ""

            # Parent chunk — the entire class body including all methods.
            # Stored so the EXPAND path can fetch it when a child scores borderline.
            documents.append(Document(
                id=class_id,
                content=class_content,
                type="code",
                source=filepath,
                parent_id=None,
                metadata={"class_name": curr.name, "start_line": curr.lineno},
            ))

            # Child chunks — one per method inside this class.
            # Each points back to the class via parent_id.
            for method in curr.body:
                if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    documents.append(Document(
                        id=f"{class_id}::{method.name}",
                        content=ast.get_source_segment(source, method) or "",
                        type="code",
                        source=filepath,
                        parent_id=class_id,  # links back to the class chunk
                        metadata={
                            "function_name": method.name,
                            "class_name": curr.name,
                            "start_line": method.lineno,  # method's own line, not the class's
                        },
                    ))

        elif isinstance(curr, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Top-level function — not inside any class, so no parent.
            documents.append(Document(
                id=f"{filepath}::{curr.name}",
                content=ast.get_source_segment(source, curr) or "",
                type="code",
                source=filepath,
                parent_id=None,
                metadata={"function_name": curr.name, "start_line": curr.lineno},
            ))

    return documents

def chunk_markdown_file(source: str, filepath: str) -> list[Document]:
    documents = []
    lines = source.splitlines()

    curr_heading = None
    curr_lines = []
    inside_code = False
    seen_ids: dict[str, int] = {}  # tracks how many times each id has appeared

    def make_id(heading: str) -> str:
        base = f"{filepath}::{heading}"
        if base in seen_ids:
            seen_ids[base] += 1
            return f"{base}::{seen_ids[base]}"
        seen_ids[base] = 0
        return base

    for line in lines:

        if line.startswith("```"):
            inside_code = not inside_code # toggle
            continue

        if inside_code:
            curr_lines.append(line)
            continue

        if line.startswith("#"):
            if curr_heading is not None:
                documents.append(Document(
                    id=make_id(curr_heading),
                    content=curr_heading + "\n" + "\n".join(curr_lines),
                    type="doc",
                    source=filepath,
                    parent_id=None,
                    metadata={"heading": curr_heading}
                ))
            curr_heading = line.lstrip("#").strip()
            curr_lines = []
        else:
            curr_lines.append(line)
    if curr_heading is not None:
        documents.append(Document(
                    id=make_id(curr_heading),
                    content=curr_heading + "\n" + "\n".join(curr_lines),
                    type="doc",
                    source=filepath,
                    parent_id=None,
                    metadata={"heading": curr_heading}
        ))
    return documents

def chunk_issue(issues: list) -> list[Document]:
    documents = []
    for issue in issues:
        documents.append(Document(
            id=f"issue::{issue.number}",
            content=f"{issue.title}\n\n{issue.body}",
            type="issue",
            source=issue.url,
            parent_id=None,
            metadata={"issue_number" : issue.number, "state" : issue.state, "labels" : issue.labels}
        ))
    return documents


