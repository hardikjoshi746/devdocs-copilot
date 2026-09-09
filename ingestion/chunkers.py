from dataclasses import dataclass
from typing import Literal
import ast
from pathlib import Path

import tree_sitter_python as tspython
import tree_sitter_javascript as tsjavascript
import tree_sitter_typescript as tstypescript
import tree_sitter_java as tsjava
from tree_sitter import Language, Parser


# ---------------------------------------------------------------------------
# Language registry — extension → (tree-sitter language module, language name)
# ---------------------------------------------------------------------------

_LANGUAGE_MAP: dict[str, tuple] = {
    ".py":   (tspython.language(),                 "python"),
    ".js":   (tsjavascript.language(),             "javascript"),
    ".jsx":  (tsjavascript.language(),             "javascript"),  # React JSX
    ".ts":   (tstypescript.language_typescript(),  "typescript"),
    ".tsx":  (tstypescript.language_tsx(),         "typescript"),  # React TSX
    ".java": (tsjava.language(),                   "java"),
}

# Node type names that represent a function or class in each language's grammar
_FUNCTION_NODES = {
    "python":     {"function_definition", "async_function_definition"},
    "javascript": {"function_declaration", "arrow_function", "method_definition", "function_expression"},
    "typescript": {"function_declaration", "arrow_function", "method_definition", "function_expression"},
    "java":       {"method_declaration", "constructor_declaration"},
}

_CLASS_NODES = {
    "python":     {"class_definition"},
    "javascript": {"class_declaration"},
    "typescript": {"class_declaration"},
    "java":       {"class_declaration", "interface_declaration", "enum_declaration"},
}

# Fixed-size fallback: chunk size and overlap in lines
_FALLBACK_CHUNK_LINES = 100
_FALLBACK_OVERLAP_LINES = 20


@dataclass
class Document:
    id: str            # unique identifier: repo::filepath::ClassName::method_name
    content: str       # raw source text of this chunk
    type: Literal["code", "doc", "issue"]
    source: str        # file path or issue URL — where this chunk came from
    parent_id: str | None  # id of the containing chunk (class for methods, None for top-level)
    metadata: dict     # function_name, class_name, start_line, language, repo, etc.


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------

def detect_language(filepath: str) -> str | None:
    """
    Returns the language name for a given file path based on its extension,
    or None if the extension is not in the supported language map.
    """
    ext = Path(filepath).suffix.lower()
    entry = _LANGUAGE_MAP.get(ext)
    return entry[1] if entry else None


# ---------------------------------------------------------------------------
# Tree-sitter chunker (Python, JS, TS)
# ---------------------------------------------------------------------------

def _get_node_name(node, source_bytes: bytes) -> str:
    """Extract the identifier name from a function/class node.

    Different grammars use different node types for names:
    - Python, JS functions: "identifier"
    - JS/TS class methods: "property_identifier"
    We check both so one helper works across all supported languages.
    """
    # Check identifier first — in Java, method_declaration children include the
    # return type (type_identifier) before the method name (identifier), so we
    # must not let type_identifier win over identifier.
    for child in node.children:
        if child.type in ("identifier", "property_identifier"):
            return source_bytes[child.start_byte:child.end_byte].decode("utf-8", errors="replace")
    # Fall back to type_identifier only if no plain identifier found (e.g. class names in TS)
    for child in node.children:
        if child.type == "type_identifier":
            return source_bytes[child.start_byte:child.end_byte].decode("utf-8", errors="replace")
    return "unknown"


def chunk_code_file(source: str, filepath: str, repo: str = "") -> list[Document]:
    """
    Parse a source file using tree-sitter and split into Document chunks at
    function/class boundaries.

    Supports Python, JavaScript, and TypeScript via the _LANGUAGE_MAP registry.
    Falls back to fixed-size line chunking for unsupported file types.

    Parent-child structure mirrors chunk_python_file:
    - Class node  → parent chunk (parent_id=None)
    - Method inside class → child chunk (parent_id=class chunk id)
    - Top-level function → standalone chunk (parent_id=None)

    New metadata fields vs the old Python-only chunker:
    - language: detected language string
    - repo: repo slug passed by the caller (e.g. "tiangolo/fastapi")
    """
    ext = Path(filepath).suffix.lower()
    language_entry = _LANGUAGE_MAP.get(ext)

    if language_entry is None:
        return _chunk_fallback(source, filepath, repo)

    lang_ptr, lang_name = language_entry
    language = Language(lang_ptr)
    parser = Parser(language)

    source_bytes = source.encode("utf-8")
    tree = parser.parse(source_bytes)

    fn_types = _FUNCTION_NODES.get(lang_name, set())
    cls_types = _CLASS_NODES.get(lang_name, set())

    documents = []
    id_prefix = f"{repo}::{filepath}" if repo else filepath
    seen_ids: dict[str, int] = {}

    def unique_id(base: str) -> str:
        if base not in seen_ids:
            seen_ids[base] = 0
            return base
        seen_ids[base] += 1
        return f"{base}::{seen_ids[base]}"

    for node in tree.root_node.children:
        if node.type in cls_types:
            class_name = _get_node_name(node, source_bytes)
            class_id = unique_id(f"{id_prefix}::{class_name}")
            class_content = source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")

            documents.append(Document(
                id=class_id,
                content=class_content,
                type="code",
                source=filepath,
                parent_id=None,
                metadata={
                    "class_name": class_name,
                    "start_line": node.start_point[0] + 1,
                    "language": lang_name,
                    "repo": repo,
                },
            ))

            # Methods inside the class — may be wrapped in a "block" or "class_body"
            # node depending on the language grammar, so we search one level deeper.
            body_nodes = []
            for child in node.children:
                if child.type in ("block", "class_body"):
                    body_nodes = child.children
                    break
            else:
                body_nodes = node.children

            for child in body_nodes:
                if child.type in fn_types:
                    method_name = _get_node_name(child, source_bytes)
                    documents.append(Document(
                        id=unique_id(f"{class_id}::{method_name}"),
                        content=source_bytes[child.start_byte:child.end_byte].decode("utf-8", errors="replace"),
                        type="code",
                        source=filepath,
                        parent_id=class_id,
                        metadata={
                            "function_name": method_name,
                            "class_name": class_name,
                            "start_line": child.start_point[0] + 1,
                            "language": lang_name,
                            "repo": repo,
                        },
                    ))

        elif node.type in fn_types:
            fn_name = _get_node_name(node, source_bytes)
            documents.append(Document(
                id=unique_id(f"{id_prefix}::{fn_name}"),
                content=source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace"),
                type="code",
                source=filepath,
                parent_id=None,
                metadata={
                    "function_name": fn_name,
                    "start_line": node.start_point[0] + 1,
                    "language": lang_name,
                    "repo": repo,
                },
            ))

    return documents


def _chunk_fallback(source: str, filepath: str, repo: str = "") -> list[Document]:
    """
    Fixed-size line chunker for file types not supported by tree-sitter.
    Splits every _FALLBACK_CHUNK_LINES lines with _FALLBACK_OVERLAP_LINES overlap
    so context is not lost at chunk boundaries.
    """
    lines = source.splitlines()
    documents = []
    step = _FALLBACK_CHUNK_LINES - _FALLBACK_OVERLAP_LINES
    id_prefix = f"{repo}::{filepath}" if repo else filepath

    for i, start in enumerate(range(0, len(lines), step)):
        chunk_lines = lines[start:start + _FALLBACK_CHUNK_LINES]
        documents.append(Document(
            id=f"{id_prefix}::chunk_{i}",
            content="\n".join(chunk_lines),
            type="code",
            source=filepath,
            parent_id=None,
            metadata={
                "start_line": start + 1,
                "language": "unknown",
                "repo": repo,
            },
        ))

    return documents


# ---------------------------------------------------------------------------
# Original Python-only chunker — kept for backward compatibility
# Uses ast.parse internally; new code should prefer chunk_code_file()
# ---------------------------------------------------------------------------

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
    tree = ast.parse(source)

    for curr in tree.body:
        if isinstance(curr, ast.ClassDef):
            class_id = f"{filepath}::{curr.name}"
            class_content = ast.get_source_segment(source, curr) or ""

            documents.append(Document(
                id=class_id,
                content=class_content,
                type="code",
                source=filepath,
                parent_id=None,
                metadata={"class_name": curr.name, "start_line": curr.lineno},
            ))

            for method in curr.body:
                if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    documents.append(Document(
                        id=f"{class_id}::{method.name}",
                        content=ast.get_source_segment(source, method) or "",
                        type="code",
                        source=filepath,
                        parent_id=class_id,
                        metadata={
                            "function_name": method.name,
                            "class_name": curr.name,
                            "start_line": method.lineno,
                        },
                    ))

        elif isinstance(curr, (ast.FunctionDef, ast.AsyncFunctionDef)):
            documents.append(Document(
                id=f"{filepath}::{curr.name}",
                content=ast.get_source_segment(source, curr) or "",
                type="code",
                source=filepath,
                parent_id=None,
                metadata={"function_name": curr.name, "start_line": curr.lineno},
            ))

    return documents


# ---------------------------------------------------------------------------
# Markdown chunker
# ---------------------------------------------------------------------------

def chunk_markdown_file(source: str, filepath: str) -> list[Document]:
    documents = []
    lines = source.splitlines()

    curr_heading = None
    curr_lines = []
    inside_code = False
    seen_ids: dict[str, int] = {}

    def make_id(heading: str) -> str:
        base = f"{filepath}::{heading}"
        if base in seen_ids:
            seen_ids[base] += 1
            return f"{base}::{seen_ids[base]}"
        seen_ids[base] = 0
        return base

    for line in lines:
        if line.startswith("```"):
            inside_code = not inside_code
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


# ---------------------------------------------------------------------------
# Issue chunker
# ---------------------------------------------------------------------------

def chunk_issue(issues: list) -> list[Document]:
    documents = []
    for issue in issues:
        documents.append(Document(
            id=f"issue::{issue.number}",
            content=f"{issue.title}\n\n{issue.body}",
            type="issue",
            source=issue.url,
            parent_id=None,
            metadata={"issue_number": issue.number, "state": issue.state, "labels": issue.labels}
        ))
    return documents
