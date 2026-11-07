"""
tests/test_chunkers.py

Unit tests for all three chunkers. No network calls, no disk I/O —
pure function tests with real inputs.
"""

import pytest
from ingestion.chunkers import chunk_python_file, chunk_markdown_file, chunk_issue, Document
from ingestion.fetch_repo import Issue


# ---------------------------------------------------------------------------
# chunk_python_file
# ---------------------------------------------------------------------------

def test_top_level_function():
    source = "def hello(name: str) -> str:\n    return f'Hello {name}'"
    docs = chunk_python_file(source, "test.py")
    assert len(docs) == 1
    assert docs[0].id == "test.py::hello"
    assert docs[0].type == "code"
    assert docs[0].parent_id is None
    assert docs[0].source == "test.py"
    assert "def hello" in docs[0].content


def test_async_function():
    source = "async def fetch(url: str) -> dict:\n    pass"
    docs = chunk_python_file(source, "test.py")
    assert len(docs) == 1
    assert docs[0].id == "test.py::fetch"
    assert docs[0].parent_id is None


def test_class_with_methods():
    source = (
        "class Router:\n"
        "    def get(self): pass\n"
        "    def post(self): pass\n"
    )
    docs = chunk_python_file(source, "test.py")
    assert len(docs) == 3

    ids = [d.id for d in docs]
    assert "test.py::Router" in ids
    assert "test.py::Router::get" in ids
    assert "test.py::Router::post" in ids


def test_class_parent_id():
    source = (
        "class Router:\n"
        "    def get(self): pass\n"
    )
    docs = chunk_python_file(source, "test.py")
    class_doc = next(d for d in docs if d.id == "test.py::Router")
    method_doc = next(d for d in docs if d.id == "test.py::Router::get")

    assert class_doc.parent_id is None
    assert method_doc.parent_id == "test.py::Router"


def test_empty_file():
    docs = chunk_python_file("", "empty.py")
    assert docs == []


def test_file_with_only_imports():
    source = "import os\nimport sys\n"
    docs = chunk_python_file(source, "test.py")
    assert docs == []


def test_multiple_top_level_functions():
    source = (
        "def foo(): pass\n"
        "def bar(): pass\n"
        "def baz(): pass\n"
    )
    docs = chunk_python_file(source, "test.py")
    assert len(docs) == 3
    assert all(d.parent_id is None for d in docs)


def test_metadata_contains_start_line():
    source = "def hello(): pass\n"
    docs = chunk_python_file(source, "test.py")
    assert "start_line" in docs[0].metadata


def test_metadata_contains_function_name():
    source = "def hello(): pass\n"
    docs = chunk_python_file(source, "test.py")
    assert docs[0].metadata["function_name"] == "hello"


def test_class_metadata_contains_class_name():
    source = "class Foo:\n    pass\n"
    docs = chunk_python_file(source, "test.py")
    assert docs[0].metadata["class_name"] == "Foo"


# ---------------------------------------------------------------------------
# chunk_markdown_file
# ---------------------------------------------------------------------------

def test_markdown_splits_at_headings():
    source = "# Installation\npip install fastapi\n\n# Usage\nimport fastapi\n"
    docs = chunk_markdown_file(source, "test.md")
    assert len(docs) == 2
    assert docs[0].metadata["heading"] == "Installation"
    assert docs[1].metadata["heading"] == "Usage"


def test_markdown_id_convention():
    source = "# Installation\npip install fastapi\n"
    docs = chunk_markdown_file(source, "test.md")
    assert docs[0].id == "test.md::Installation"


def test_markdown_type_is_doc():
    source = "# Installation\npip install fastapi\n"
    docs = chunk_markdown_file(source, "test.md")
    assert docs[0].type == "doc"


def test_markdown_parent_id_is_none():
    source = "# Installation\npip install fastapi\n"
    docs = chunk_markdown_file(source, "test.md")
    assert docs[0].parent_id is None


def test_markdown_skips_hash_inside_code_block():
    source = (
        "# Real Heading\n"
        "```python\n"
        "# this is a comment, not a heading\n"
        "```\n"
        "some text\n"
    )
    docs = chunk_markdown_file(source, "test.md")
    assert len(docs) == 1
    assert docs[0].metadata["heading"] == "Real Heading"


def test_markdown_duplicate_headings_get_unique_ids():
    source = (
        "# Features\nfeature 1\n"
        "# Features\nfeature 2\n"
        "# Features\nfeature 3\n"
    )
    docs = chunk_markdown_file(source, "test.md")
    ids = [d.id for d in docs]
    assert len(ids) == len(set(ids))  # all unique


def test_markdown_empty_file():
    docs = chunk_markdown_file("", "test.md")
    assert docs == []


def test_markdown_no_headings():
    source = "just some text\nno headings here\n"
    docs = chunk_markdown_file(source, "test.md")
    assert docs == []


def test_markdown_content_includes_heading():
    source = "# Installation\npip install fastapi\n"
    docs = chunk_markdown_file(source, "test.md")
    assert "Installation" in docs[0].content


# ---------------------------------------------------------------------------
# chunk_issue
# ---------------------------------------------------------------------------

def make_issue(**kwargs):
    defaults = dict(
        id=1, number=42, title="Bug report",
        body="Something is broken", state="open",
        url="https://github.com/tiangolo/fastapi/issues/42",
        labels=[]
    )
    defaults.update(kwargs)
    return Issue(**defaults)


def test_issue_id_convention():
    issues = [make_issue(number=42)]
    docs = chunk_issue(issues)
    assert docs[0].id == "issue::42"


def test_issue_content_combines_title_and_body():
    issues = [make_issue(title="My Bug", body="Here is the body")]
    docs = chunk_issue(issues)
    assert "My Bug" in docs[0].content
    assert "Here is the body" in docs[0].content


def test_issue_type_is_issue():
    issues = [make_issue()]
    docs = chunk_issue(issues)
    assert docs[0].type == "issue"


def test_issue_source_is_url():
    url = "https://github.com/tiangolo/fastapi/issues/42"
    issues = [make_issue(url=url)]
    docs = chunk_issue(issues)
    assert docs[0].source == url


def test_issue_parent_id_is_none():
    issues = [make_issue()]
    docs = chunk_issue(issues)
    assert docs[0].parent_id is None


def test_multiple_issues():
    issues = [make_issue(number=1), make_issue(number=2), make_issue(number=3)]
    docs = chunk_issue(issues)
    assert len(docs) == 3
    assert [d.id for d in docs] == ["issue::1", "issue::2", "issue::3"]


def test_empty_issues():
    docs = chunk_issue([])
    assert docs == []