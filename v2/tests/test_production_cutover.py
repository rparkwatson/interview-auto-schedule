from __future__ import annotations

import ast
from pathlib import Path

from streamlit.testing.v1 import AppTest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _requirement_lines(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def test_production_launcher_registers_only_hidden_v2_page():
    launcher_path = REPOSITORY_ROOT / "app.py"
    tree = ast.parse(launcher_path.read_text(encoding="utf-8"))

    imports_v2_main = any(
        isinstance(node, ast.ImportFrom)
        and node.module == "v2.interview_scheduler_v2.admin_app"
        and any(alias.name == "main" for alias in node.names)
        for node in tree.body
    )
    page_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "st"
        and node.func.attr == "Page"
    ]
    navigation_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "st"
        and node.func.attr == "navigation"
    ]

    assert imports_v2_main
    assert len(page_calls) == 1
    assert isinstance(page_calls[0].args[0], ast.Name)
    assert page_calls[0].args[0].id == "main"
    assert len(navigation_calls) == 1
    position = next(
        keyword.value
        for keyword in navigation_calls[0].keywords
        if keyword.arg == "position"
    )
    assert isinstance(position, ast.Constant)
    assert position.value == "hidden"


def test_production_entrypoint_starts_v2_without_exceptions():
    app = AppTest.from_file(
        str(REPOSITORY_ROOT / "app.py"),
        default_timeout=15,
    ).run()

    assert not app.exception
    assert app.title[0].value == "Interview Scheduler"
    assert len(app.get("file_uploader")) == 2


def test_production_requirements_match_reviewed_v2_requirements():
    production = _requirement_lines(REPOSITORY_ROOT / "requirements.txt")
    reviewed_v2 = _requirement_lines(
        REPOSITORY_ROOT / "v2" / "requirements.txt"
    )

    assert production == reviewed_v2
