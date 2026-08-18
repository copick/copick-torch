import ast
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).parents[1]
SOURCE_ROOTS = (REPOSITORY_ROOT / "copick_torch", REPOSITORY_ROOT / "examples")
OPEN_METHODS = {"open", "open_array", "open_group"}
LITERAL_LEVEL_NAMES = {"0", "s0", "data"}


def zarr_open_calls(source):
    """Return Zarr open calls, including module and direct-import aliases."""
    tree = ast.parse(source)
    module_aliases = set()
    function_aliases = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for imported in node.names:
                if imported.name == "zarr":
                    module_aliases.add(imported.asname or imported.name)
        elif isinstance(node, ast.ImportFrom) and node.module == "zarr":
            for imported in node.names:
                if imported.name in OPEN_METHODS:
                    function_aliases[imported.asname or imported.name] = imported.name

    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in module_aliases
            and node.func.attr in OPEN_METHODS
        ):
            calls.append((node.func.attr, node))
        elif isinstance(node.func, ast.Name) and node.func.id in function_aliases:
            calls.append((function_aliases[node.func.id], node))
    return calls


def maintained_sources():
    for root in SOURCE_ROOTS:
        yield from root.rglob("*.py")


@pytest.mark.parametrize(
    ("source", "method"),
    [
        ("import zarr\nzarr.open(store)", "open"),
        ("import zarr as zr\nzr.open_group(store)", "open_group"),
        ("from zarr import open_array\nopen_array(store)", "open_array"),
        ("from zarr import open as zopen\nzopen(store)", "open"),
    ],
)
def test_zarr_open_scanner_catches_supported_aliases(source, method):
    assert [name for name, _node in zarr_open_calls(source)] == [method]


def test_only_storage_module_opens_zarr_and_it_is_exactly_read_only():
    calls = []
    for path in maintained_sources():
        for method, node in zarr_open_calls(path.read_text()):
            calls.append((path.relative_to(REPOSITORY_ROOT).as_posix(), method, node))

    assert len(calls) == 1
    path, method, node = calls[0]
    assert path == "copick_torch/storage.py"
    assert method == "open_group"
    assert node.args == []
    assert {keyword.arg for keyword in node.keywords} == {"store", "mode"}
    mode = next(keyword.value for keyword in node.keywords if keyword.arg == "mode")
    assert isinstance(mode, ast.Constant) and mode.value == "r"


def test_no_metadata_blind_literal_level_selection_remains():
    violations = []
    for path in maintained_sources():
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
                if node.slice.value in LITERAL_LEVEL_NAMES:
                    violations.append((path.relative_to(REPOSITORY_ROOT), node.lineno, node.slice.value))

    assert violations == []
