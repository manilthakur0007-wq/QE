import py_compile
import glob
import pytest


def test_compile_all_py_files():
    files = glob.glob("**/*.py", recursive=True)
    assert len(files) > 0
    for f in files:
        # Ensure source compiles without executing imports
        py_compile.compile(f, doraise=True)
