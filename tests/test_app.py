import importlib
import pytest

def test_app_importable():
    mod = importlib.import_module("app")
    assert mod is not None
