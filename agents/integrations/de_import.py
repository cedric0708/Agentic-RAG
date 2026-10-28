"""de/scripts 모듈을 agents에서 안전하게 import하기 위한 로더."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Optional

_de_scripts: Optional[Path] = None
_loaded: dict[str, ModuleType] = {}


def de_scripts_dir() -> Path:
    global _de_scripts
    if _de_scripts is None:
        _de_scripts = Path(__file__).resolve().parents[2] / "de" / "scripts"
    return _de_scripts


def load_de_module(module_name: str) -> ModuleType:
    """vector_tools / graph_tools 등 de/scripts/*.py 를 로드합니다."""
    if module_name in _loaded:
        return _loaded[module_name]

    path = de_scripts_dir() / f"{module_name}.py"
    if not path.is_file():
        raise ImportError(f"DE module not found: {path}")

    scripts = str(de_scripts_dir())
    if scripts not in sys.path:
        sys.path.insert(0, scripts)

    spec = importlib.util.spec_from_file_location(f"de_{module_name}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"de_{module_name}"] = mod
    spec.loader.exec_module(mod)
    _loaded[module_name] = mod
    return mod
