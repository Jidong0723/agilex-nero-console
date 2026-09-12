import ctypes
import os
from pathlib import Path
from typing import List


def is_64bit_process():
    return ctypes.sizeof(ctypes.c_void_p) == 8


def arch_dir_name():
    return "x64" if is_64bit_process() else "x32"


def _candidate_dll_paths():
    here = Path(__file__).resolve().parent
    arch_dir = arch_dir_name()

    candidates = [here / "bin" / arch_dir / "cando.dll"]  # type: List[Path]
    return candidates


def resolve_cando_dll_path():
    for candidate in _candidate_dll_paths():
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "cando.dll not found. Expected only under agx_cando/bin/x64 or agx_cando/bin/x32."
    )


def load_cando_dll():
    if os.name != "nt":
        raise OSError("agx_cando is supported on Windows only")
    dll_path = resolve_cando_dll_path()
    if dll_path.is_absolute():
        parent = str(dll_path.parent)
        os.environ["PATH"] = parent + os.pathsep + os.environ.get("PATH", "")
        try:
            os.add_dll_directory(parent)
        except (AttributeError, FileNotFoundError):
            pass
    return ctypes.WinDLL(str(dll_path))
