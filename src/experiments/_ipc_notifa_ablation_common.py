from pathlib import Path


BASE_FILE = Path(__file__).with_name("OmniGenV2_IPC_DINO_Importance_Aircraft_AR_slim_noTIFA.py")


def _replace_once(source: str, old: str, new: str) -> str:
    if old not in source:
        raise RuntimeError(f"Replacement target not found:\n{old}")
    return source.replace(old, new, 1)


def exec_ablation(wrapper_file: str, replacements: list[tuple[str, str]]) -> None:
    source = BASE_FILE.read_text(encoding="utf-8")
    for old, new in replacements:
        source = _replace_once(source, old, new)
    exec_globals = {
        "__name__": "__main__",
        "__file__": wrapper_file,
    }
    exec(compile(source, wrapper_file, "exec"), exec_globals)
