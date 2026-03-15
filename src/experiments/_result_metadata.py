import json
import os
import shlex
import socket
import sys
from datetime import datetime
from typing import Any


_SENSITIVE_PARTS = ("key", "token", "secret", "password")


def _is_sensitive_name(name: str) -> bool:
    normalized = name.lower().replace("-", "_")
    return any(part in normalized for part in _SENSITIVE_PARTS)


def _sanitize_value(name: str, value: Any) -> Any:
    if _is_sensitive_name(name):
        return "<redacted>"
    if isinstance(value, dict):
        return {str(k): _sanitize_value(str(k), v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_sanitize_value(name, item) for item in value]
    if isinstance(value, list):
        return [_sanitize_value(name, item) for item in value]
    return value


def _sanitize_argv(argv: list[str]) -> list[str]:
    sanitized: list[str] = []
    index = 0
    while index < len(argv):
        token = argv[index]
        if token.startswith("--"):
            option, has_equals, value = token.partition("=")
            option_name = option.lstrip("-")
            if _is_sensitive_name(option_name):
                if has_equals:
                    sanitized.append(f"{option}=<redacted>")
                else:
                    sanitized.append(option)
                    if index + 1 < len(argv) and not argv[index + 1].startswith("-"):
                        sanitized.append("<redacted>")
                        index += 1
            else:
                sanitized.append(token)
        else:
            sanitized.append(token)
        index += 1
    return sanitized


def collect_run_metadata(script_path: str, args: Any, *, argv: list[str] | None = None,
                         result_dir: str | None = None, logs_dir: str | None = None) -> dict[str, Any]:
    raw_args = vars(args) if hasattr(args, "__dict__") else dict(args)
    sanitized_args = {key: _sanitize_value(key, raw_args[key]) for key in sorted(raw_args)}
    sanitized_argv = _sanitize_argv(list(argv or sys.argv))
    return {
        "script_name": os.path.basename(script_path),
        "script_path": os.path.abspath(script_path),
        "cwd": os.getcwd(),
        "launched_at": datetime.now().isoformat(timespec="seconds"),
        "python_executable": sys.executable,
        "hostname": socket.gethostname(),
        "result_dir": os.path.abspath(result_dir) if result_dir else None,
        "logs_dir": os.path.abspath(logs_dir) if logs_dir else None,
        "command_redacted": shlex.join(sanitized_argv),
        "argv_redacted": sanitized_argv,
        "args": sanitized_args,
    }


def _format_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def write_run_metadata_block(handle: Any, metadata: dict[str, Any], *, heading: str = "Run Metadata") -> None:
    handle.write(f">>> {heading}\n")
    for key in [
        "script_name",
        "script_path",
        "launched_at",
        "cwd",
        "python_executable",
        "hostname",
        "result_dir",
        "logs_dir",
        "command_redacted",
    ]:
        value = metadata.get(key)
        if value:
            handle.write(f"  {key}: {_format_value(value)}\n")
    handle.write("  args:\n")
    for key, value in metadata.get("args", {}).items():
        handle.write(f"    {key}: {_format_value(value)}\n")
    handle.write("\n")


def write_run_metadata_json(logs_dir: str, metadata: dict[str, Any], *, file_name: str = "run_metadata.json") -> str:
    path = os.path.join(logs_dir, file_name)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
    return path
