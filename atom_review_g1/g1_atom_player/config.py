from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


def _read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value[:1] == value[-1:] and value.startswith(("'", '"')):
            value = value[1:-1]
        values[key] = value
    return values


def _bool(value: str, default: bool = False) -> bool:
    if not value:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class AppConfig:
    root: Path
    host: str
    port: int
    catalog_path: Path
    sim_source_dir: Path
    preview_root: Path
    limits_path: Path
    review_store_path: Path
    audit_log_path: Path
    runner_path: Path
    real_enabled: bool
    network_interface: str


def load_config(root: str | Path) -> AppConfig:
    project_root = Path(root).expanduser().resolve()
    file_values = _read_env_file(project_root / "config.env")

    def value(name: str, default: str = "") -> str:
        return os.environ.get(name, file_values.get(name, default)).strip()

    def path_value(name: str, default: str) -> Path:
        raw = Path(value(name, default)).expanduser()
        return raw.resolve() if raw.is_absolute() else (project_root / raw).resolve()

    config = AppConfig(
        root=project_root,
        host=value("G1_ATOM_HOST", "127.0.0.1"),
        port=int(value("G1_ATOM_PORT", "8776")),
        catalog_path=path_value("G1_ATOM_CATALOG", "catalog.json"),
        sim_source_dir=path_value("G1_ATOM_SIM_SOURCE_DIR", "../sim"),
        preview_root=path_value("G1_ATOM_PREVIEW_ROOT", "previews"),
        limits_path=path_value("G1_ATOM_LIMITS", "config/g1_limits.json"),
        review_store_path=path_value("G1_ATOM_REVIEW_STORE", "runtime/reviews.json"),
        audit_log_path=path_value("G1_ATOM_AUDIT_LOG", "runtime/audit.jsonl"),
        runner_path=path_value("G1_ATOM_RUNNER", "build/g1_atom_motion_cli"),
        real_enabled=_bool(value("G1_ATOM_REAL_ENABLED", "0")),
        network_interface=value("G1_ATOM_NETWORK_INTERFACE"),
    )
    if not (1 <= config.port <= 65535):
        raise ValueError("G1_ATOM_PORT must be between 1 and 65535")
    if config.real_enabled:
        if config.host not in {"127.0.0.1", "::1", "localhost"}:
            raise ValueError("real mode requires a loopback-only G1_ATOM_HOST")
        if not config.network_interface:
            raise ValueError("real mode requires G1_ATOM_NETWORK_INTERFACE")
    return config
