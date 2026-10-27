"""Formal research commands run only in the user-designated server directory."""
from __future__ import annotations

import getpass
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVER_BASE = Path('/home/HDD/upc/lwj').resolve()


def require_server(path: Path | None = None) -> Path:
    where = ROOT if path is None else Path(path).resolve()
    if getpass.getuser() != 'upc' or not ROOT.is_relative_to(SERVER_BASE):
        raise RuntimeError('正式实验仅在 upc 用户的 /home/HDD/upc/lwj 项目目录运行。')
    if not where.is_relative_to(ROOT):
        raise RuntimeError('输出路径须位于本项目目录内。')
    return where


def sha(path: Path) -> str:
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path: Path, value: object) -> None:
    path = require_server(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()
