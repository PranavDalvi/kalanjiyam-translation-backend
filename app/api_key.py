import os
import sqlite3
import hashlib
import secrets
import logging
from typing import Optional, List, Dict, Tuple
from datetime import datetime
from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader

logger = logging.getLogger("translation_backend.api_key")

DB_PATH_ENV = "API_KEY_DB_PATH"
DEFAULT_DB_PATH = "api_keys.db"

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def get_db_path() -> str:
    return os.environ.get(DB_PATH_ENV, DEFAULT_DB_PATH)


def init_db(db_path: Optional[str] = None) -> None:
    path = db_path or get_db_path()
    with sqlite3.connect(path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS api_keys (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key_prefix TEXT NOT NULL,
                key_hash TEXT NOT NULL UNIQUE,
                name TEXT NOT NULL,
                created_at TEXT NOT NULL,
                is_active INTEGER NOT NULL DEFAULT 1
            )
        """)
        conn.commit()


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def generate_api_key(name: str, db_path: Optional[str] = None) -> Tuple[str, str]:
    """
    Generates a secure API key, stores its SHA-256 hash in SQLite, and returns (raw_key, prefix).
    Raw key format: kt_<32 hex chars>
    """
    path = db_path or get_db_path()
    init_db(path)

    raw_token = secrets.token_hex(16)
    raw_key = f"kt_{raw_token}"
    prefix = raw_key[:7]
    key_h = hash_key(raw_key)
    created_at = datetime.utcnow().isoformat()

    with sqlite3.connect(path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO api_keys (key_prefix, key_hash, name, created_at, is_active) VALUES (?, ?, ?, ?, 1)",
            (prefix, key_h, name, created_at)
        )
        conn.commit()

    return raw_key, prefix


def verify_key(raw_key: str, db_path: Optional[str] = None) -> bool:
    if not raw_key:
        return False
    path = db_path or get_db_path()
    if not os.path.exists(path):
        return False

    key_h = hash_key(raw_key)
    with sqlite3.connect(path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id FROM api_keys WHERE key_hash = ? AND is_active = 1",
            (key_h,)
        )
        row = cursor.fetchone()
        return row is not None


def list_api_keys(db_path: Optional[str] = None) -> List[Dict[str, str]]:
    path = db_path or get_db_path()
    if not os.path.exists(path):
        return []

    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT id, key_prefix, name, created_at, is_active FROM api_keys ORDER BY id ASC")
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


def revoke_api_key(prefix_or_id: str, db_path: Optional[str] = None) -> bool:
    path = db_path or get_db_path()
    if not os.path.exists(path):
        return False

    with sqlite3.connect(path) as conn:
        cursor = conn.cursor()
        if prefix_or_id.isdigit():
            cursor.execute("UPDATE api_keys SET is_active = 0 WHERE id = ?", (int(prefix_or_id),))
        else:
            cursor.execute("UPDATE api_keys SET is_active = 0 WHERE key_prefix = ?", (prefix_or_id,))
        updated = cursor.rowcount > 0
        conn.commit()
        return updated


async def verify_api_key_dependency(api_key: Optional[str] = Security(API_KEY_HEADER)) -> Optional[str]:
    enable_auth = os.environ.get("ENABLE_API_KEY_AUTH", "0").lower() in ("1", "true", "yes")
    if not enable_auth:
        return None

    if not api_key or not verify_key(api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key. Include 'X-API-Key' header.",
        )
    return api_key
