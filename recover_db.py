#!/usr/bin/env python3
"""
SQLite Database Recovery Script

Copies providers, model_mappings, and settings from the old (bloated) proxy.db
to a fresh new database, skipping request logs entirely.

Usage:
    python recover_db.py
    python recover_db.py --old path/to/old.db --new path/to/new.db
"""

import argparse
import sqlite3
import os
import sys
import shutil
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "proxy.db")


def create_new_db(db_path: str) -> sqlite3.Connection:
    """Create a fresh database with the correct schema."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS providers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            api_type TEXT NOT NULL,
            endpoint_url TEXT NOT NULL,
            api_key TEXT NOT NULL,
            model_name TEXT NOT NULL,
            is_active INTEGER DEFAULT 0,
            rate_limit_tps REAL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_mappings (
            model_id TEXT PRIMARY KEY,
            provider_id INTEGER NOT NULL,
            FOREIGN KEY (provider_id) REFERENCES providers (id) ON DELETE CASCADE
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            provider_name TEXT,
            request_method TEXT,
            request_path TEXT,
            request_body TEXT,
            response_status INTEGER,
            response_body TEXT,
            tokens_sent INTEGER DEFAULT 0,
            tokens_received INTEGER DEFAULT 0,
            latency_ms INTEGER DEFAULT 0
        )
    """)

    conn.commit()
    return conn


def recover(old_db_path: str, new_db_path: str):
    if not os.path.exists(old_db_path):
        print(f"ERROR: Old database not found: {old_db_path}")
        sys.exit(1)

    old_size = os.path.getsize(old_db_path)
    print(f"Old database: {old_db_path} ({old_size / 1024 / 1024:.1f} MB)")

    if os.path.exists(new_db_path):
        backup = new_db_path + f".bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        print(f"Backing up existing new DB to: {backup}")
        shutil.copy2(new_db_path, backup)

    old_conn = sqlite3.connect(old_db_path)
    old_conn.row_factory = sqlite3.Row

    new_conn = create_new_db(new_db_path)
    new_cursor = new_conn.cursor()

    # --- Copy providers ---
    old_providers = old_conn.execute("SELECT * FROM providers").fetchall()
    provider_id_map = {}  # old_id -> new_id

    print(f"\nCopying {len(old_providers)} providers...")
    for row in old_providers:
        d = dict(row)
        new_cursor.execute("""
            INSERT INTO providers (name, api_type, endpoint_url, api_key, model_name, is_active, rate_limit_tps)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (d["name"], d["api_type"], d["endpoint_url"], d["api_key"],
              d["model_name"], d["is_active"], d["rate_limit_tps"]))
        new_id = new_cursor.lastrowid
        provider_id_map[d["id"]] = new_id
        status = " (ACTIVE)" if d["is_active"] else ""
        print(f"  [{d['id']} -> {new_id}] {d['name']} ({d['api_type']}) - {d['model_name']}{status}")

    new_conn.commit()

    # --- Copy settings ---
    old_settings = old_conn.execute("SELECT * FROM settings").fetchall()
    print(f"\nCopying {len(old_settings)} settings...")
    for row in old_settings:
        d = dict(row)
        new_cursor.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)",
                           (d["key"], d["value"]))
        print(f"  {d['key']} = {d['value']}")

    new_conn.commit()

    # --- Copy model_mappings (with ID remapping, skip dangling refs) ---
    old_mappings = old_conn.execute("SELECT * FROM model_mappings").fetchall()
    print(f"\nCopying {len(old_mappings)} model mappings...")
    skipped = 0
    for row in old_mappings:
        d = dict(row)
        old_pid = d["provider_id"]
        if old_pid in provider_id_map:
            new_pid = provider_id_map[old_pid]
            new_cursor.execute("INSERT OR REPLACE INTO model_mappings (model_id, provider_id) VALUES (?, ?)",
                               (d["model_id"], new_pid))
            print(f"  '{d['model_id']}' -> provider {old_pid} -> {new_pid}")
        else:
            skipped += 1
            print(f"  SKIPPED '{d['model_id']}' (old provider_id {old_pid} not found)")

    new_conn.commit()

    if skipped:
        print(f"\n  {skipped} dangling mapping(s) skipped")

    new_conn.close()
    old_conn.close()

    new_size = os.path.getsize(new_db_path)
    print(f"\nRecovery complete!")
    print(f"  New database: {new_db_path} ({new_size / 1024:.1f} KB)")
    print(f"  Size reduction: {old_size / 1024 / 1024:.1f} MB -> {new_size / 1024:.1f} KB")
    print(f"\nNext steps:")
    print(f"  1. Stop the server")
    print(f"  2. Rename old DB: ren proxy.db proxy.db.old")
    print(f"  3. Rename new DB: ren {os.path.basename(new_db_path)} proxy.db")
    print(f"  4. Restart the server")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recover proxy.db to a clean database")
    parser.add_argument("--old", default=DB_PATH, help="Path to old database")
    parser.add_argument("--new", default=None, help="Path to new database (default: proxy_recovered.db)")
    args = parser.parse_args()

    new_path = args.new or os.path.join(os.path.dirname(args.old), "proxy_recovered.db")
    recover(args.old, new_path)
