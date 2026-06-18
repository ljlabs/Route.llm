import sqlite3
import json
from datetime import datetime
import os

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "proxy.db")

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Drop existing providers if schema mismatch or just to recreate with endpoint_url
    try:
        cursor.execute("SELECT base_url FROM providers LIMIT 1")
        # If no exception, it has the old schema. Drop it to update.
        cursor.execute("DROP TABLE IF EXISTS providers")
    except sqlite3.OperationalError:
        pass
        
    # Create providers table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS providers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            api_type TEXT NOT NULL, -- 'openai' or 'anthropic'
            endpoint_url TEXT NOT NULL,
            api_key TEXT NOT NULL,
            model_name TEXT NOT NULL,
            is_active INTEGER DEFAULT 0,
            rate_limit_tps REAL,
            max_tokens INTEGER
        )
    """)

    # Migration: Add rate_limit_tps if it doesn't exist (for existing databases)
    try:
        cursor.execute("ALTER TABLE providers ADD COLUMN rate_limit_tps REAL")
    except sqlite3.OperationalError:
        # Column already exists
        pass

    # Migration: Add max_tokens if it doesn't exist (for existing databases)
    try:
        cursor.execute("ALTER TABLE providers ADD COLUMN max_tokens INTEGER")
    except sqlite3.OperationalError:
        # Column already exists
        pass

    # Create model mappings table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_mappings (
            model_id TEXT PRIMARY KEY,
            provider_id INTEGER NOT NULL,
            FOREIGN KEY (provider_id) REFERENCES providers (id) ON DELETE CASCADE
        )
    """)

    # Create settings table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
    """)
    
    # Create logs table
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

    # Migration: Add metrics columns if they don't exist
    for col in ["tokens_sent", "tokens_received", "latency_ms"]:
        try:
            cursor.execute(f"ALTER TABLE logs ADD COLUMN {col} INTEGER DEFAULT 0")
        except sqlite3.OperationalError:
            # Column already exists
            pass
    
    # Insert default settings if not exist
    cursor.execute("INSERT OR IGNORE INTO settings (key, value) VALUES ('log_limit', '50')")
    cursor.execute("INSERT OR IGNORE INTO settings (key, value) VALUES ('rate_limit_tps', '0')") # 0 means disabled
    cursor.execute("INSERT OR IGNORE INTO settings (key, value) VALUES ('max_tokens', '32000')")

    conn.commit()
    conn.close()

# Provider operations
def get_providers():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM providers")
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def get_active_provider():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM providers WHERE is_active = 1 LIMIT 1")
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def add_provider(name, api_type, endpoint_url, api_key, model_name, is_active=0, rate_limit_tps=None, max_tokens=None):
    conn = get_db_connection()
    cursor = conn.cursor()
    if is_active == 1:
        cursor.execute("UPDATE providers SET is_active = 0")
    cursor.execute("""
        INSERT INTO providers (name, api_type, endpoint_url, api_key, model_name, is_active, rate_limit_tps, max_tokens)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (name, api_type, endpoint_url, api_key, model_name, is_active, rate_limit_tps, max_tokens))
    conn.commit()
    conn.close()

def update_provider(provider_id, name, api_type, endpoint_url, api_key, model_name, is_active, rate_limit_tps=None, max_tokens=None):
    conn = get_db_connection()
    cursor = conn.cursor()
    if is_active == 1:
        cursor.execute("UPDATE providers SET is_active = 0 WHERE id != ?", (provider_id,))
    cursor.execute("""
        UPDATE providers
        SET name = ?, api_type = ?, endpoint_url = ?, api_key = ?, model_name = ?, is_active = ?, rate_limit_tps = ?, max_tokens = ?
        WHERE id = ?
    """, (name, api_type, endpoint_url, api_key, model_name, is_active, rate_limit_tps, max_tokens, provider_id))
    conn.commit()
    conn.close()

def delete_provider(provider_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    # Check if we're deleting the active one
    cursor.execute("SELECT is_active FROM providers WHERE id = ?", (provider_id,))
    row = cursor.fetchone()
    is_active = row['is_active'] if row else 0
    
    cursor.execute("DELETE FROM providers WHERE id = ?", (provider_id,))
    
    if is_active == 1:
        # Activate the next available one
        cursor.execute("UPDATE providers SET is_active = 1 WHERE id = (SELECT id FROM providers LIMIT 1)")
        
    conn.commit()
    conn.close()

def set_active_provider(provider_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE providers SET is_active = 0")
    cursor.execute("UPDATE providers SET is_active = 1 WHERE id = ?", (provider_id,))
    conn.commit()
    conn.close()

# Model Mapping operations
def get_model_mappings():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT m.model_id, p.name as provider_name, p.id as provider_id
        FROM model_mappings m
        JOIN providers p ON m.provider_id = p.id
    """)
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def add_model_mapping(model_id, provider_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT OR REPLACE INTO model_mappings (model_id, provider_id) VALUES (?, ?)", (model_id, provider_id))
    conn.commit()
    conn.close()

def delete_model_mapping(model_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM model_mappings WHERE model_id = ?", (model_id,))
    conn.commit()
    conn.close()

# Settings operations
def get_log_limit():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT value FROM settings WHERE key = 'log_limit'")
    row = cursor.fetchone()
    conn.close()
    return int(row['value']) if row else 50

def get_rate_limit_tps():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT value FROM settings WHERE key = 'rate_limit_tps'")
    row = cursor.fetchone()
    conn.close()
    return float(row['value']) if row else 0.0

def set_log_limit(limit):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT OR REPLACE INTO settings (key, value) VALUES ('log_limit', ?)", (str(limit),))
    conn.commit()
    conn.close()
    # Enforce limit immediately
    enforce_log_limit()

def set_rate_limit_tps(tps):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT OR REPLACE INTO settings (key, value) VALUES ('rate_limit_tps', ?)", (str(tps),))
    conn.commit()
    conn.close()

def get_max_tokens():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT value FROM settings WHERE key = 'max_tokens'")
    row = cursor.fetchone()
    conn.close()
    return int(row['value']) if row else 32000

def set_max_tokens(max_tokens):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT OR REPLACE INTO settings (key, value) VALUES ('max_tokens', ?)", (str(max_tokens),))
    conn.commit()
    conn.close()

# Log operations
def add_log(provider_name, request_method, request_path, request_body, response_status, response_body, tokens_sent=0, tokens_received=0, latency_ms=0):
    limit = get_log_limit()
    if limit == -1:
        return # Logging is disabled

    conn = get_db_connection()
    cursor = conn.cursor()
    timestamp = datetime.now().isoformat()
    cursor.execute("""
        INSERT INTO logs (timestamp, provider_name, request_method, request_path, request_body, response_status, response_body, tokens_sent, tokens_received, latency_ms)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (timestamp, provider_name, request_method, request_path, request_body, response_status, response_body, tokens_sent, tokens_received, latency_ms))
    conn.commit()
    conn.close()

    enforce_log_limit()

def enforce_log_limit():
    limit = get_log_limit()
    if limit == -1:
        # Delete all logs if disabled
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM logs")
        conn.commit()
        conn.close()
        return
        
    conn = get_db_connection()
    cursor = conn.cursor()
    # Get current log count
    cursor.execute("SELECT COUNT(*) as count FROM logs")
    count = cursor.fetchone()['count']
    if count > limit:
        to_delete = count - limit
        cursor.execute(f"DELETE FROM logs WHERE id IN (SELECT id FROM logs ORDER BY timestamp ASC LIMIT {to_delete})")
        conn.commit()
    conn.close()

def get_logs():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM logs ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]
def clear_logs():
    """Clear all logs from the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM logs")
    conn.commit()
    conn.close()

def get_metrics_summary():
    """Get aggregated metrics per provider."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT
            provider_name,
            COUNT(*) as request_count,
            SUM(tokens_sent) as total_tokens_sent,
            SUM(tokens_received) as total_tokens_received,
            AVG(latency_ms) as avg_latency
        FROM logs
        GROUP BY provider_name
    """)
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def get_latency_history(limit=50):
    """Get recent latency history for line chart."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT 
            timestamp, 
            provider_name, 
            latency_ms 
        FROM logs 
        WHERE response_status = 200 
        ORDER BY timestamp DESC 
        LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in reversed(rows)]

