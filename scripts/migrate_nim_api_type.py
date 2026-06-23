"""
One-time migration: set api_type = 'nvidia_nim' for all providers
pointing at integrate.api.nvidia.com that are still typed as 'openai'.
"""
import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "proxy.db")

conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

cursor.execute(
    "SELECT id, name, api_type FROM providers WHERE endpoint_url LIKE ?",
    ("%integrate.api.nvidia.com%",)
)
rows = cursor.fetchall()

print("Before migration:")
for r in rows:
    print(f"  id={r['id']}  name={r['name']}  api_type={r['api_type']}")

cursor.execute(
    "UPDATE providers SET api_type = 'nvidia_nim' "
    "WHERE endpoint_url LIKE ? AND api_type = 'openai'",
    ("%integrate.api.nvidia.com%",)
)
updated = cursor.rowcount
conn.commit()

cursor.execute(
    "SELECT id, name, api_type FROM providers WHERE endpoint_url LIKE ?",
    ("%integrate.api.nvidia.com%",)
)
rows = cursor.fetchall()

print(f"\nUpdated {updated} row(s).")
print("After migration:")
for r in rows:
    print(f"  id={r['id']}  name={r['name']}  api_type={r['api_type']}")

conn.close()
