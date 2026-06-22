"""
Integration test for SQLite concurrent write handling.

Verifies that multiple threads/connections writing to the database
simultaneously do not produce "database is locked" errors, thanks to
WAL mode and the connection timeout.
"""

import os
import sys
import threading
import time
import sqlite3
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import database as db


@pytest.fixture(autouse=True)
def fresh_db(tmp_path, monkeypatch):
    """Point DB_PATH at a temp file and init."""
    test_db = str(tmp_path / "concurrency_test.db")
    monkeypatch.setattr(db, "DB_PATH", test_db)
    db.init_db()
    yield test_db


class TestConcurrentWrites:
    """Simulate multiple concurrent writers hitting the DB — the scenario
    that caused 'database is locked' when seeding providers + mappings."""

    def test_concurrent_provider_inserts(self, fresh_db):
        """50 threads each inserting a provider simultaneously."""
        errors = []
        barrier = threading.Barrier(50, timeout=10)

        def insert_provider(idx):
            try:
                barrier.wait()  # All threads start at the same time
                db.add_provider(
                    name=f"Provider-{idx}",
                    api_type="openai",
                    endpoint_url=f"http://localhost:900{idx}/v1/chat",
                    api_key=f"sk-{idx}",
                    model_name=f"model-{idx}",
                    is_active=0,
                )
            except Exception as e:
                errors.append((idx, str(e)))

        threads = [threading.Thread(target=insert_provider, args=(i,)) for i in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)

        assert errors == [], f"Concurrent inserts failed: {errors}"

        providers = db.get_providers()
        assert len(providers) == 50

    def test_concurrent_log_lifecycle_writes(self, fresh_db):
        """Multiple request lifecycles writing events concurrently."""
        errors = []
        num_requests = 20
        barrier = threading.Barrier(num_requests, timeout=10)

        def simulate_request(idx):
            try:
                barrier.wait()
                # Stage 1
                request_id = db.start_request_log(
                    provider_name=f"Provider-{idx}",
                    request_method="POST",
                    request_path="/v1/chat/completions",
                    request_body=f'{{"request":{idx}}}',
                )
                # Stage 2
                db.add_log_event(request_id, "provider_request", body=f'{{"wrapped":{idx}}}')
                # Simulate slight delay like a real HTTP call
                time.sleep(0.01)
                # Stage 3
                db.add_log_event(
                    request_id, "provider_response",
                    body=f'{{"response":{idx}}}',
                    status_code=200,
                )
                # Stage 4
                db.complete_request_log(
                    request_id=request_id,
                    response_status=200,
                    response_body=f'{{"final":{idx}}}',
                    tokens_sent=10,
                    tokens_received=20,
                    latency_ms=100,
                )
            except Exception as e:
                errors.append((idx, str(e)))

        threads = [threading.Thread(target=simulate_request, args=(i,)) for i in range(num_requests)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=20)

        assert errors == [], f"Concurrent lifecycle writes failed: {errors}"

        logs = db.get_logs()
        assert len(logs) == num_requests

        # Each log should have exactly 4 events
        for log in logs:
            assert len(log["events"]) == 4, (
                f"Log id={log['id']} has {len(log['events'])} events, expected 4"
            )

    def test_concurrent_reads_and_writes(self, fresh_db):
        """Writers and readers operating simultaneously don't deadlock."""
        errors = []
        stop_event = threading.Event()

        # Pre-seed some data
        for i in range(5):
            db.add_provider(f"Pre-{i}", "openai", f"http://x/{i}", "k", f"m-{i}")

        def writer(idx):
            try:
                for j in range(10):
                    if stop_event.is_set():
                        return
                    db.add_provider(
                        name=f"Writer-{idx}-{j}",
                        api_type="openai",
                        endpoint_url=f"http://w/{idx}/{j}",
                        api_key="k",
                        model_name=f"w-{idx}-{j}",
                    )
                    time.sleep(0.005)
            except Exception as e:
                errors.append(("writer", idx, str(e)))

        def reader(idx):
            try:
                for _ in range(20):
                    if stop_event.is_set():
                        return
                    db.get_providers()
                    db.get_logs()
                    time.sleep(0.002)
            except Exception as e:
                errors.append(("reader", idx, str(e)))

        writers = [threading.Thread(target=writer, args=(i,)) for i in range(5)]
        readers = [threading.Thread(target=reader, args=(i,)) for i in range(10)]

        all_threads = writers + readers
        for t in all_threads:
            t.start()
        for t in all_threads:
            t.join(timeout=15)

        stop_event.set()
        assert errors == [], f"Concurrent read/write errors: {errors}"

    def test_held_connection_does_not_block_indefinitely(self, fresh_db):
        """A long-held read connection shouldn't block writers forever with WAL mode."""
        errors = []

        def hold_read_connection():
            """Open a connection and hold a read transaction for 2 seconds."""
            conn = sqlite3.connect(fresh_db, timeout=10)
            conn.execute("PRAGMA journal_mode = WAL")
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM providers")
            cursor.fetchall()
            # Hold the connection open (simulates a slow reader)
            time.sleep(2)
            conn.close()

        def try_write():
            """Try to write while a reader is holding a connection."""
            time.sleep(0.2)  # Let the reader start first
            try:
                db.add_provider("WriteWhileRead", "openai", "http://x", "k", "m")
            except Exception as e:
                errors.append(str(e))

        reader_thread = threading.Thread(target=hold_read_connection)
        writer_thread = threading.Thread(target=try_write)

        reader_thread.start()
        writer_thread.start()

        reader_thread.join(timeout=5)
        writer_thread.join(timeout=5)

        assert errors == [], f"Write blocked by reader: {errors}"
        providers = db.get_providers()
        assert any(p["name"] == "WriteWhileRead" for p in providers)
