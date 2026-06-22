"""Quick script to check what get_logs() returns."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import database as db
import json

db.init_db()
logs = db.get_logs()
print(f"Total logs: {len(logs)}")
for log in logs[:3]:
    print(f"\n  id={log['id']} status={log['response_status']} events={len(log['events'])}")
    for e in log["events"]:
        print(f"    stage={e['stage']} ts={e['timestamp']} status_code={e['status_code']}")

# Print what the API would return as JSON for the first log
if logs:
    print("\n\nJSON sample (first log):")
    print(json.dumps(logs[0], indent=2)[:2000])
