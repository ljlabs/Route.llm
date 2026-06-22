"""
Seed providers and model mappings into a running model_router instance via its REST API.

Usage:
    python scripts/seed_providers.py [--base-url http://localhost:8000] [--clear]

Reads providers.json and model_mappings.json from the same directory and
POSTs each entry to the appropriate API endpoint.

Note: Model mappings reference providers by ID. Since IDs are auto-assigned
on insert, the mapping provider_id values assume providers are inserted in
manifest order starting from 1 (i.e. you used --clear first on a fresh DB).
"""

import json
import sys
import os
import argparse
import requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PROVIDERS = os.path.join(SCRIPT_DIR, "providers.json")
DEFAULT_MAPPINGS = os.path.join(SCRIPT_DIR, "model_mappings.json")
DEFAULT_BASE_URL = "http://localhost:8000"


def seed_providers(base_url, manifest_path, clear=False):
    api_url = f"{base_url}/api/providers"

    with open(manifest_path, "r") as f:
        providers = json.load(f)

    print(f"Loaded {len(providers)} providers from {manifest_path}")
    print(f"Target: {api_url}\n")

    if clear:
        print("Clearing existing providers...")
        resp = requests.get(api_url)
        resp.raise_for_status()
        existing = resp.json()
        for p in existing:
            del_resp = requests.delete(f"{api_url}/{p['id']}")
            if del_resp.ok:
                print(f"  Deleted: {p['name']} (id={p['id']})")
            else:
                print(f"  Failed to delete {p['name']}: {del_resp.status_code}")
        print()

    success = 0
    failed = 0
    for provider in providers:
        resp = requests.post(api_url, json=provider)
        if resp.ok:
            print(f"  \u2713 {provider['name']} ({provider['model_name']})")
            success += 1
        else:
            print(f"  \u2717 {provider['name']} \u2014 {resp.status_code}: {resp.text}")
            failed += 1

    print(f"\nProviders: {success} created, {failed} failed")
    return failed


def seed_mappings(base_url, manifest_path, clear=False):
    api_url = f"{base_url}/api/routing"

    if not os.path.exists(manifest_path):
        print(f"No mappings manifest found at {manifest_path}, skipping.")
        return 0

    with open(manifest_path, "r") as f:
        mappings = json.load(f)

    print(f"\nLoaded {len(mappings)} model mappings from {manifest_path}")
    print(f"Target: {api_url}\n")

    if clear:
        print("Clearing existing model mappings...")
        resp = requests.get(api_url)
        resp.raise_for_status()
        existing = resp.json()
        for m in existing:
            del_resp = requests.delete(f"{api_url}/{m['model_id']}")
            if del_resp.ok:
                print(f"  Deleted: {m['model_id']}")
            else:
                print(f"  Failed to delete {m['model_id']}: {del_resp.status_code}")
        print()

    success = 0
    failed = 0
    for mapping in mappings:
        resp = requests.post(api_url, json=mapping)
        if resp.ok:
            print(f"  \u2713 {mapping['model_id']} \u2192 provider_id={mapping['provider_id']}")
            success += 1
        else:
            print(f"  \u2717 {mapping['model_id']} \u2014 {resp.status_code}: {resp.text}")
            failed += 1

    print(f"\nMappings: {success} created, {failed} failed")
    return failed


def main():
    parser = argparse.ArgumentParser(description="Seed providers and model mappings into model_router")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Base URL of the running router")
    parser.add_argument("--providers", default=DEFAULT_PROVIDERS, help="Path to providers.json")
    parser.add_argument("--mappings", default=DEFAULT_MAPPINGS, help="Path to model_mappings.json")
    parser.add_argument("--clear", action="store_true", help="Delete all existing data first")
    args = parser.parse_args()

    base = args.base_url.rstrip("/")

    total_failures = 0
    total_failures += seed_providers(base, args.providers, clear=args.clear)
    total_failures += seed_mappings(base, args.mappings, clear=args.clear)

    print("\n" + "=" * 40)
    if total_failures == 0:
        print("All done — seed complete.")
    else:
        print(f"Completed with {total_failures} failure(s).")

    return 0 if total_failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
