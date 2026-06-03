"""
Standalone test for SFIS _2A_URL and _PVS_URL endpoints.
_TRAVELER_URL is already confirmed working — this script isolates the other two.

Usage:
    python test/test_sfis_2a_pvs.py

Edit the CONFIG block below before running.
"""

import sys
import json
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests
from src.sfis import (
    get_session,
    check_connectivity,
    _2A_URL,
    _PVS_URL,
    query_2a_defects,
    query_pvs,
)

# ------------------------------------------------------------------
# CONFIG — edit these before running
# ------------------------------------------------------------------

# For 2A test: MUST include HH:MM (server passes directly into Oracle To_date).
# Use today: 2026/06/03 00:00 → 2026/06/03 23:59
TEST_2A_FROM_DATE   = "2026/06/03 00:00"
TEST_2A_TO_DATE     = "2026/06/03 23:59"
TEST_2A_MODEL_NAME  = ""        # leave "" for all models
TEST_2A_GROUP_NAME  = "ALL"
TEST_2A_LINE_NAME   = ""

# For PVS test: MUST provide at least a SN — empty query = full table scan = timeout
# Date format: 'YYYY/MM/DD' (no HH:MM needed, unlike 2A)
TEST_PVS_SN         = "HMHHTX00E960000LQ7"   # confirmed working SN from browser test
TEST_PVS_LOCATION   = "U7000"                 # component location
TEST_PVS_FROM_DATE  = "2026/06/02"
TEST_PVS_TO_DATE    = "2026/06/03"
TEST_PVS_MODEL_NAME = ""        # leave "" for all models

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

SEP = "=" * 60

def print_section(title: str):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)



def dump_parsed_tables(tables: dict):
    """Print a summary of all parsed tables."""
    if not tables:
        print("  !! No tables parsed — tableContents may be empty or malformed")
        return
    print(f"  Parsed {len(tables)} table(s):")
    for name, rows in tables.items():
        cols = list(rows[0].keys()) if rows else []
        print(f"    [{name}]  {len(rows)} row(s)  cols={cols}")
        for i, row in enumerate(rows[:3]):   # show first 3 rows
            print(f"      ROW {i+1}: {dict(list(row.items())[:6])}")  # first 6 fields
        if len(rows) > 3:
            print(f"      ... {len(rows)-3} more rows")


# ------------------------------------------------------------------
# Test 1 — Raw 2A request (bypass query_2a_defects, hit URL directly)
# ------------------------------------------------------------------

def test_2a_raw(session: requests.Session):
    print_section("TEST 1: Raw _2A_URL request")
    print(f"  URL: {_2A_URL}")
    print(f"  Date range: {TEST_2A_FROM_DATE} → {TEST_2A_TO_DATE}")

    # Matches the confirmed-working URL exactly
    params = {
        "profitCenter": "0000000025",
        "projectVersion": "ALL",
        "fromDate": TEST_2A_FROM_DATE,
        "toDate": TEST_2A_TO_DATE,
        "BU": "", "Customer": "", "buildEvent": "", "family": "",
        "buildConfig": "", "MO": "ALL", "modelSerial": "",
        "modelName": TEST_2A_MODEL_NAME, "lotNo": "", "bigLot": "",
        "testStation": "ALL", "lineName": TEST_2A_LINE_NAME,
        "groupName": TEST_2A_GROUP_NAME, "errorCode": "",
        "majorProject": "", "projectName": "", "productName": "",
        "retestSequence": "FIRST", "recordType": "ALL",
        "empNo": "ALL", "processType": "ALL",
        "cbxSerialNumber": "Yes",
        "cbxGroupName": "Yes",
        "cbxStationName": "Yes",
        "cbxTestTime": "Yes",
        "cbxErrorCode": "Yes",
        "cbxErrorMessage": "Yes",
        "group_name": TEST_2A_GROUP_NAME, "mo_number": "ALL",
    }
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
        "X-Requested-With": "XMLHttpRequest",
    }

    try:
        r = session.get(_2A_URL, params=params, headers=headers, timeout=20)
    except Exception as e:
        print(f"  !! Request failed: {e}")
        return

    ct = r.headers.get("Content-Type", "")
    print(f"  HTTP {r.status_code}  Content-Type: {ct}")
    if "application/json" not in ct:
        print(f"  !! Not JSON — first 500 chars: {r.text[:500]}")
        return

    data = r.json()
    print(f"  Top-level keys: {list(data.keys())}")

    if "error" in data:
        print(f"  !! Server error: {data['error']}")
        print(f"  SQL: {data.get('sql', '')[:300]}")
        return

    records = data.get("data", [])
    total   = data.get("totalcount", 0)
    print(f"  totalcount={total}  records in response={len(records)}")
    for i, rec in enumerate(records[:5]):
        print(f"  RECORD {i+1}: {json.dumps(rec, indent=4)}")
    if len(records) > 5:
        print(f"  ... {len(records)-5} more records")


# ------------------------------------------------------------------
# Test 2 — query_2a_defects() high-level function
# ------------------------------------------------------------------

def test_2a_highlevel():
    print_section("TEST 2: query_2a_defects() high-level function")
    result = query_2a_defects(
        from_date=TEST_2A_FROM_DATE,
        to_date=TEST_2A_TO_DATE,
        model_name=TEST_2A_MODEL_NAME,
        line_name=TEST_2A_LINE_NAME,
        group_name=TEST_2A_GROUP_NAME,
    )
    print(result[:2000])
    if len(result) > 2000:
        print(f"\n  ... (truncated, total {len(result)} chars)")


# ------------------------------------------------------------------
# Test 3 — Raw PVS request (bypass query_pvs, hit URL directly)
# ------------------------------------------------------------------

def test_pvs_raw(session: requests.Session):
    print_section("TEST 3: Raw _PVS_URL request")
    if not TEST_PVS_SN and not TEST_PVS_MODEL_NAME:
        print("  SKIPPED — set TEST_PVS_SN (or TEST_PVS_MODEL_NAME) in the CONFIG block.")
        print("  Querying with no filters causes a full table scan and times out.")
        return
    print(f"  URL: {_PVS_URL}")
    print(f"  SN={TEST_PVS_SN!r}  location={TEST_PVS_LOCATION!r}  model={TEST_PVS_MODEL_NAME!r}")

    payload = {
        "projectver": "",
        "fromDate": TEST_PVS_FROM_DATE,
        "toDate": TEST_PVS_TO_DATE,
        "buildevent": "",
        "modelname": TEST_PVS_MODEL_NAME,
        "family": "",
        "sn": TEST_PVS_SN,
        "config": "",
        "comppn": "",
        "location": TEST_PVS_LOCATION,
        "mo": "",
        "carton_no": "",
    }
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        "X-Requested-With": "XMLHttpRequest",
    }

    try:
        r = session.post(_PVS_URL, data=payload, headers=headers, timeout=30)
    except Exception as e:
        print(f"  !! Request failed: {e}")
        return

    ct = r.headers.get("Content-Type", "")
    print(f"  HTTP {r.status_code}  Content-Type: {ct}")
    if "application/json" not in ct:
        print(f"  !! Not JSON — first 500 chars:")
        print(f"  {r.text[:500]}")
        return

    data = r.json()
    print(f"  Top-level keys: {list(data.keys())}")
    records = data.get("data", [])
    print(f"  Records returned: {len(records)}")
    for i, item in enumerate(records[:5]):
        print(f"  RECORD {i+1}: {json.dumps(item, indent=4)}")
    if len(records) > 5:
        print(f"  ... {len(records)-5} more records")


# ------------------------------------------------------------------
# Test 4 — query_pvs() high-level function
# ------------------------------------------------------------------

def test_pvs_highlevel():
    print_section("TEST 4: query_pvs() high-level function")
    if not TEST_PVS_SN and not TEST_PVS_MODEL_NAME:
        print("  SKIPPED — set TEST_PVS_SN (or TEST_PVS_MODEL_NAME) in the CONFIG block.")
        return
    result = query_pvs(
        sn=TEST_PVS_SN,
        location=TEST_PVS_LOCATION,
        model_name=TEST_PVS_MODEL_NAME,
        from_date=TEST_PVS_FROM_DATE,
        to_date=TEST_PVS_TO_DATE,
    )
    print(result[:2000])
    if len(result) > 2000:
        print(f"\n  ... (truncated, total {len(result)} chars)")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

if __name__ == "__main__":
    print("SFIS 2A + PVS endpoint test")
    print(f"Server: http://10.52.1.9")

    if not check_connectivity():
        print("!! SFIS server not reachable — aborting.")
        sys.exit(1)

    print("\nLogging in ...")
    try:
        session = get_session()
        print("Login OK")
    except Exception as e:
        print(f"!! Login failed: {e}")
        sys.exit(1)

    try:
        test_2a_raw(session)
        test_pvs_raw(session)
    finally:
        session.close()
        print("\nSession closed.")

    # High-level functions open their own sessions
    test_2a_highlevel()
    test_pvs_highlevel()

    print(f"\n{SEP}")
    print("  Done.")
    print(SEP)
