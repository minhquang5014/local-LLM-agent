"""
SFIS client — headless version (no Tkinter).

Reads credentials from sfis_cred.json in the project root.
If the file doesn't exist or credentials are wrong, raises SFISAuthError.

Usage:
    from src.sfis import query_sn
    result = query_sn("ABC123456")
    print(result)          # formatted string ready for the LLM
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

SFIS_BASE = "http://10.52.1.9"
LOGIN_URL = f"{SFIS_BASE}/SFIS/Member/resources/func_login.jsp"
_2A_URL = f"{SFIS_BASE}/SFIS/Yield/Manager_2A/resources/getQueryJSON.jsp"
_PVS_URL = f"{SFIS_BASE}/SFIS/PVS-vs-SFIS/SN/resources/getQuery.jsp"
CRED_FILE = Path(__file__).resolve().parent.parent / "sfis_cred.json"

_LOGIN_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
    "X-Requested-With": "XMLHttpRequest",
}

# Station names that indicate functional/burn-in failures
_FAILURE_STATIONS = [
    "burn_in", "dfu", "fct", "cell", "wifi",
    "s-cond", "s_cond", "t269", "t-269",
    "a-cond", "a_cond",
]


class SFISAuthError(Exception):
    pass


# ------------------------------------------------------------------
# Connectivity check
# ------------------------------------------------------------------

def check_connectivity() -> bool:
    """Return True if the SFIS server (10.52.1.9) is reachable."""
    print(f"[SFIS] Checking connectivity to {SFIS_BASE} ...")
    try:
        r = requests.get(SFIS_BASE, timeout=5, allow_redirects=True)
        reachable = r.status_code < 500
        print(f"[SFIS] Server {'reachable' if reachable else 'returned error'} (HTTP {r.status_code})")
        return reachable
    except Exception as e:
        print(f"[SFIS] ERROR — server not reachable: {e}")
        return False


# ------------------------------------------------------------------
# Credentials
# ------------------------------------------------------------------

def load_credentials() -> tuple[str, str]:
    if not CRED_FILE.exists():
        return "", ""
    try:
        data = json.loads(CRED_FILE.read_text(encoding="utf-8"))
        return data.get("username", ""), data.get("password", "")
    except Exception:
        return "", ""


def save_credentials(username: str, password: str) -> None:
    CRED_FILE.write_text(
        json.dumps({"username": username, "password": password}, indent=2),
        encoding="utf-8",
    )


# ------------------------------------------------------------------
# Auth
# ------------------------------------------------------------------

def _login(session: requests.Session, username: str, password: str) -> bool:
    print(f"[SFIS] Authenticating as '{username}' ...")
    try:
        r = session.post(
            LOGIN_URL,
            data={"User_Name": username, "Pass_Word": password, "Func_Name": "LOGIN"},
            headers=_LOGIN_HEADERS,
            timeout=10,
        )
        r.raise_for_status()
        ok = r.json().get("RES") == "OK"
        print(f"[SFIS] Login {'SUCCESS' if ok else 'FAILED — wrong credentials?'}")
        return ok
    except Exception as e:
        print(f"[SFIS] ERROR — login request failed: {e}")
        logger.warning("SFIS login attempt failed: %s", e)
        return False


def get_session() -> requests.Session:
    """Return an authenticated SFIS session, or raise SFISAuthError."""
    username, password = load_credentials()
    if not username or not password:
        print(f"[SFIS] ERROR — no credentials found in {CRED_FILE}")
        raise SFISAuthError(
            "No SFIS credentials found. "
            f"Create '{CRED_FILE}' with {{\"username\": \"...\", \"password\": \"...\"}} "
            "or ask the agent to save them."
        )
    session = requests.Session()
    if not _login(session, username, password):
        session.close()
        raise SFISAuthError("SFIS login failed — check credentials in sfis_cred.json.")
    return session


# ------------------------------------------------------------------
# HTML table parser (shared by both structured and generic paths)
# ------------------------------------------------------------------

def _parse_tables_from_json(json_obj: dict) -> dict[str, list[dict]]:
    """Parse SFIS JSON response into {table_title: [row_dict, ...]}."""
    raw_html = json_obj.get("tableContents", "")
    if not raw_html:
        return {}

    soup = BeautifulSoup(raw_html, "html.parser")
    important_tags = soup.find_all(["font", "h1", "h2", "h3", "h4", "h5", "h6", "table"])

    result: dict = {}
    pending_title: Optional[str] = None
    tbl_index = 0
    name_count: dict = {}

    def make_unique(name: str) -> str:
        key = name.strip()
        if not key:
            return key
        if key not in name_count:
            name_count[key] = 1
            return key
        name_count[key] += 1
        return f"{key}#{name_count[key]}"

    for tag in important_tags:
        tag_name = tag.name.lower()
        if tag_name in ("font", "h1", "h2", "h3", "h4", "h5", "h6"):
            text = tag.get_text(separator=" ", strip=True)
            if text:
                pending_title = text
        elif tag_name == "table":
            tbl_index += 1
            title = pending_title or f"table_{tbl_index}"
            pending_title = None
            unique_title = make_unique(title)

            thead = tag.find("thead")
            tbody = tag.find("tbody")
            headers: list[str] = []
            if thead:
                headers = [td.get_text(strip=True) for td in thead.find_all("td")]
            if not headers or not tbody:
                result[unique_title] = []
                continue

            parsed: list[dict] = []
            rows = tbody.find_all("tr")
            if rows:
                for tr in rows:
                    cols = [td.get_text(strip=True) for td in tr.find_all("td")]
                    if len(cols) == len(headers):
                        parsed.append(dict(zip(headers, cols)))
            else:
                tds = tbody.find_all("td")
                if len(tds) == len(headers):
                    parsed.append(dict(zip(headers, [td.get_text(strip=True) for td in tds])))

            result[unique_title] = parsed

    return result


def _convert_ec_multiline(ec_string: str) -> str:
    return "\n".join(ec.strip() for ec in ec_string.split(";") if ec.strip())


# ------------------------------------------------------------------
# Structured traveler extraction (mirrors test/get_fa_data_on_sfis 6.py)
# ------------------------------------------------------------------

def _extract_structured_fields(tables: dict[str, list[dict]]) -> dict[str, str]:
    """
    Pull the specific fields engineers care about from the parsed tables.
    Returns a flat dict of named fields.
    """
    fields: dict[str, str] = {
        "PHASE": "", "MODEL": "", "CONFIG": "",
        "LINE": "", "PANEL SN": "", "SN SEQ IN PANEL": "",
        "FAILED DATE": "", "LAB IN TIME": "",
        "GROUP NAME": "", "FAILURE MESSAGE": "",
        "LIST OF FAILING TESTS": "",
    }

    print(f"[SFIS] _extract_structured_fields: tables available: {list(tables.keys())}")

    # Phase / Model / Config  ← "Work Order / Model Data"
    model_table = tables.get("Work Order / Model Data", [])
    if not model_table:
        print(f"[SFIS]   Work Order / Model Data: NOT FOUND")
    for row in model_table:
        if row.get("HW BOM"):
            fields["PHASE"] = row.get("VERSION CODE", "")
            fields["MODEL"] = row.get("HW BOM", "")
            fields["CONFIG"] = row.get("SW BOM", "")
            print(f"[SFIS]   PHASE={fields['PHASE']!r}  MODEL={fields['MODEL']!r}  CONFIG={fields['CONFIG']!r}")
            break
    else:
        if model_table:
            print(f"[SFIS]   Work Order / Model Data: no row with HW BOM — first row: {model_table[0]}")

    # SMT Line / Panel SN  ← "SN Detail Data"
    sn_detail = tables.get("SN Detail Data", [])
    if not sn_detail:
        print(f"[SFIS]   SN Detail Data: NOT FOUND")
    for row in sn_detail:
        if row.get("VIRTUAL LINE1"):
            fields["LINE"] = row.get("VIRTUAL LINE1", "")
            fields["PANEL SN"] = row.get("TRACK NO", "")
            print(f"[SFIS]   LINE={fields['LINE']!r}  PANEL SN={fields['PANEL SN']!r}")
            break
    else:
        if sn_detail:
            print(f"[SFIS]   SN Detail Data: no row with VIRTUAL LINE1 — first row: {sn_detail[0]}")

    # SN position in panel  ← "Wip Tracking Data"
    wip_table = tables.get("Wip Tracking Data", [])
    if not wip_table:
        print(f"[SFIS]   Wip Tracking Data: NOT FOUND")
    for row in wip_table:
        seq = row.get("SN SEQ IN PANEL", "")
        if seq:
            fields["SN SEQ IN PANEL"] = seq
            print(f"[SFIS]   SN SEQ IN PANEL={fields['SN SEQ IN PANEL']!r}")
            break
    else:
        if wip_table:
            print(f"[SFIS]   Wip Tracking Data: SN SEQ IN PANEL empty — first row: {wip_table[0]}")

    # Failure date / group / test codes  ← "SN Repair Data"
    repair_table = tables.get("SN Repair Data", [])
    if not repair_table:
        print(f"[SFIS]   SN Repair Data: NOT FOUND")
    _repair_matched = False
    for row in repair_table:
        station = (row.get("TEST STATION") or "").lower()
        if any(s in station for s in _FAILURE_STATIONS):
            test_code = row.get("TEST CODE", "")
            if test_code:
                fields["FAILED DATE"] = row.get("TEST TIME", "")
                fields["GROUP NAME"] = row.get("TEST GROUP", "")
                fields["LIST OF FAILING TESTS"] = test_code
                print(f"[SFIS]   SN Repair match station={station!r}  FAILED DATE={fields['FAILED DATE']!r}  GROUP={fields['GROUP NAME']!r}")
                _repair_matched = True
                break
    if repair_table and not _repair_matched:
        stations_seen = [row.get("TEST STATION", "") for row in repair_table]
        print(f"[SFIS]   SN Repair Data: no matching failure station — stations in table: {stations_seen}")

    # Lab-in time  ← "Laboratory In/Out"
    lab_table = tables.get("Laboratory In/Out", [])
    if not lab_table:
        print(f"[SFIS]   Laboratory In/Out: NOT FOUND")
    for row in lab_table:
        if row.get("LAB IN EMP") and row.get("LAB IN TIME"):
            fields["LAB IN TIME"] = row.get("LAB IN TIME", "")
            print(f"[SFIS]   LAB IN TIME={fields['LAB IN TIME']!r}")
            break
    else:
        if lab_table:
            print(f"[SFIS]   Laboratory In/Out: no row with LAB IN EMP+TIME — first row: {lab_table[0]}")

    # Failure message + refined test list  ← "Bobcat Data"
    bobcat_table = tables.get("Bobcat Data", [])
    failed_date = fields.get("FAILED DATE", "")
    if not bobcat_table:
        print(f"[SFIS]   Bobcat Data: NOT FOUND")
    elif not failed_date:
        print(f"[SFIS]   Bobcat Data: skipped (no FAILED DATE to match against)")
    else:
        try:
            dt = datetime.strptime(failed_date, "%Y/%m/%d %H:%M:%S")
            target_time = dt.strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            target_time = ""
        print(f"[SFIS]   Bobcat Data: looking for FAAP station with STOP TIME={target_time!r}")
        _bobcat_matched = False
        for row in bobcat_table:
            station_id = row.get("STATION ID", "")
            stop_time = row.get("STOP TIME", "")
            failing_tests = row.get("LIST OF FAILING TESTS", "")
            if failing_tests and "FAAP" in station_id and stop_time == target_time:
                fields["FAILURE MESSAGE"] = row.get("FAILURE MESSAGE", "")
                fields["LIST OF FAILING TESTS"] = _convert_ec_multiline(failing_tests)
                print(f"[SFIS]   Bobcat match: station={station_id!r}  FAILURE MESSAGE={fields['FAILURE MESSAGE']!r}")
                _bobcat_matched = True
                break
        if not _bobcat_matched:
            faap_rows = [(r.get("STATION ID", ""), r.get("STOP TIME", "")) for r in bobcat_table if "FAAP" in (r.get("STATION ID") or "")]
            print(f"[SFIS]   Bobcat Data: no FAAP match — FAAP rows (station, stop_time): {faap_rows}")

    # Fallback: if LIST OF FAILING TESTS still has semicolons, expand them
    lot = fields["LIST OF FAILING TESTS"]
    if ";" in lot:
        fields["LIST OF FAILING TESTS"] = _convert_ec_multiline(lot)

    print(f"[SFIS] Extracted fields result: {fields}")
    return fields


# ------------------------------------------------------------------
# Traveler query
# ------------------------------------------------------------------

_TRAVELER_URL = (
    f"{SFIS_BASE}/SFIS/Production/Travelers/Trav_1/resources/getQueryJSON.jsp"
    "?ColItem=serial_number&Field_Kind=ALLFIELD"
    "&fdSerial_Number=Y&fdModel_Serial=Y&fdMo_Number=Y&fdLine_Name=Y"
    "&fdSection_Name=Y&fdGroup_Name=Y&fdStation_Name=Y"
    "&fdIn_Station_Time=Y&fdOut_Station_Time=Y&fdRETEST_SEQ=Y"
    "&fdEmp_No=Y&fdQa_NO=Y&fdQa_Result=Y&fdPallet_No=Y&fdCarton_NO=Y"
    "&fdPO_NO=Y&fdCONTAINER_NO=Y&fdSHIPPING_SN=Y&fdMAC=Y&fdTRACK_NO=Y"
    "&fdKEY_PART_NO=Y&fdMODEL_NAME=Y&fdBill_NO=Y&fdError_flag=Y"
    "&fdFinish_flag=Y&fdVersion_Code=Y&fdSpecial_route=Y&fdCust_model=Y"
    "&fdCust_PN=Y&fdInv_no=Y&fdOther_MaC=Y&fdKP_NO_C=Y&fdMain_Product=Y"
    "&fdProduct_Name=Y&fdBox_No=Y&fdLOTN=Y&fdLOTB=Y&fdOut_Line_Time=Y"
    "&fdDRYBOX=Y&fdVIRTUAL_LINE1=Y&fdVIRTUAL_LINE2=Y&fdPanelSeq=Y"
    "&fdBCadd=Y&fdBCqry=Y&InpData={sn}&FromURL=N"
)

_TRAVELER_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json",
    "X-Requested-With": "XMLHttpRequest",
}


def _query_traveler(session: requests.Session, sn: str) -> tuple[dict[str, str], dict[str, list[dict]]]:
    """
    Fetch and parse the traveler for *sn*.
    Returns (structured_fields, all_tables) — callers can use either.
    """
    url = _TRAVELER_URL.format(sn=sn)
    print(f"[SFIS] Querying traveler for SN: {sn}")
    print(f"[SFIS] Full URL: {url}")
    try:
        r = session.get(url, headers=_TRAVELER_HEADERS, timeout=15)
        print(f"[SFIS] Traveler response: HTTP {r.status_code}, Content-Type: {r.headers.get('Content-Type', 'unknown')}")
    except requests.Timeout:
        print(f"[SFIS] ERROR — traveler query timed out for SN: {sn}")
        raise RuntimeError("SFIS traveler query timed out.")

    if "application/json" not in r.headers.get("Content-Type", ""):
        print(f"[SFIS] ERROR — expected JSON but got: {r.headers.get('Content-Type')} (session may have expired)")
        return {}, {}

    try:
        json_obj = r.json()
        raw_html = json_obj.get("tableContents", "")
        print(f"[SFIS] tableContents length: {len(raw_html)} chars")
        tables = _parse_tables_from_json(json_obj)
        print(f"[SFIS] Parsed {len(tables)} table(s):")
        for tname, rows in tables.items():
            cols = list(rows[0].keys()) if rows else []
            print(f"[SFIS]   '{tname}': {len(rows)} row(s) | cols: {cols}")
    except Exception as e:
        print(f"[SFIS] ERROR — failed to parse JSON response: {e}")
        logger.warning("Failed to parse SFIS JSON: %s", e)
        return {}, {}

    structured = _extract_structured_fields(tables)
    filled = {k: v for k, v in structured.items() if v}
    print(f"[SFIS] Extracted fields: {list(filled.keys()) if filled else 'none — SN not found'}")
    return structured, tables


# ------------------------------------------------------------------
# Vendor query
# ------------------------------------------------------------------

def _query_vendor(session: requests.Session, sn: str, location: str) -> dict:
    url = f"{SFIS_BASE}/SFIS/PVS-vs-SFIS/SN/resources/getQuery.jsp"
    print(f"[SFIS] Querying vendor data for SN: {sn}, location: {location}")
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        "Referer": url,
        "X-Requested-With": "XMLHttpRequest",
    }
    payload = {
        "projectver": "", "disable_period": "on", "fromDate": "", "toDate": "",
        "buildevent": "", "modelname": "", "family": "", "sn": sn, "config": "",
        "comppn": "", "location": location, "mo": "", "carton_no": "",
    }
    try:
        r = session.post(url, data=payload, headers=headers, timeout=15)
        print(f"[SFIS] Vendor response: HTTP {r.status_code}")
    except requests.Timeout:
        print(f"[SFIS] ERROR — vendor query timed out for SN: {sn}, location: {location}")
        return {}

    if "application/json" not in r.headers.get("Content-Type", ""):
        print(f"[SFIS] ERROR — vendor response is HTML (session may have expired)")
        logger.warning("SFIS vendor response is HTML — session may have expired.")
        return {}
    data = r.json()
    if not data.get("data"):
        print(f"[SFIS] Vendor query: no records found")
        return {}
    item = data["data"][0]
    result = {
        "VENDOR": item.get("VENDOR", ""),
        "LOT NO": item.get("LOT_NO", ""),
        "DATE CODE": item.get("DATE_CODE", ""),
        "COMPONENT SN": item.get("COMPONENT_SN", ""),
    }
    print(f"[SFIS] Vendor data: {result}")
    return result


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def query_sn(serial_number: str, component: Optional[str] = None) -> str:
    """
    Query SFIS for a serial number and return a formatted summary string.
    Automatically checks connectivity, authenticates, and validates the SN.
    Optionally include a component location (e.g. 'R2251') for vendor data.
    """
    print(f"\n[SFIS] ── query_sn: SN={serial_number}, component={component} ──")
    if not check_connectivity():
        return "SFIS server is not reachable (http://10.52.1.9). Check network connectivity."

    session = get_session()
    try:
        structured, _ = _query_traveler(session, serial_number)
        if not any(structured.values()):
            print(f"[SFIS] SN '{serial_number}' not found — returning invalid SN message")
            return (
                f"Serial number '{serial_number}' was not found in SFIS. "
                "It may be invalid or not yet recorded in the system."
            )

        vendor: dict = {}
        if component:
            vendor = _query_vendor(session, serial_number, component)
    finally:
        session.close()
        print(f"[SFIS] Session closed")

    lines = [f"SFIS Data for SN: {serial_number}", "=" * 40]

    # Ordered structured fields
    field_order = [
        "PHASE", "MODEL", "CONFIG", "LINE", "PANEL SN", "SN SEQ IN PANEL",
        "FAILED DATE", "LAB IN TIME", "GROUP NAME",
        "FAILURE MESSAGE", "LIST OF FAILING TESTS",
    ]
    for key in field_order:
        val = structured.get(key, "")
        if val:
            lines.append(f"{key:<24}: {val}")

    if vendor:
        lines.append("\nComponent Vendor Data")
        lines.append("-" * 40)
        for key, val in vendor.items():
            if val:
                lines.append(f"{key:<24}: {val}")

    output = "\n".join(lines)
    print(f"[SFIS] Returning to agent:\n{output}")
    return output


# ------------------------------------------------------------------
# 2A defect query (time period)
# ------------------------------------------------------------------

def query_2a_defects(
    from_date: str,
    to_date: str,
    *,
    profit_center: str = "0000000025",
    mo: str = "ALL",
    model_serial: str = "",
    model_name: str = "",
    line_name: str = "",
    group_name: str = "ALL",
    error_code: str = "",
    retest_sequence: str = "FIRST",
) -> str:
    """Query 2A defect data for a date range. Returns formatted table output."""
    print(f"\n[SFIS] ── query_2a_defects: {from_date} → {to_date}, model='{model_name}', group='{group_name}' ──")
    if not check_connectivity():
        return "SFIS server is not reachable (http://10.52.1.9). Check network connectivity."

    session = get_session()
    try:
        params = {
            "profitCenter": profit_center,
            "projectVersion": "ALL",
            "fromDate": from_date,
            "toDate": to_date,
            "BU": "", "Customer": "", "buildEvent": "", "family": "",
            "buildConfig": "", "MO": mo, "modelSerial": model_serial,
            "modelName": model_name, "lotNo": "", "bigLot": "",
            "testStation": "ALL", "lineName": line_name,
            "groupName": group_name, "errorCode": error_code,
            "majorProject": "", "projectName": "", "productName": "",
            "retestSequence": retest_sequence, "recordType": "ALL",
            "empNo": "ALL", "processType": "ALL",
            "cbxGroupName": "Yes", "cbxTestTime": "Yes", "cbxErrorCode": "Yes",
            "group_name": group_name, "mo_number": mo,
        }
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json",
            "X-Requested-With": "XMLHttpRequest",
        }
        print(f"[SFIS] Sending 2A request to {_2A_URL}")
        r = session.get(_2A_URL, params=params, headers=headers, timeout=20)
        print(f"[SFIS] 2A response: HTTP {r.status_code}, Content-Type: {r.headers.get('Content-Type', 'unknown')}")
        if "application/json" not in r.headers.get("Content-Type", ""):
            print(f"[SFIS] ERROR — 2A query returned non-JSON")
            return "2A query returned a non-JSON response."

        tables = _parse_tables_from_json(r.json())
        if not tables:
            print(f"[SFIS] 2A query: no records found")
            return f"No 2A defect data found for {from_date} → {to_date}."
        print(f"[SFIS] 2A query: {sum(len(r) for r in tables.values())} rows across {len(tables)} table(s)")

        lines: list[str] = [f"2A Defects  {from_date} → {to_date}", "=" * 50]
        for table_name, rows in tables.items():
            if not rows:
                continue
            lines.append(f"\n[{table_name}]")
            for row in rows:
                for key, val in row.items():
                    if val:
                        if ";" in str(val):
                            lines.append(f"  {key}:")
                            for part in _convert_ec_multiline(val).splitlines():
                                lines.append(f"    {part}")
                        else:
                            lines.append(f"  {key}: {val}")
                lines.append("")
        return "\n".join(lines)
    finally:
        session.close()


# ------------------------------------------------------------------
# PVS-vs-SFIS query (flexible component/vendor lookup)
# ------------------------------------------------------------------

def query_pvs(
    sn: str = "",
    location: str = "",
    model_name: str = "",
    family: str = "",
    from_date: str = "",
    to_date: str = "",
    mo: str = "",
    carton_no: str = "",
    comp_pn: str = "",
) -> str:
    """Query PVS-vs-SFIS for vendor, lot, date-code, and component traceability."""
    print(f"\n[SFIS] ── query_pvs: sn='{sn}', location='{location}', model='{model_name}' ──")
    if not check_connectivity():
        return "SFIS server is not reachable (http://10.52.1.9). Check network connectivity."

    session = get_session()
    try:
        payload = {
            "projectver": "",
            "disable_period": "on" if not (from_date or to_date) else "",
            "fromDate": from_date,
            "toDate": to_date,
            "buildevent": "",
            "modelname": model_name,
            "family": family,
            "sn": sn,
            "config": "",
            "comppn": comp_pn,
            "location": location,
            "mo": mo,
            "carton_no": carton_no,
        }
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
            "X-Requested-With": "XMLHttpRequest",
        }
        print(f"[SFIS] Sending PVS request to {_PVS_URL}")
        r = session.post(_PVS_URL, data=payload, headers=headers, timeout=15)
        print(f"[SFIS] PVS response: HTTP {r.status_code}, Content-Type: {r.headers.get('Content-Type', 'unknown')}")
        if "application/json" not in r.headers.get("Content-Type", ""):
            print(f"[SFIS] ERROR — PVS query returned non-JSON")
            return "PVS query returned a non-JSON response."

        data = r.json()
        records = data.get("data", [])
        if not records:
            print(f"[SFIS] PVS query: no records found")
            return "No PVS data found for the given parameters."
        print(f"[SFIS] PVS query: {len(records)} record(s) found")

        lines: list[str] = ["PVS-vs-SFIS Results", "=" * 40]
        for item in records:
            for field, label in [
                ("SN", "SN"), ("MODEL_NAME", "Model Name"), ("VENDOR", "Vendor"),
                ("LOT_NO", "Lot No"), ("DATE_CODE", "Date Code"),
                ("COMPONENT_SN", "Component SN"), ("LOCATION", "Location"),
            ]:
                val = item.get(field, "")
                if val:
                    lines.append(f"  {label:<16}: {val}")
            lines.append("")
        return "\n".join(lines)
    finally:
        session.close()
