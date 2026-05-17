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
    try:
        r = session.post(
            LOGIN_URL,
            data={"User_Name": username, "Pass_Word": password, "Func_Name": "LOGIN"},
            headers=_LOGIN_HEADERS,
            timeout=10,
        )
        r.raise_for_status()
        return r.json().get("RES") == "OK"
    except Exception as e:
        logger.warning("SFIS login attempt failed: %s", e)
        return False


def get_session() -> requests.Session:
    """Return an authenticated SFIS session, or raise SFISAuthError."""
    username, password = load_credentials()
    if not username or not password:
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

    # Phase / Model / Config  ← "Work Order / Model Data"
    model_table = tables.get("Work Order / Model Data", [])
    for row in model_table:
        if row.get("HW BOM"):
            fields["PHASE"] = row.get("VERSION CODE", "")
            fields["MODEL"] = row.get("HW BOM", "")
            fields["CONFIG"] = row.get("SW BOM", "")
            break

    # SMT Line / Panel SN  ← "SN Detail Data"
    sn_detail = tables.get("SN Detail Data", [])
    for row in sn_detail:
        if row.get("VIRTUAL LINE1"):
            fields["LINE"] = row.get("VIRTUAL LINE1", "")
            fields["PANEL SN"] = row.get("TRACK NO", "")
            break

    # SN position in panel  ← "Wip Tracking Data"
    wip_table = tables.get("Wip Tracking Data", [])
    for row in wip_table:
        seq = row.get("SN SEQ IN PANEL", "")
        if seq:
            fields["SN SEQ IN PANEL"] = seq
            break

    # Failure date / group / test codes  ← "SN Repair Data"
    repair_table = tables.get("SN Repair Data", [])
    for row in repair_table:
        station = (row.get("TEST STATION") or "").lower()
        if any(s in station for s in _FAILURE_STATIONS):
            test_code = row.get("TEST CODE", "")
            if test_code:
                fields["FAILED DATE"] = row.get("TEST TIME", "")
                fields["GROUP NAME"] = row.get("TEST GROUP", "")
                fields["LIST OF FAILING TESTS"] = test_code
                break

    # Lab-in time  ← "Laboratory In/Out"
    lab_table = tables.get("Laboratory In/Out", [])
    for row in lab_table:
        if row.get("LAB IN EMP") and row.get("LAB IN TIME"):
            fields["LAB IN TIME"] = row.get("LAB IN TIME", "")
            break

    # Failure message + refined test list  ← "Bobcat Data"
    bobcat_table = tables.get("Bobcat Data", [])
    failed_date = fields.get("FAILED DATE", "")
    if bobcat_table and failed_date:
        try:
            dt = datetime.strptime(failed_date, "%Y/%m/%d %H:%M:%S")
            target_time = dt.strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            target_time = ""

        for row in bobcat_table:
            station_id = row.get("STATION ID", "")
            stop_time = row.get("STOP TIME", "")
            failing_tests = row.get("LIST OF FAILING TESTS", "")
            if failing_tests and "FAAP" in station_id and stop_time == target_time:
                fields["FAILURE MESSAGE"] = row.get("FAILURE MESSAGE", "")
                fields["LIST OF FAILING TESTS"] = _convert_ec_multiline(failing_tests)
                break

    # Fallback: if LIST OF FAILING TESTS still has semicolons, expand them
    lot = fields["LIST OF FAILING TESTS"]
    if ";" in lot:
        fields["LIST OF FAILING TESTS"] = _convert_ec_multiline(lot)

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
    try:
        r = session.get(url, headers=_TRAVELER_HEADERS, timeout=15)
    except requests.Timeout:
        raise RuntimeError("SFIS traveler query timed out.")

    if "application/json" not in r.headers.get("Content-Type", ""):
        return {}, {}

    try:
        tables = _parse_tables_from_json(r.json())
    except Exception as e:
        logger.warning("Failed to parse SFIS JSON: %s", e)
        return {}, {}

    return _extract_structured_fields(tables), tables


# ------------------------------------------------------------------
# Vendor query
# ------------------------------------------------------------------

def _query_vendor(session: requests.Session, sn: str, location: str) -> dict:
    url = f"{SFIS_BASE}/SFIS/PVS-vs-SFIS/SN/resources/getQuery.jsp"
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
    except requests.Timeout:
        return {}

    if "application/json" not in r.headers.get("Content-Type", ""):
        logger.warning("SFIS vendor response is HTML — session may have expired.")
        return {}
    data = r.json()
    if not data.get("data"):
        return {}
    item = data["data"][0]
    return {
        "VENDOR": item.get("VENDOR", ""),
        "LOT NO": item.get("LOT_NO", ""),
        "DATE CODE": item.get("DATE_CODE", ""),
        "COMPONENT SN": item.get("COMPONENT_SN", ""),
    }


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def query_sn(serial_number: str, component: Optional[str] = None) -> str:
    """
    Query SFIS for a serial number and return a formatted summary string.
    Optionally include a component location (e.g. 'R2251') for vendor data.
    """
    session = get_session()
    try:
        structured, _ = _query_traveler(session, serial_number)
        if not structured:
            return f"No data found in SFIS for serial number: {serial_number}"

        vendor: dict = {}
        if component:
            vendor = _query_vendor(session, serial_number, component)
    finally:
        session.close()

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

    return "\n".join(lines)
