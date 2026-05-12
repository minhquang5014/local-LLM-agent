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
# JSON / HTML table parsers (ported from original script)
# ------------------------------------------------------------------

def _parse_tables_from_json(json_obj: dict) -> dict:
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
            headers = []
            if thead:
                headers = [td.get_text(strip=True) for td in thead.find_all("td")]
            if not headers or not tbody:
                result[unique_title] = []
                continue

            parsed = []
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
# Traveler query — returns ALL table data as formatted text
# ------------------------------------------------------------------

def _query_traveler(session: requests.Session, sn: str) -> str:
    url = (
        f"{SFIS_BASE}/SFIS/Production/Travelers/Trav_1/resources/getQueryJSON.jsp"
        f"?ColItem=serial_number&Field_Kind=ALLFIELD"
        f"&fdSerial_Number=Y&fdModel_Serial=Y&fdMo_Number=Y&fdLine_Name=Y"
        f"&fdSection_Name=Y&fdGroup_Name=Y&fdStation_Name=Y"
        f"&fdIn_Station_Time=Y&fdOut_Station_Time=Y&fdRETEST_SEQ=Y"
        f"&fdEmp_No=Y&fdQa_NO=Y&fdQa_Result=Y&fdPallet_No=Y&fdCarton_NO=Y"
        f"&fdPO_NO=Y&fdCONTAINER_NO=Y&fdSHIPPING_SN=Y&fdMAC=Y&fdTRACK_NO=Y"
        f"&fdKEY_PART_NO=Y&fdMODEL_NAME=Y&fdBill_NO=Y&fdError_flag=Y"
        f"&fdFinish_flag=Y&fdVersion_Code=Y&fdSpecial_route=Y&fdCust_model=Y"
        f"&fdCust_PN=Y&fdInv_no=Y&fdOther_MaC=Y&fdKP_NO_C=Y&fdMain_Product=Y"
        f"&fdProduct_Name=Y&fdBox_No=Y&fdLOTN=Y&fdLOTB=Y&fdOut_Line_Time=Y"
        f"&fdDRYBOX=Y&fdVIRTUAL_LINE1=Y&fdVIRTUAL_LINE2=Y&fdPanelSeq=Y"
        f"&fdBCadd=Y&fdBCqry=Y&InpData={sn}&FromURL=N"
    )
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
        "X-Requested-With": "XMLHttpRequest",
    }
    r = session.get(url, headers=headers, timeout=15)
    if "application/json" not in r.headers.get("Content-Type", ""):
        return ""

    tables = _parse_tables_from_json(r.json())
    if not tables:
        return ""

    lines: list[str] = []
    for table_name, rows in tables.items():
        if not rows:
            continue
        lines.append(f"\n[{table_name}]")
        for row in rows:
            for key, val in row.items():
                if val:
                    # Expand multi-value EC strings onto separate lines
                    if ";" in str(val):
                        lines.append(f"  {key}:")
                        for part in _convert_ec_multiline(val).splitlines():
                            lines.append(f"    {part}")
                    else:
                        lines.append(f"  {key}: {val}")
            lines.append("")

    return "\n".join(lines)


# ------------------------------------------------------------------
# Vendor query
# ------------------------------------------------------------------

def _query_vendor(session: requests.Session, sn: str, location: str) -> dict:
    url = f"{SFIS_BASE}/SFIS/PVS-vs-SFIS/SN/resources/getQuery.jsp"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        "X-Requested-With": "XMLHttpRequest",
    }
    payload = {
        "projectver": "", "disable_period": "on", "fromDate": "", "toDate": "",
        "buildevent": "", "modelname": "", "family": "", "sn": sn, "config": "",
        "comppn": "", "location": location, "mo": "", "carton_no": "",
    }
    r = session.post(url, data=payload, headers=headers, timeout=15)
    if "application/json" not in r.headers.get("Content-Type", ""):
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
    Optionally include a component location for vendor data.
    """
    session = get_session()
    try:
        traveler_text = _query_traveler(session, serial_number)
        if not traveler_text.strip():
            return f"No data found in SFIS for serial number: {serial_number}"

        vendor: dict = {}
        if component:
            vendor = _query_vendor(session, serial_number, component)
    finally:
        session.close()

    lines = [f"SFIS Data for SN: {serial_number}", "=" * 40, traveler_text]

    if vendor:
        lines.append("\nComponent Vendor Data")
        lines.append("-" * 40)
        for key, val in vendor.items():
            if val:
                lines.append(f"{key:<24}: {val}")

    return "\n".join(lines)
