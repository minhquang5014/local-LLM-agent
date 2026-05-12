import pandas as pd
import os
import sys
import json
import threading
import requests
import openpyxl
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from bs4 import BeautifulSoup
from datetime import datetime


start_time = datetime.now()
print("start time: " , start_time)

#------------------------------------------login sfis--------------------------------------------
SFIS_BASE = "http://10.52.1.9"
LOGIN_URL = f"{SFIS_BASE}/SFIS/Member/resources/func_login.jsp"

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
    "X-Requested-With": "XMLHttpRequest",
}

if getattr(sys, 'frozen', False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CRED_FILE = os.path.join(BASE_DIR, "sfis_cred.json")
def load_credentials():
    """Load username/password from json file"""
     # Create new file json 
    if not os.path.exists(CRED_FILE):
        save_credentials("", "")
        return "", ""
    
    try:
        with open(CRED_FILE, "r", encoding="utf8") as f:
            data = json.load(f)
            return data.get("username", ""), data.get("password", "")
    except:
        return "", ""

def save_credentials(username, password):
    """Save username/password to file JSON."""

    with open(CRED_FILE, "w", encoding="utf8") as f:
        json.dump({"username": username, "password": password}, f, indent=4)

def try_login_api(session, username, password):

    payload = {
        "User_Name": username,
        "Pass_Word": password,
        "Func_Name": "LOGIN"
    }

    try:
        r = session.post(LOGIN_URL, data=payload, headers=HEADERS, timeout=10)
        r.raise_for_status()
        result = r.json()

    except requests.exceptions.ConnectionError:
        messagebox.showerror("Network Error", "Network error. Please check network.")
        raise

    except requests.exceptions.Timeout:
        messagebox.showerror("Network Error", "Request timeout. Please check network.")
        raise

    return result.get("RES") == "OK"

def ask_for_credentials(parent=None):

    if parent is None:
        parent = tk._default_root

    dialog = tk.Toplevel(parent)
    dialog.title("SFIS Login")
    dialog.resizable(False, False)
    dialog.transient(parent)
    dialog.grab_set()

    tk.Label(dialog, text="Username:").grid(row=0, column=0, padx=10, pady=5)
    tk.Label(dialog, text="Password:").grid(row=1, column=0, padx=10, pady=5)

    username_var = tk.StringVar()
    password_var = tk.StringVar()

    username_entry = tk.Entry(dialog, textvariable=username_var, width=25)
    password_entry = tk.Entry(dialog, textvariable=password_var, show="*", width=25)

    username_entry.grid(row=0, column=1, padx=10, pady=5)
    password_entry.grid(row=1, column=1, padx=10, pady=5)

    error_label = tk.Label(dialog, text="", fg="red")
    error_label.grid(row=2, column=0, columnspan=2)

    result = {"ok": False}

    def on_login(event=None):

        username = username_var.get().strip()
        password = password_var.get().strip()

        if not username or not password:
            error_label.config(text="Username / Password cannot be empty")
            return

        result["ok"] = True
        dialog.destroy()

    def on_cancel():
        dialog.destroy()

    tk.Button(dialog, text="Login", command=on_login, width=10).grid(row=3, column=0, pady=10)
    tk.Button(dialog, text="Cancel", command=on_cancel, width=10).grid(row=3, column=1)

    dialog.bind("<Return>", on_login)

    username_entry.focus()
    parent.wait_window(dialog)

    if not result["ok"]:
        return None, None

    return username_var.get(), password_var.get()

def login_with_retry(session):

    username, password = load_credentials()

    if username and password:
        if try_login_api(session, username, password):
            return True

    while True:

        username, password = ask_for_credentials()

        if not username:
            return False

        if try_login_api(session, username, password):
            save_credentials(username, password)
            return True

        messagebox.showerror(
            "Login Failed",
            "Invalid username or password."
        )

def create_sfis_session() -> requests.Session | None:
    """
    Create SFIS session and login once
    """
    session = requests.Session()

    try:
        if not login_with_retry(session):
            return None
        return session
    except Exception:
        session.close()
        return None


#------------------------------------------query and process data---------------------------
def parse_single_table(table_tag):
    """convert <table> to list of dict, for exp: {'Work Order / Model Data': [{'MO NUMBER': '1016544-VGR201H',...}], 'Bobcat':[{'MO NUMBER': '1016544-VGR201H',...}] """
    
    rows = table_tag.find_all("tr") #get all rows in table
    if not rows:
        return []

    # Get header
    header_cells = rows[0].find_all(["th", "td"])
    headers = [cell.get_text(separator=" ", strip=True) for cell in header_cells]
    data_rows = []

    # Check header, value in table
    for row in rows[1:]:
        cells = row.find_all(["td", "th"])
        values = [cell.get_text(separator=" ", strip=True) for cell in cells]
        if len(values) < len(headers):
            values += [""] * (len(headers) - len(values))
        row_dict = dict(zip(headers, values))
        if len(values) > len(headers):
            for i in range(len(headers), len(values)):
                row_dict[f"extra_col_{i+1}"] = values[i]
        data_rows.append(row_dict)

    return data_rows

def parse_tables_from_json(json_data):
    """
    Get data from json and convert to dict
    """
    # print(json_data)  # giữ comment nếu cần debug

    # 1. Convert Response -> dict nếu cần
    if hasattr(json_data, "json") and not isinstance(json_data, dict):
        try:
            json_obj = json_data.json()
        except Exception as e:
            raise ValueError(f"Can not convert Response -> json: {e}")
    else:
        json_obj = json_data

    # 2. Lấy HTML chứa bảng
    raw_html = json_obj.get("tableContents", "")
    if not raw_html:
        return {}

    # 3. Parse HTML
    soup = BeautifulSoup(raw_html, "html.parser")

    # Các tag có ý nghĩa (title + table)
    important_tags = soup.find_all(
        ["font", "h1", "h2", "h3", "h4", "h5", "h6", "table"]
    )

    result = {}
    pending_title = None
    tbl_index = 0
    name_count = {}

    # 4. Make unique title cho table
    def make_unique(name):
        key = name.strip()
        if not key:
            return key
        if key not in name_count:
            name_count[key] = 1
            return key
        else:
            name_count[key] += 1
            return f"{key}#{name_count[key]}"

    # 5. Parse từng tag
    for tag in important_tags:
        # ---- Title ----
        if tag.name.lower() in ("font", "h1", "h2", "h3", "h4", "h5", "h6"):
            text = tag.get_text(separator=" ", strip=True)
            if text:
                pending_title = text

        # ---- Table ----
        elif tag.name.lower() == "table":
            tbl_index += 1

            if pending_title:
                title = pending_title
                pending_title = None
            else:
                title = f"table_{tbl_index}"

            unique_title = make_unique(title)
            parsed = []

            thead = tag.find("thead")
            tbody = tag.find("tbody")

            headers = []
            if thead:
                headers = [
                    td.get_text(strip=True)
                    for td in thead.find_all("td")
                ]

            if not headers:
                result[unique_title] = []
                continue

            if not tbody:
                result[unique_title] = []
                continue

            rows = tbody.find_all("tr")
            if rows:
                for tr in rows:
                    cols = [
                        td.get_text(strip=True)
                        for td in tr.find_all("td")
                    ]
                    if len(cols) == len(headers):
                        parsed.append(dict(zip(headers, cols)))
            else:
                tds = tbody.find_all("td")
                if len(tds) == len(headers):
                    row = [td.get_text(strip=True) for td in tds]
                    parsed.append(dict(zip(headers, row)))

            result[unique_title] = parsed

    return result

def sfis_filter_er(text: str):
    result = text.split(";",1)[0].strip()
    return result

def convert_ec_string_to_multiline(ec_string):
    """
    Chuyển chuỗi EC phân tách bởi ';'
    thành chuỗi nhiều dòng trong 1 ô Excel
    """
    ec_list = [
        ec.strip()
        for ec in ec_string.split(";")
        if ec.strip()
    ]
    return "\n".join(ec_list)

def query_sfis_inputdata(session: requests.Session, serial_number: str):
    SFIS_BASE = "http://10.52.1.9"
    SFIS_QUERY_URL = f"{SFIS_BASE}/SFIS/Production/Travelers/Trav_1/resources/getQueryJSON.jsp?ColItem=serial_number&Field_Kind=ALLFIELD&fdSerial_Number=Y&fdModel_Serial=Y&fdMo_Number=Y&fdLine_Name=Y&fdSection_Name=Y&fdGroup_Name=Y&fdStation_Name=Y&fdIn_Station_Time=Y&fdOut_Station_Time=Y&fdRETEST_SEQ=Y&fdEmp_No=Y&fdQa_NO=Y&fdQa_Result=Y&fdPallet_No=Y&fdCarton_NO=Y&fdPO_NO=Y&fdCONTAINER_NO=Y&fdSHIPPING_SN=Y&fdMAC=Y&fdTRACK_NO=Y&fdKEY_PART_NO=Y&fdMODEL_NAME=Y&fdBill_NO=Y&fdError_flag=Y&fdFinish_flag=Y&fdVersion_Code=Y&fdSpecial_route=Y&fdCust_model=Y&fdCust_PN=Y&fdInv_no=Y&fdOther_MaC=Y&fdKP_NO_C=Y&fdMain_Product=Y&fdProduct_Name=Y&fdBox_No=Y&fdLOTN=Y&fdLOTB=Y&fdOut_Line_Time=Y&fdDRYBOX=Y&fdVIRTUAL_LINE1=Y&fdVIRTUAL_LINE2=Y&fdPanelSeq=Y&fdBCadd=Y&fdBCqry=Y&InpData={serial_number}&FromURL=N"

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
        "X-Requested-With": "XMLHttpRequest",
    }

    query_payload = {
    "ColItem": serial_number,
    "Field_Kind": "ALLFIELD",
    "fdSerial_Number": "Y",
    "fdModel_Serial": "Y",
    "fdMo_Number": "Y",
    "fdLine_Name": "Y",
    "fdSection_Name": "Y",
    "fdGroup_Name": "Y",
    "fdStation_Name": "Y",
    "fdIn_Station_Time": "Y",
    "fdOut_Station_Time": "Y",
    "fdRETEST_SEQ": "Y",
    "fdEmp_No": "Y",
    "fdQa_NO": "Y",
    "fdQa_Result": "Y",
    "fdPallet_No": "Y",
    "fdCarton_NO": "Y",
    "fdPO_NO": "Y",
    "fdCONTAINER_NO": "Y",
    "fdSHIPPING_SN": "Y",
    "fdMAC": "Y",
    "fdTRACK_NO": "Y",
    "fdKEY_PART_NO": "Y",
    "fdMODEL_NAME": "Y",
    "fdBill_NO": "Y",
    "fdError_flag": "Y",
    "fdFinish_flag": "Y",
    "fdVersion_Code": "Y",
    "fdSpecial_route": "Y",
    "fdCust_model": "Y",
    "fdCust_PN": "Y",
    "fdInv_no": "Y",
    "fdOther_MaC": "Y",
    "fdKP_NO_C": "Y",
    "fdMain_Product": "Y",
    "fdProduct_Name": "Y",
    "fdBox_No": "Y",
    "fdLOTN": "Y",
    "fdLOTB": "Y",
    "fdOut_Line_Time": "Y",
    "fdDRYBOX": "Y",
    "fdVIRTUAL_LINE1": "Y",
    "fdVIRTUAL_LINE2": "Y",
    "fdPanelSeq": "Y",
    "fdBCadd": "Y",
    "fdBCqry": "Y",
    "InpData": serial_number,
    "FromURL": "N"
    }

    try:
        query_response = session.get(SFIS_QUERY_URL, data=query_payload, headers=headers, timeout=15)
    except Exception as e:
        print(f"SFIS query request failed: {e}")
        session.close()
        return ""

    content_type = query_response.headers.get("Content-Type", "")
    if "application/json" in content_type:
        try:
            query_result = query_response.json()
            tables = parse_tables_from_json(query_result)
        except Exception as e:
            print(f"Failed to parse SFIS JSON response: {e}")
            session.close()
            return ""
        
        list_results = {
            "PHASE": "",
            "MODEL": "",
            "CONFIG": "",
            "LINE": "",
            "PANEL SN": "",
            "SN SEQ IN PANEL": "",
            "FAILED DATE": "",
            "LAB IN TIME": "",
            "GROUP NAME": "",
            "FAILURE MESSAGE": "",
            "LIST OF FAILING TESTS": ""
        }

        ### Get Phase, Model
        model_table = tables.get("Work Order / Model Data")
        if model_table:
            for headers in model_table:
                if "HW BOM" in headers and headers["HW BOM"] is not None:
                    sip_phase = headers["VERSION CODE"]
                    sip_model = headers["HW BOM"]
                    sip_config = headers["SW BOM"]
                if sip_phase and sip_model and sip_config:
                    list_results["PHASE"] = sip_phase
                    list_results["MODEL"] = sip_model
                    list_results["CONFIG"] = sip_config
                    break
                else:
                    continue
        else:
            print("No have model_table")

        ### Get SMT Line
        smt_table = tables.get("SN Detail Data")
        if smt_table:
            for headers in smt_table:
                if "VIRTUAL LINE1" in headers and headers["VIRTUAL LINE1"] is not None:
                    sip_smt_line = headers["VIRTUAL LINE1"]
                    sip_panel = headers["TRACK NO"]
                if sip_smt_line and sip_panel:
                    list_results["LINE"] = sip_smt_line
                    list_results["PANEL SN"] = sip_panel
                    break
                else:
                    continue
        else:
            print("No have smt_table")
        
        ### Get array in panel
        sip_array = None
        wip_table = tables.get("Wip Tracking Data")
        if wip_table:
            for headers in wip_table:
                if "SN SEQ IN PANEL" in headers and headers["SN SEQ IN PANEL"] is not None:
                    sip_array = headers["SN SEQ IN PANEL"]
                if sip_array:
                    list_results["SN SEQ IN PANEL"] = sip_array
                    break
                else:
                    continue
        else:
            print("No have wip_table")

        ### Get test code
        station_list = ["burn_in","dfu","fct","cell","wifi","s-cond","s_cond","t269","t-269","a-cond","a_cond"]
        repair_table = tables.get("SN Repair Data")
        sip_test_code = None
        test_time = None
        test_group = None
        if repair_table:
            for headers in repair_table:
                sip_station = headers["TEST STATION"]
                sip_station_cp = sip_station.lower()
                for station in station_list:
                    if station in sip_station_cp:
                        if "TEST CODE" in headers and headers["TEST CODE"] is not None:
                            test_time = headers["TEST TIME"]
                            test_group = headers["TEST GROUP"]
                            # failure_message = headers["TEST CODE"]
                            sip_test_code = headers["TEST CODE"]
                            if test_time and test_group:
                                list_results["FAILED DATE"] = test_time
                                list_results["GROUP NAME"] = test_group
                                # list_results["FAILURE MESSAGE"] = failure_message
                                list_results["LIST OF FAILING TESTS"] = sip_test_code
                                break
                if sip_test_code and test_time and test_group:
                    break
                else:
                    continue
        else:
            print("No have repair_table")
        
        ### get lab in time
        laboratory_table = tables.get("Laboratory In/Out")
        lab_in_time = ""
        if laboratory_table:
            for headers in laboratory_table:
                if "LAB IN EMP" in headers and headers["LAB IN EMP"] is not None:
                    staff_id = headers["LAB IN EMP"]
                if sip_array:
                    lab_in_time = headers["LAB IN TIME"]
                    list_results["LAB IN TIME"] = lab_in_time
                else:
                    continue
        else:
            print("No have laboratory in/out table")
        
        #### get symptom
        table_bobcat = tables.get("Bobcat Data")
        
        failure_message = None
        if table_bobcat:
            if test_time:
                dt = datetime.strptime(test_time, "%Y/%m/%d %H:%M:%S")
                new_time_str = dt.strftime("%Y-%m-%d %H:%M:%S")
                for headers in table_bobcat:
                    if "LIST OF FAILING TESTS" in headers and headers["LIST OF FAILING TESTS"] is not None:
                        if "STATION ID" in headers and headers["STATION ID"] is not None:
                            station_id = headers["STATION ID"]
                            stop_time = headers["STOP TIME"]
                            if "FAAP" in station_id and new_time_str == stop_time:
                                failure_message = headers["FAILURE MESSAGE"]
                                sip_test_code = headers["LIST OF FAILING TESTS"]
                    if failure_message and sip_test_code:
                        sip_test_code= convert_ec_string_to_multiline(sip_test_code)
                        list_results["FAILURE MESSAGE"] = failure_message
                        list_results["LIST OF FAILING TESTS"] = sip_test_code
                        break
                    else:
                        continue
        else:
            print("No data returned for this query. Check parameters or page structure.")

    else:
        print("Server did not return JSON. Check debug logs.")

    session.close()
    return list_results

def query_sfis_vendor(session: requests.Session, serial_number: str, location: str) -> dict:
    SFIS_BASE = "http://10.52.1.9"
    url = f"{SFIS_BASE}/SFIS/PVS-vs-SFIS/SN/resources/getQuery.jsp"

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        "Referer": url,
        "X-Requested-With": "XMLHttpRequest",
    }

    payload = {
        "projectver": "",
        "disable_period": "on",
        "fromDate": "",
        "toDate": "",
        "buildevent": "",
        "modelname": "",
        "family": "",
        "sn": serial_number,
        "config": "",
        "comppn": "",
        "location": location,
        "mo": "",
        "carton_no": ""
    }

    r = session.post(url, data=payload, headers=headers, timeout=15)

    if "application/json" not in r.headers.get("Content-Type", ""):
        print("[SFIS] Vendor response is HTML → login expired")
        return {}

    data = r.json()

    if not data.get("data"):
        print(f"{serial_number} don't have component data of {location}")
        return {}

    item = data["data"][0]

    return {
        "VENDOR": item.get("VENDOR"),
        "LOT NO": item.get("LOT_NO"),
        "DATE CODE": item.get("DATE_CODE"),
        "COMPONENT SN": item.get("COMPONENT_SN")
    }

def adjust_component_name(component):

    monitor_list = ["UNDER FA", "UNDER FA CHECK-IN"]
    if component == "" or component is None or component.upper() in monitor_list:
        return component

    golden_name = ['Location1', 'R2251', 'R2331', 'R1802A', 'R1808A', 'R1813A', 'R1818A', 'R1823A', 'R1828A', 'R1801C', 'R1807C', 'R1825C', 'R1838C', 'R1844C', 'R1856C', 'R1863C', 'R1870C', 'R1875C', 'R1880C', 'R1886C', 'R1891C', 'R1896C', 'R4007', 'R1000', 'R2800', 'R2312', 'R5407_RF', 'R3000', 'R2671', 'R4517', 'R5404_RF', 'R5606_RF', 'R5801_RF', 'Z5901_RF', 'R3902', 'R0401', 'R1653', 'R1675', 'R4635', 'R1006', 'R4003', 'R1650', 'R1801', 'R1806', 'R1805D', 'R1810D', 'R1815D', 'R1820D', 'R1825D', 'R1830D', 'R1835D', 'R2611', 'R1210', 'R7501', 'R2310', 'R1602', 'R3901', 'R7200', 'R2653', 'R1002', 'R4006', 'R6110_RF', 'R1645', 'R1601', 'R2300', 'R6301_RF', 'R1631', 'R7001', 'R4505', 'R2250', 'R1641', 'R7608', 'R5401_RF', 'C7600', 'C7612', 'C7521', 'C2600B', 'C2602', 'C1004', 'C2005', 'C2028', 'C2083', 'C2215', 'C2631', 'C3842', 'C4578', 'C2053', 'C4052', 'C1200', 'C2250', 'C4540', 'C5800_RF', 'C1600', 'C7404', 'C4003', 'C502', 'C5907_RF', 'C4005', 'C4022', 'C4207', 'C4215', 'C4284', 'C4486', 'C4597', 'C4611', 'C7027', 'C4212', 'C6205_RF', 'C2615', 'C1052', 'C2300', 'C7005', 'C7021', 'C7504', 'C7535', 'C504', 'C2251', 'C5218_RF', 'C7539', 'C3851', 'C2654', 'C2611', 'C2683', 'C7023', 'C1601', 'C4610', 'C5706_RF', 'C1057', 'C2632', 'C5413_RF', 'C5712_RF', 'C2607', 'C2603', 'C7525', 'C5503_RF', 'C2004', 'C2031', 'C2090', 'C2198', 'C2467', 'C1002', 'C1402', 'C2240', 'C5205_RF', 'C3012', 'C5403_RF', 'C5209_RF', 'C1311', 'C1435', 'C7004', 'C7015', 'C2640', 'C3010', 'C3097', 'C1300', 'C2109', 'C2216', 'C2283', 'C2801', 'C3023', 'C5404_RF', 'C5604_RF', 'C5207_RF', 'C5220_RF', 'C5211_RF', 'C1342', 'C2626', 'C3001', 'C5711_RF', 'C3002', 'C2660', 'C7002', 'C1436', 'C2050', 'C1060', 'C7505', 'C3802', 'R7431', 'L4540', 'L7502', 'L5200_RF', 'L3851', 'L7600', 'L5203_RF', 'L2004', 'L2009', 'L2000', 'L2001', 'L5701_RF', 'L5700_RF', 'L2003', 'L5201_RF', 'L4022', 'FL4020', 'FL7500', 'FL4603', 'L4208', 'FL4613', 'L4001', 'CP6100_RF', 'FL4406', 'FL1007', 'FL1001', 'Y7200', 'Y1200', 'Y7500', 'Y5300_RF', 'U5600_RF', 'U1900', 'U5601_RF', 'U_BB_RF', 'U_ETM_RF', 'U_PMU_RF', 'U3800', 'U2801', 'U2600', 'U6200_RF', 'U3100', 'Q2010', 'Q500', 'Q2800', 'D4602', 'DZ4010', 'D4601', 'C500', 'C2605', 'C2600A', 'C4580', 'Z6105_RF', 'R7434', 'Z6100_RF', 'FL6100_RF', 'Q3902', 'Q4201', 'R5604_RF', 'SHLD_FENCE_ETIC', 'R4052', 'R4450', 'D40B0', 'R4210', 'R7411', 'R7412', 'R5300_RF', 'C2006', 'C1014', 'C1062', 'C1301', 'C1341', 'C1404', 'C2014', 'C2042', 'C2087', 'C2103', 'C2114', 'C2239', 'C3103', 'C5707_RF', 'C3020', 'C5718_RF', 'C5200_RF', 'C5302_RF', 'C5409_RF', 'C6203_RF', 'C2803', 'U_HB_RF', 'U7400', 'U2802', 'R7432', 'C7617', 'U7402', 'C7616', 'R7461', 'Y2201', 'R7000', 'D40B1', 'R2110', 'R5301_RF', 'U1000', 'U3000', 'U7000', 'R7462', 'Z6104_RF', 'R7440', 'C7529', 'C7200', 'Z6101_RF', 'U7500', 'U2800', 'R2334', 'R1807', 'L6300_RF', 'C2334', 'C5904_RF', 'C6304_RF', 'C5902_RF', 'C6306_RF', 'C3302', 'C6301_RF', 'C5903_RF', 'C6300_RF', 'C6305_RF', 'C3301', 'L6301_RF', 'U3300', 'U6300_RF', 'U_LB_RF', 'J6200_RF', 'J0420', 'J4303', 'J4302', 'J4301', 'BS0401', 'Z5804_RF', 'U6301_RF', 'SHLD_CAN_GNSS', 'SHLD_CAN_LBPA', 'Z5801_RF', 'C3808', 'R2690', 'R1803A', 'R1809A', 'R1814A', 'R1819A', 'R1824A', 'R1802C', 'R1817C', 'R1826C', 'R1839C', 'R1845C', 'R1858C', 'R1865C', 'R1871C', 'R1876C', 'R1881C', 'R1887C', 'R1892C', 'R1897C', 'R4201', 'R1003', 'R2801', 'R2425', 'R5409_RF', 'R3900', 'R2672', 'R4518', 'R5405_RF', 'R5607_RF', 'R5901_RF', 'Z6110_RF', 'R0411', 'R1654', 'R1676', 'R1300', 'R4004', 'R1651', 'R1802', 'R1801D', 'R1806D', 'R1811D', 'R1816D', 'R1821D', 'R1826D', 'R1831D', 'R7507', 'R2311', 'R1630', 'R7201', 'R5200_RF', 'R1005', 'R4420', 'R5402_RF', 'R3801', 'R7609', 'C7603', 'C7620', 'C7530', 'C2601B', 'C1051', 'C2009', 'C2033', 'C2092', 'C2218', 'C2642', 'C4280', 'C4581', 'C4205', 'C4053', 'C1201', 'C4541', 'C5804_RF', 'C2331', 'C7405', 'C4004', 'C503', 'C4006', 'C4023', 'C4208', 'C4217', 'C4423', 'C4516', 'C4605', 'C4612', 'C6103_RF', 'C4214', 'C2662', 'C1316', 'C2421', 'C7010', 'C7022', 'C7509', 'C7536', 'C505', 'C5713_RF', 'C3852', 'C2617', 'C2684', 'C7400', 'C4407', 'C5301_RF', 'C1067', 'C2641', 'C5414_RF', 'C2608', 'C2604', 'C7534', 'C5504_RF', 'C2010', 'C2032', 'C2091', 'C2222', 'C2630', 'C1003', 'C2102', 'C5212_RF', 'C3013', 'C5411_RF', 'C1403', 'C1461', 'C7006', 'C7020', 'C5708_RF', 'C3011', 'C3098', 'C2003', 'C2200', 'C2217', 'C2285', 'C2807', 'C3104', 'C5405_RF', 'C5701_RF', 'C5208_RF', 'C5715_RF', 'C1900', 'C2627', 'C3019', 'C3003', 'C2661', 'C7009', 'C2000', 'C2052', 'C1061', 'C7517', 'C3806', 'L4541', 'L7601', 'L5702_RF', 'L2006', 'L2005', 'L2002', 'L5202_RF', 'L4023', 'FL4021', 'FL4604', 'L4002', 'FL4581', 'FL6200_RF', 'Q501', 'Q2801', 'D40B3', 'DZ40C0', 'C501', 'C2606', 'C2601A', 'C4582', 'Q3903', 'R4053', 'R4452', 'R4211', 'R7441', 'C2072', 'C1032', 'C1066', 'C1312', 'C1360', 'C1405', 'C2016', 'C2062', 'C2093', 'C2105', 'C2124', 'C2243', 'C3906', 'C5709_RF', 'C3021', 'C5202_RF', 'C5303_RF', 'C5410_RF', 'C7618', 'C7619', 'D40B2', 'C7531', 'C7201', '', 'R1808', 'R6300_RF', 'C6307_RF', 'C6200_RF', 'C3303', 'C6302_RF', 'J0422', 'R1805A', 'R1810A', 'R1815A', 'R1820A', 'R1825A', 'R1804C', 'R1818C', 'R1827C', 'R1840C', 'R1848C', 'R1859C', 'R1866C', 'R1872C', 'R1877C', 'R1882C', 'R1888C', 'R1893C', 'R1898C', 'R4203', 'R1004', 'R2603', 'R5601_RF', 'R7500', 'R2673', 'R4611', 'R5406_RF', 'R5608_RF', 'R6100_RF', 'R0413', 'R1655', 'R2202', 'R1803', 'R1802D', 'R1807D', 'R1812D', 'R1817D', 'R1822D', 'R1827D', 'R1832D', 'R2313', 'R5701_RF', 'R1301', 'R4421', 'C1071', 'C2013', 'C2041', 'C2201', 'C2280', 'C3017', 'C4282', 'C4583', 'C4206', 'C4517', 'C4542', 'C5808_RF', 'C2332', 'C7406', 'C4007', 'C4051', 'C4210', 'C4218', 'C4450', 'C4520', 'C4606', 'C4613', 'C40B0', 'C4216', 'C1320', 'C3801', 'C7011', 'C7024', 'C7522', 'C7537', 'C2253', 'C2618', 'C7403', 'C4451', 'C1330', 'C2802', 'C5415_RF', 'C2643', 'C5500_RF', 'C5600_RF', 'C2011', 'C2040', 'C2113', 'C2223', 'C3800', 'C1376', 'C2104', 'C5214_RF', 'C3014', 'C1408', 'C7000', 'C7007', 'C5710_RF', 'C3094', 'C3099', 'C2012', 'C2202', 'C2252', 'C2286', 'C3004', 'C4607', 'C5406_RF', 'C5703_RF', 'C5213_RF', 'C2622', 'C3810', 'C5801_RF', 'C2025', 'C2089', 'C1063', 'C7519', 'C3809', 'L4542', 'L2010', 'L2008', 'L2007', 'FL4213', 'FL4582', 'D40B4', 'C4585', 'Q3904', 'C2081', 'C1033', 'C1070', 'C1333', 'C1365', 'C1462', 'C2022', 'C2066', 'C2094', 'C2106', 'C2133', 'C3100', 'C7012', 'C5714_RF', 'C7500', 'C5203_RF', 'C5400_RF', 'C5602_RF', 'R6400_RF', 'C6308_RF', 'C6303_RF', 'R1806A', 'R1811A', 'R1816A', 'R1821A', 'R1826A', 'R1805C', 'R1820C', 'R1828C', 'R1842C', 'R1852C', 'R1861C', 'R1868C', 'R1873C', 'R1878C', 'R1883C', 'R1889C', 'R1894C', 'R5603_RF', 'R2654', 'R2608', 'R7505', 'R3102', 'R4612', 'R5602_RF', 'R5609_RF', 'Z5808_RF', 'R1600', 'R1662', 'R2664', 'R1804', 'R1803D', 'R1808D', 'R1813D', 'R1818D', 'R1823D', 'R1828D', 'R1833D', 'R4407', 'R1646', 'R4422', 'C1421', 'C2015', 'C2061', 'C2210', 'C2284', 'C3093', 'C4406', 'C4601', 'C4584', 'C4518', 'C4543', 'C6107_RF', 'C2333', 'C4020', 'C4201', 'C4211', 'C4281', 'C4452', 'C4521', 'C4608', 'C4635', 'C40C0', 'C7025', 'C1343', 'C3811', 'C7016', 'C7030', 'C7527', 'C7538', 'C2425', 'C2681', 'C4507', 'C1355', 'C3022', 'C2644', 'C5501_RF', 'C5601_RF', 'C2020', 'C2060', 'C2126', 'C2224', 'C7511', 'C1400', 'C2111', 'C5215_RF', 'C3016', 'C1414', 'C7001', 'C7008', 'C3095', 'C2098', 'C2204', 'C2281', 'C2420', 'C3015', 'C5221_RF', 'C5407_RF', 'C5217_RF', 'C2623', 'C7506', 'C2026', 'C1064', 'C3812', 'L4543', 'L3090', 'L2011', 'FL4215', 'C4586', 'Q3905', 'C1046', 'C1072', 'C1335', 'C1366', 'C2007', 'C2034', 'C2080', 'C2095', 'C2108', 'C2144', 'C3101', 'C7502', 'C5717_RF', 'C7526', 'C5206_RF', 'C5401_RF', 'C5716_RF', 'R6401_RF', 'R1807A', 'R1812A', 'R1817A', 'R1822A', 'R1827A', 'R1806C', 'R1822C', 'R1837C', 'R1843C', 'R1853C', 'R1862C', 'R1869C', 'R1874C', 'R1879C', 'R1885C', 'R1890C', 'R1895C', 'R5700_RF', 'R4506', 'R4008', 'R5403_RF', 'R5605_RF', 'R5800_RF', 'Z5811_RF', 'R1652', 'R1663', 'R2665', 'R1805', 'R1804D', 'R1809D', 'R1814D', 'R1819D', 'R1824D', 'R1829D', 'R1834D', 'R4005', 'R4454', 'C2001', 'C2021', 'C2071', 'C2211', 'C2616', 'C3841', 'C4430', 'C4602', 'C2690', 'C4021', 'C4203', 'C4213', 'C4283', 'C4454', 'C4558', 'C4609', 'C7026', 'C1344', 'C3822', 'C7018', 'C7031', 'C7532', 'C7540', 'C4010', 'C2682', 'C4599', 'C2268', 'C5412_RF', 'C2645', 'C5502_RF', 'C5603_RF', 'C2030', 'C2070', 'C2197', 'C2225', 'C1401', 'C2134', 'C5216_RF', 'C3024', 'C1434', 'C7003', 'C7014', 'C3096', 'C2099', 'C2214', 'C2282', 'C2468', 'C3018', 'C5222_RF', 'C5408_RF', 'C5219_RF', 'C2624', 'C2027', 'C1065', 'L2012', 'FL4583', 'Q4200', 'C1058', 'C1073', 'C1340', 'C1375', 'C2008', 'C2035', 'C2082', 'C2101', 'C2112', 'C2196', 'C3102', 'C5702_RF', 'C5201_RF', 'C5210_RF', 'C5402_RF', 'C6202_RF', 'R2700', 'RL2700', 'R8140', 'R8102', 'R2314', 'R2741', 'R6622', 'R4200', 'R1809', 'R1824', 'R1849', 'R2707', 'R8145', 'R8425', 'R8527', 'R9524', 'R2612A', 'R6303_RF', 'R4202', 'R1685', 'R8409', 'R9235', 'R3101', 'R3910', 'FL8160', 'R1810', 'R1846', 'R1853', 'R1858', 'R1863', 'R1868', 'R1873', 'R1878', 'R1883', 'R1874A', 'R1879B', 'R2703', 'R2701', 'R6527', 'R8608', 'R6526', 'R8923', 'R8928', 'R8413', 'C5221', 'C6602', 'C3905', 'C2719', 'C7056', 'C2663', 'C2703', 'C2287', 'C2728', 'C8111', 'C8545', 'C8602', 'C8145', 'C9523', 'C6101_RF', 'C8401', 'C2335', 'C7408', 'C2750', 'C2609', 'C8140', 'C9764', 'C2727', 'C8101', 'C8164', 'C8407', 'C8427', 'C8528', 'C8544', 'C8914', 'C9211', 'C9221', 'C9241', 'C9522', 'C9533', 'C8923', 'C8928', 'C6506', 'C8107', 'C6105_RF', 'C2650', 'C6523', 'C5307_RF', 'C9541', 'C6601', 'C2621', 'C7401', 'C9219', 'C3904', 'C2751', 'C2723', 'C2620', 'C2633', 'C2125', 'C6513', 'C2725', 'C2718', 'C2700', 'C2732', 'C2780', 'C2785', 'C5225_RF', 'C2717', 'C6521', 'C2812', 'C3909', 'C2705', 'C2712', 'L8402', 'L6601', 'R7458', 'R7422', 'L2702', 'L2701', 'FL8412', 'FL9532', 'FL9214', 'FL8920', 'FL8922', 'CP7401', 'L8100', 'FL9217', 'FL7401', 'L2704', 'R9741', 'L8400', 'FL8915', 'FL9222', 'FL3100', 'U2701', 'U3900', 'U6500', 'U3101', 'U2710', 'U6600', 'U_DSM_RF', 'U_DPDT_RF', 'Q4202', 'Q9501', 'Q2810', 'Q2701', 'DZ8914', 'DZ9200', 'C7013', 'C7622', 'R7471', 'PD_SF_ETIC', 'PD_SF_RFFE', 'C9741', 'C2706', 'C8500', 'D1601', 'Q1601', 'R6604', 'C7624', 'L2600', 'U7401', 'R1830', 'R1835', 'R1890', 'R1873B', 'C1602', 'C1603', 'U5302_RF', 'U2700', 'R6602', 'R6614', 'R6617', 'R7460', 'R7451', 'L6613', 'U1600', 'Z6111_RF', 'Z6107_RF', 'U6302_RF', 'R1841', 'R6615', 'FL6600', 'R8141', 'R8103', 'R2804', 'R3110', 'R8414', 'R1800', 'R1820', 'R1825', 'R1910', 'R2718', 'R8146', 'R8426', 'R8914', 'R9540', 'R5317_RF', 'R6000_RF', 'R6305_RF', 'R1668', 'R1686', 'R8415', 'FL8161', 'R1842', 'R1847', 'R1854', 'R1859', 'R1864', 'R1869', 'R1874', 'R1879', 'R1870A', 'R1875A', 'R1870D', 'R7502', 'R2802', 'R2702', 'R6528', 'R8609', 'R8924', 'R8929', 'C6501', 'C2666', 'C2600', 'C2768', 'C8409', 'C8550', 'C9204', 'C8146', 'C9524', 'C8402', 'C7409', 'C6605', 'C8141', 'C2769', 'C8102', 'C8165', 'C8420', 'C8520', 'C8530', 'C8603', 'C8916', 'C9212', 'C9222', 'C9511', 'C9526', 'C9540', 'C8924', 'C8929', 'C6516', 'C8108', 'C5909_RF', 'C2651', 'C2702', 'C6524', 'C6100_RF', 'C9750', 'C6603', 'C5306_RF', 'C8408', 'C2601', 'C2667', 'C2628', 'C6531', 'C2701', 'C2733', 'C2781', 'C2786', 'C3900', 'C2822', 'C3110', 'C2707', 'C2724', 'L8403', 'L2703', 'FL9521', 'FL9533', 'FL9215', 'FL8921', 'FL8923', 'L8101', 'R9742', 'L8401', 'FL8916', 'Q4203', 'Q2820', 'DZ9201', 'C7623', 'C2664', 'C2665', 'C9742', 'C2708', 'C8501', 'C7627', 'R1831', 'R1836', 'R1891', 'R1874B', 'R7452', 'R6618', 'R7450', 'L6616', 'R2335', 'R8110', 'R2600', 'R6524', 'R1821', 'R1826', 'R1911', 'R7421', 'R8162', 'R8427', 'R9240', 'R9750', 'Z6001_RF', 'R1669', 'R1687', 'R8421', 'R1843', 'R1850', 'R1855', 'R1860', 'R1865', 'R1870', 'R1875', 'R1880', 'R1871A', 'R1876A', 'R7503', 'R2823', 'R8903', 'R8610', 'R8925', 'R8930', 'C6511', 'C5223_RF', 'C2778', 'C3843', 'C8411', 'C8551', 'C9205', 'C8410', 'C7410', 'C6106_RF', 'C8403', 'C9761', 'C2779', 'C8103', 'C8192', 'C8421', 'C8525', 'C8531', 'C8605', 'C8917', 'C9216', 'C9223', 'C9517', 'C9530', 'C9751', 'C8925', 'C8930', 'C6544', 'C2652', 'C7017', 'C6102_RF', 'C8914A', 'C6604', 'C6309_RF', 'C8413', 'C6502', 'C6541', 'C2704', 'C2782', 'C2787', 'C2610', 'C3053', 'C3908', 'C3051', 'C2709', 'C2734', 'L8404', 'R7496', 'FL9522', 'R9743', 'L8600', 'FL8917', 'Q4204', 'DZ9540', 'C9743', 'C3000', 'C8502', 'R1832', 'R1837', 'R1870B', 'R1875B', 'R7457', 'C9213', 'R8111', 'R2601', 'R6525', 'R5410_RF', 'R1822', 'R1827', 'R1912', 'R7423', 'R8166', 'R8525', 'R9241', 'R2610A', 'R1688', 'R8501', 'R1844', 'R1851', 'R1856', 'R1861', 'R1866', 'R1871', 'R1876', 'R1881', 'R1872A', 'R1877A', 'R8931', 'R8603', 'R9200', 'R8926', 'R8932', 'C6533', 'C3912', 'C8415', 'C8552', 'C9510', 'C8412', 'C6002_RF', 'C8404', 'C9762', 'C8147', 'C8194', 'C8425', 'C8526', 'C8534', 'C8612', 'C8918', 'C9217', 'C9235', 'C9518', 'C9531', 'C8926', 'C8931', 'C2653', 'C2625', 'C8414', 'C2811', 'C2115', 'C6503', 'C6542', 'C2730', 'C2783', 'C3907', 'C6550', 'C3054', 'C3913', 'C3052', 'C2710', 'C2735', 'L8405', 'R7498', 'FL9530', 'R9744', 'FL9220', 'Q4205', 'DZ9750', 'C9744', 'C3803', 'C8503', 'R1833', 'R1838', 'R1871B', 'R1876B', 'R7470', 'R2740', 'R6601', 'R5411_RF', 'R1823', 'R1848', 'R2607', 'R7443', 'R8167', 'R8526', 'R9523', 'R2611A', 'R5900_RF', 'R6302_RF', 'R8602', 'R1845', 'R1852', 'R1857', 'R1862', 'R1867', 'R1872', 'R1877', 'R1882', 'R1873A', 'R1878B', 'R7453', 'R8927', 'C6543', 'C2685', 'C8110', 'C8535', 'C8601', 'C9516', 'C8416', 'C6003_RF', 'C9763', 'C8148', 'C8406', 'C8426', 'C8527', 'C8540', 'C8901', 'C9200', 'C9218', 'C9240', 'C9521', 'C9532', 'C8927', 'C8106', 'C2810', 'C8604', 'C2821', 'C6512', 'C6555', 'C2731', 'C2784', 'C5224_RF', 'C2614', 'C3120', 'C3055', 'C2711', 'FL9531', 'FL9221', 'Q9500', 'C3901', 'R1834', 'R1839', 'R1872B', 'R7497', 'R7400', 'R6201_RF', 'R1801E', 'R1806E', 'R1811E', 'R1816E', 'R1821E', 'R1826E', 'R6412_RF', 'L6411_RF', 'U6411_RF', 'FL6201_RF', 'R1802E', 'R1807E', 'R1812E', 'R1817E', 'R1822E', 'R1827E', 'R1828E', 'C6412_RF', 'C6411_RF', 'R1803E', 'R1808E', 'R1813E', 'R1818E', 'R1823E', 'R1829E', 'R1804E', 'R1809E', 'R1814E', 'R1819E', 'R1824E', 'R1830E', 'C4550', 'R1805E', 'R1810E', 'R1815E', 'R1820E', 'R1825E', 'R1831E']
    component = component.split(" ")[0]
    component = component.upper()
    if  component in golden_name:
        return component

    candidate = component + "_RF"
    if candidate in golden_name:
        return candidate

    return component

def check_caterory(classify):


    dfm_key = [
        "SHORT","OPEN","STRING","MISSING","CRACK","FM","BROKEN",
        "EXCESSIVE","CHIP","DAMAGED","POOR","HIP","REVERSE",
        "VOID","NON-WETTING","BENDING","DEPRESSION"
    ]
    monitor_list = ["UNDER FA", "UNDER FA CHECK-IN"]
    if classify.upper() in monitor_list:
        return ""

    if not classify:
        return ""

    list_classify = str(classify).upper().split()

    if "ISSUE" in list_classify:
        return "MAT"

    if "NTF" in list_classify:
        return "NTF"

    for key in list_classify:
        if key in dfm_key:
            return "DFM"

    return "KEEP"

def split_component_classify(row):

    component_value = row.get("COMPONENT")
    sn = row.get("Serial Number")
    monitor_list = ["UNDER FA", "UNDER FA CHECK-IN"]

    if isinstance(component_value, float):
        print("Đây là float: ", sn)
        return row

    if component_value.upper() in monitor_list:
        return row

    if pd.isna(component_value):
        return row

    parts = str(component_value).strip().split()
    if not parts:
        return row

    row["COMPONENT"] = parts[0].upper()
    classify_value = row.get("CLASSIFY")
    if not pd.isna(classify_value) and str(classify_value).strip() != "":
        return row

    elif len(parts) > 1:
        row["CLASSIFY"] = " ".join(parts[1:])
    else:
        row["CLASSIFY"] = parts[0].upper()

    return row

def process_excel():
    global input_excel_path, output_excel_path,start_time
    input_excel_path = entry_excel_window.get().strip()
    output_excel_path = output_excel_window.get().strip()
    ordered_columns = [
    "SERIAL NUMBER", "PHASE", "MODEL", "LINE", "CONFIG",
    "PANEL SN", "SN SEQ IN PANEL",
    "FAILED DATE","LAB IN TIME", "GROUP NAME", "LIST OF FAILING TESTS",
    "FAILURE MESSAGE", "SYMPTOM", "RADAR NO", "COMPONENT",
    "CLASSIFY", "CATERORY", "DETAIL FA", "VENDOR",
    "DATE CODE", "LOT NO", "COMPONENT SN", "LOG KEY", "DFM NET",
    "FE-VQA FEEDBACK", "REMARK"
    ]

    start_query = datetime.now()
    df_input = pd.read_excel(input_excel_path)

    df_input.columns = df_input.columns.str.upper().str.strip()

    required_cols = ["SERIAL NUMBER", "COMPONENT"]
    for col in required_cols:
        if col not in df_input.columns:
            raise ValueError(f"Thiếu cột bắt buộc: {col}")
    
    df_input = df_input.apply(split_component_classify,axis=1)

    output_rows = []

    count_sip = 0
    for _, row in df_input.iterrows():
        count_sip += 1
        print(count_sip)
        vendor_data = {}
        traveler_data = {}
        serial = str(row["SERIAL NUMBER"]).strip()
        component = row.get("COMPONENT", "")

        new_row = row.to_dict()
        classify = new_row.get("CLASSIFY","")
        caterory = new_row.get("CATERORY","")

        # Query SFIS

        new_session = create_sfis_session()
        traveler_data = query_sfis_inputdata(new_session,serial)
        if not pd.isna(component) and str(component).strip() != "":
            component = adjust_component_name(component)
            vendor_data   = query_sfis_vendor(new_session, serial, location=component)
            print("SN:", serial)
            if not pd.isna(classify) and str(classify).strip() != "":
                new_row["CATERORY"] = check_caterory(classify)
        else:
            print("Don't have component value")
        final_row = {
            **traveler_data,
            **vendor_data
        }
        new_row.update(final_row)

        if isinstance(new_row, dict):
            for k, v in new_row.items():
                new_row[k] = v

        new_row["SERIAL NUMBER"] = serial
        new_row["COMPONENT"] = component

        output_rows.append(new_row)

    df_output = pd.DataFrame(output_rows)
    df_output.columns = df_output.columns.str.upper().str.strip()
    for col in ordered_columns:
        if col not in df_output.columns:
            df_output[col] = ""

    remaining_cols = [c for c in df_output.columns if c not in ordered_columns]
    df_output = df_output[ordered_columns + remaining_cols]
    df_output.to_excel(output_excel_path, index=False)

    print("start time: ", start_time)
    print("start query:", start_query)
    stop_time = datetime.now()
    print("stop time", stop_time)
    messagebox.showinfo(
        "Completed",
        f"Query completed!\n\nFile has been saved to:\n{output_excel_path}"
    )

# =============================
# Tkinter UI
# =============================
def input_botton():
    file_path = filedialog.askopenfilename(
        title="Choose Excel file",
        filetypes=[
            ("Excel files", "*.xlsx *.xls"),
            ("All files", "*.*")
        ]
    )

    if not file_path:
        return

    # Show path on Entry
    entry_excel_window.delete(0, tk.END)
    entry_excel_window.insert(0, file_path)

def output_botton():
    folder_path = filedialog.askdirectory(
        title="Choose Output Folder"
    )

    if folder_path:
        today_str = datetime.now().strftime("%Y%m%d")
        file_name = f"FA detail_{today_str}.xlsx"

        full_path = os.path.join(folder_path, file_name)

        output_excel_window.delete(0, tk.END)
        output_excel_window.insert(0, full_path)

root = tk.Tk()
root.title("Get Data For FA Detail")
root.geometry("500x200")

input_frame = ttk.Frame(root)
input_frame.pack(pady=10)


input_btn_browse = ttk.Button(input_frame, text="Input File", command=input_botton)
input_btn_browse.grid(row=0, column=1, padx=5, pady=5)
entry_excel_window = ttk.Entry(input_frame, width=60)
entry_excel_window.grid(row=0, column=3, padx=5, pady=5)

output_btn_browse = ttk.Button(input_frame, text="Output File", command=output_botton)
output_btn_browse.grid(row=1, column=1, padx=5, pady=5)
output_excel_window = ttk.Entry(input_frame, width=60)
output_excel_window.grid(row=1, column=3, padx=5, pady=5)

input_excel_path = entry_excel_window.get().strip()
output_excel_path = output_excel_window.get().strip()

# Search button
btn_search = ttk.Button(root, text="Search", command=process_excel)
btn_search.pack(pady=10)

note_label = ttk.Label(
    root,
    text=(
        "Note:\n"
        "• SERIAL NUMBER and COMPONENT colums are required in excel file.\n"
    ),
    justify="left",
    foreground="gray"
)
note_label.pack(fill="x", padx=10, pady=10)

root.mainloop()