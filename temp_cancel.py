import os
import json
import requests
from dotenv import dotenv_values

def cancel_all():
    env = dotenv_values(".env")
    account_no = env.get("KIWOOM_MOCK_ACCOUNT_NO")
    appkey = env.get("KIWOOM_MOCK_APP_KEY")
    appsecret = env.get("KIWOOM_MOCK_APP_SECRET")
    base_url = "https://mockapi.kiwoom.com"
    
    # Get token
    url = f"{base_url}/oauth2/token"
    payload = json.dumps({"grant_type": "client_credentials", "appkey": appkey, "secretkey": appsecret})
    headers = {"Content-Type": "application/json;charset=UTF-8"}
    resp = requests.post(url, data=payload, headers=headers)
    token = resp.json().get("token") or resp.json().get("access_token")
    print("Token fetched:", token[:10] if token else "None")
    
    # Get open orders
    url = f"{base_url}/api/us/acnt"
    headers = {
        "Content-Type": "application/json;charset=UTF-8",
        "authorization": f"Bearer {token}",
        "api-id": "ust21050",
        "cont-yn": "N",
        "next-key": ""
    }
    body = {"cano": account_no, "acnt_prdt_cd": "01", "qry_tp": "0"}
    res = requests.post(url, data=json.dumps(body), headers=headers).json()
    print("Open Orders API Response:", res)
    orders = res.get("result_list", [])
    print(f"Found {len(orders)} open orders.")
    
    for ord_info in orders:
        ord_sym = str(ord_info.get("stk_cd") or "").strip().upper()
        ord_no = str(ord_info.get("ord_no") or "").strip()
        stex_nm = str(ord_info.get("stex_nm") or "").strip()
        
        stex_tp = "NY" if ord_sym in ["SOXL", "SOXS", "SPY", "DIA", "VIXY", "VICI", "LEN", "ROL", "MCD", "CCL", "PCG", "ROP", "EFX", "INTU", "CLX", "ZTS", "OTIS", "GIS", "HD"] else "ND"
        
        print(f"Canceling {ord_sym} (Order No: {ord_no})")
        cancel_body = {
            "cano": account_no,
            "acnt_prdt_cd": "01",
            "orig_ord_no": ord_no,
            "stex_tp": stex_tp,
            "stk_cd": ord_sym,
            "ord_qty": "0",
            "ord_uv": "0",
            "trde_tp": "00"
        }
        headers["api-id"] = "ust20003"
        c_res = requests.post(f"{base_url}/api/us/ordr", data=json.dumps(cancel_body), headers=headers).json()
        print("Cancel Result:", c_res)
        import time
        time.sleep(0.5)

if __name__ == "__main__":
    cancel_all()
