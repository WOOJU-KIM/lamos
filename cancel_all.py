import os
import json
import requests
import time
from dotenv import dotenv_values

def get_all_open_orders():
    env = dotenv_values(".env")
    account_no = env.get("KIWOOM_MOCK_ACCOUNT_NO")
    appkey = env.get("KIWOOM_MOCK_APP_KEY")
    appsecret = env.get("KIWOOM_MOCK_APP_SECRET")
    base_url = "https://mockapi.kiwoom.com"
    
    url = f"{base_url}/oauth2/token"
    payload = json.dumps({"grant_type": "client_credentials", "appkey": appkey, "secretkey": appsecret})
    headers = {"Content-Type": "application/json;charset=UTF-8"}
    resp = requests.post(url, data=payload, headers=headers)
    token = resp.json().get("token") or resp.json().get("access_token")
    
    orders = []
    next_key = ""
    
    url = f"{base_url}/api/us/acnt"
    while True:
        headers = {
            "Content-Type": "application/json;charset=UTF-8",
            "authorization": f"Bearer {token}",
            "api-id": "ust21050",
            "cont-yn": "N" if not next_key else "Y",
            "next-key": next_key
        }
        body = {"cano": account_no, "acnt_prdt_cd": "01", "qry_tp": "0"}
        res = requests.post(url, data=json.dumps(body), headers=headers).json()
        
        batch = res.get("result_list", [])
        if batch:
            orders.extend(batch)
            
        next_key = res.get("next_key")
        if not next_key or len(next_key.strip()) == 0:
            break
        time.sleep(0.5)
        
    print(f"Total open orders found: {len(orders)}")
    for o in orders:
        ord_no = o.get("ord_no")
        stk_cd = o.get("stk_cd")
        ord_qty = o.get("ord_qty")
        print(f"[{stk_cd}] Order No: {ord_no}, Qty: {ord_qty}")
        
        # Cancel
        sym_clean = stk_cd.upper().strip()
        stex = "NY" if sym_clean in ["SOXL", "SOXS", "SPY", "DIA", "VIXY", "VICI", "LEN", "ROL", "MCD", "CCL", "PCG", "ROP", "EFX", "INTU", "CLX", "ZTS", "OTIS", "GIS", "HD"] else "ND"
        
        c_body = {
            "cano": account_no,
            "acnt_prdt_cd": "01",
            "orig_ord_no": ord_no,
            "stex_tp": stex,
            "stk_cd": sym_clean,
            "ord_qty": "0",
            "ord_uv": "0",
            "trde_tp": "00"
        }
        headers["api-id"] = "ust20003"
        c_res = requests.post(f"{base_url}/api/us/ordr", data=json.dumps(c_body), headers=headers).json()
        print(f"Cancel {stk_cd}: {c_res.get('return_msg')}")
        time.sleep(0.5)

if __name__ == "__main__":
    get_all_open_orders()
