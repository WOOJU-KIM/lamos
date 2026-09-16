import sys
import json
import requests
from pathlib import Path

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.kis_client import KisClient

kis = KisClient(mode="VIRTUAL")
url = f"{kis.base_url}/uapi/overseas-stock/v1/trading/inquire-ccnl"
headers = kis._get_common_headers("VTTC8001R")

for excg in ["AMEX", "AMS", "NASD", "NYSE", "%", ""]:
    params = {
        "CANO": kis.cano,
        "ACNT_PRDT_CD": kis.acnt_prdt_cd,
        "PDNO": "%",
        "INQR_STRT_DT": "20260815",
        "INQR_END_DT": "20260820",
        "SLL_BUY_DVSN_CD": "00",
        "CCLD_DVSN": "00",
        "INQR_DVSN": "00",
        "INQR_DVSN_3": "00",
        "INQR_DVSN_1": "",
        "OVRS_EXCG_CD": excg,
        "SORT_SQN": "DS",
        "ORD_DT": "",
        "ORD_GNO_BRNO": "",
        "ODNO": "",
        "CTX_AREA_NK100": "",
        "CTX_AREA_FK100": "",
        "CTX_AREA_NK200": "",
        "CTX_AREA_FK200": ""
    }
    resp = requests.get(url, headers=headers, params=params, timeout=10)
    data = resp.json()
    out1 = data.get("output1", [])
    print(f"Excg: '{excg}' -> rt_cd={data.get('rt_cd')}, msg={data.get('msg1')}, output1 len={len(out1)}")
    for itm in out1:
        odno = itm.get('odno')
        dt = itm.get('ord_dt')
        tm = itm.get('ord_tmd')
        sym = itm.get('ovrs_pdno')
        side = 'BUY' if itm.get('sll_buy_dvsn_cd') == '02' else 'SELL'
        ord_qty = itm.get('ft_ord_qty') or itm.get('ord_qty')
        ccld_qty = itm.get('ft_ccld_qty') or itm.get('ccld_qty')
        px = itm.get('ft_ccld_unpr3') or itm.get('ovrs_ord_unpr')
        print(f"  [체결원장] 주문번호:{odno} | 일시:{dt} {tm} | {sym} {side} | 주문:{ord_qty}주, 체결:{ccld_qty}주 @ ${px}")
