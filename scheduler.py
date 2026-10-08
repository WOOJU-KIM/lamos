import schedule
import time
from datetime import datetime
import pytz
from scanner import run_premarket_scan
import subprocess

def is_dst():
    # 미국 동부(New York) 기준 서머타임 확인
    tz = pytz.timezone("America/New_York")
    now = datetime.now(tz)
    return now.dst() != type(now.dst())(0)

def market_open_logic():
    # 장 시작 30분 전 스캔 시작 (서머타임: 22:00, 해제시: 23:00)
    print("Pre-market logic triggered.")
    run_premarket_scan()
    print("Scanner finished. Launching Kiwoom Trader...")
    
    # 32비트 파이썬 환경이 필요하므로 trader.py를 별도로 실행 (필요 시 32bit python 경로로 수정)
    # subprocess.Popen(["python", "trader.py"])

def setup_schedule():
    # 매일 오후 9시 30분에 서머타임 여부 체크 후 동적 스케줄링
    # 여기서는 심플하게 고정된 시간으로 체크합니다.
    while True:
        now = datetime.now()
        dst = is_dst()
        # 장 시작은 KST 기준 서머타임 22:30, 아닐시 23:30.
        # 장 시작 30분 전에 스캔을 실행합니다.
        target_hour = 22 if dst else 23
        
        if now.hour == target_hour and now.minute == 0 and now.second == 0:
            market_open_logic()
            time.sleep(1) # 중복 실행 방지
            
        time.sleep(1)

if __name__ == "__main__":
    print("Quant Scheduler Started...")
    setup_schedule()
