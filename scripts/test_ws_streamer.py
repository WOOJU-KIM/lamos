import sys
import time
from pathlib import Path

if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.kiwoom_ws_streamer import KiwoomWebSocketStreamer

print("=== 키움 실시간 WebSocket 스트리머 시험 ===")
ws = KiwoomWebSocketStreamer()
print("1. WebSocket 접속 URL:", ws.ws_url)

def my_callback(sym, px, extra):
    print(f"⚡ [실시간 틱 수신 콜백]: {sym} -> ${px:.2f}")

ws.register_callback(my_callback)
ws.start()

print("2. WebSocket 백그라운드 리스너 가동 완료 (3초간 상태 점검)...")
time.sleep(3)
print(f"3. 연결 상태: is_connected={ws.is_connected}")
ws.stop()
print("=== WebSocket 시험 완료 ===")
