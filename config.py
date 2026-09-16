import os
from pathlib import Path
from dotenv import load_dotenv

# .env 로드
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# 퀀트 헤지펀드 핵심 설정 파라미터
INITIAL_CAPITAL_KRW = 10_000_000  # 초기 운용 자본금 10,000,000원
KILL_SWITCH_VIX_SPIKE_PCT = 15.0  # VIX 장중 +15% 이상 급등 시 킬스위치 발동
KILL_SWITCH_SOXL_DROP_PCT = -7.0  # SOXL 당일 -7% 이상 급락 시 킬스위치 발동

# ==============================================================================
# 한국투자증권(KIS) OpenAPI 듀얼 환경 설정 (VIRTUAL vs REAL)
# ==============================================================================
KIS_MODE = os.getenv("KIS_MODE", "VIRTUAL").upper()

# 1. 모의투자 (VIRTUAL)
KIS_VIRTUAL_BASE_URL = os.getenv("KIS_VIRTUAL_BASE_URL", "https://openapivts.koreainvestment.com:29443")
KIS_VIRTUAL_APP_KEY = os.getenv("KIS_VIRTUAL_APP_KEY") or os.getenv("KIS_APP_KEY")
KIS_VIRTUAL_APP_SECRET = os.getenv("KIS_VIRTUAL_APP_SECRET") or os.getenv("KIS_APP_SECRET")
KIS_VIRTUAL_CANO = os.getenv("KIS_VIRTUAL_CANO", "50201878")
KIS_VIRTUAL_ACNT_PRDT_CD = os.getenv("KIS_VIRTUAL_ACNT_PRDT_CD", "01")

# 2. 실전투자 (REAL)
KIS_REAL_BASE_URL = os.getenv("KIS_REAL_BASE_URL", "https://openapi.koreainvestment.com:9443")
KIS_REAL_APP_KEY = os.getenv("KIS_REAL_APP_KEY")
KIS_REAL_APP_SECRET = os.getenv("KIS_REAL_APP_SECRET")
KIS_REAL_CANO = os.getenv("KIS_REAL_CANO", "10031896")
KIS_REAL_ACNT_PRDT_CD = os.getenv("KIS_REAL_ACNT_PRDT_CD", "01")

PORTFOLIO_STATE_FILE = DATA_DIR / "portfolio_state.json"
EVOLUTION_LOG_FILE = DATA_DIR / "evolution_log.json"
SHADOW_RND_CACHE_FILE = DATA_DIR / "shadow_rnd_cache.json"
TRADE_LOGS_CSV = DATA_DIR / "trade_logs.csv"
TRADE_LOGS_SUMMARY_JSON = DATA_DIR / "trade_logs_summary.json"

def validate_env():
    """필수 환경변수 존재 여부 검증"""
    missing = []
    if not TELEGRAM_BOT_TOKEN:
        missing.append("TELEGRAM_BOT_TOKEN")
    if not TELEGRAM_CHAT_ID:
        missing.append("TELEGRAM_CHAT_ID")
    if not GEMINI_API_KEY:
        missing.append("GEMINI_API_KEY")
    
    if missing:
        raise ValueError(f"필수 환경 변수가 누락되었습니다: {', '.join(missing)}")
    return True
