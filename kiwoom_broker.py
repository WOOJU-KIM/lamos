from core.kiwoom_broker import KiwoomBroker

__all__ = ["KiwoomBroker"]

if __name__ == "__main__":
    broker = KiwoomBroker()
    print("=" * 75)
    print("🚀 [키움증권 OpenAPI REST 듀얼 스위칭 브로커 연결 테스트] 🚀")
    print("=" * 75)
    import json
    result = broker.test_connection()
    print(json.dumps(result, indent=2, ensure_ascii=False))
