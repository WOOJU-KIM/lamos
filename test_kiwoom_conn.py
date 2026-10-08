import json
from trader import KiwoomRESTTrader

def test_connection():
    print("========================================")
    print("키움증권 REST API (모의투자) 연결 테스트 시작")
    print("========================================")
    
    try:
        print("\n1. 토큰 발급 테스트...")
        trader = KiwoomRESTTrader()
        print(f"-> 발급된 토큰 (앞 20자리): {trader.access_token[:20]}...")
        
        print("\n2. 외화(USD) 예수금 조회 테스트 (ust21110)...")
        usd_cash = trader.get_overseas_deposit()
        print(f"-> 현재 주문 가능 USD 잔고: ${usd_cash:,.2f}")
        
        print("\n3. 해외주식 잔고 조회 테스트 (ust21070)...")
        portfolio = trader.get_portfolio()
        print(f"-> 현재 보유 종목 수: {len(portfolio)}개")
        for pos in portfolio:
            print(f"   - {pos['symbol']} : {pos['qty']}주 (평단가 ${pos['avg_price']}, 현재가 ${pos['now_price']})")
            
        print("\nO All API communication test finished successfully!")
    except Exception as e:
        print(f"\n[ERROR] API communication error: {e}")

if __name__ == "__main__":
    test_connection()
