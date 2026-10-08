from trader import KiwoomRESTTrader

def print_pf():
    t = KiwoomRESTTrader()
    pf = t.get_portfolio()
    
    print("| 종목 | 보유수량 | 매입단가 | 현재가 | 수익률(%) | 평가금액 |")
    print("|---|---|---|---|---|---|")
    
    total_eval = 0
    total_buy = 0
    for p in pf:
        sym = p['symbol']
        qty = p['qty']
        avg_px = p['avg_price']
        now_px = p['now_price']
        ret = (now_px - avg_px) / avg_px * 100 if avg_px > 0 else 0
        eval_amt = qty * now_px
        buy_amt = qty * avg_px
        
        total_eval += eval_amt
        total_buy += buy_amt
        
        if ret > 0:
            ret_str = f"\033[91m{ret:+.2f}%\033[0m"
        elif ret < 0:
            ret_str = f"\033[94m{ret:+.2f}%\033[0m"
        else:
            ret_str = f"{ret:+.2f}%"
            
        print(f"| **{sym}** | {qty}주 | ${avg_px:,.2f} | ${now_px:,.2f} | {ret_str} | ${eval_amt:,.2f} |")

    total_ret = (total_eval - total_buy) / total_buy * 100 if total_buy > 0 else 0
    if total_ret > 0:
        total_ret_str = f"\033[91m{total_ret:+.2f}%\033[0m"
    elif total_ret < 0:
        total_ret_str = f"\033[94m{total_ret:+.2f}%\033[0m"
    else:
        total_ret_str = f"{total_ret:+.2f}%"

    print(f"\n**총 매입금액:** ${total_buy:,.2f}")
    print(f"**총 평가금액:** ${total_eval:,.2f}")
    print(f"**총 수익률:** {total_ret_str}")
    print(f"**남은 달러 예수금:** ${t.get_overseas_deposit():,.2f}")

if __name__ == '__main__':
    print_pf()
