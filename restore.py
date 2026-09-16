import codecs

with codecs.open('core/live_runner.py', 'r', 'utf-8') as f:
    lines = f.readlines()

new_lines = []
skip = False

for i in range(len(lines)):
    line = lines[i]
    stripped = line.strip()
    
    if '? **???3-Out (Veto)  ??*' in line or '??**? ? ??** `{self.daily_stoploss_count}?? (? 3???)' in line or '??**?  ?:** `???????  ? ?  (VETO)`' in line or '??** ? ?:** `100% ?  (? ? ? )`' in line:
        if '? **???3-Out (Veto)  ??*' in line:
            new_lines.append('            msg = f"""🚨 **일일 3-Out 서킷 브레이커 발동**\\n- 금일 손절 횟수: {self.daily_stoploss_count}/3\\n- 조치: 신규 매수 전면 차단 (VETO)\\n- 현금: 100% 보존"""\n')
        continue

    if '? **? :**' in line and '{self.broker.broker_name}' in line and i < 800:
        new_lines.append('            buy_msg = f"""{mode_title} 신규 매수 주문 (1차)\\n🚨 **브로커:** `{self.broker.broker_name} {self.broker.mode_str}`\\n💡 **종목:** `{symbol}`\\n📊 **수량:** `{target_qty}`\\n💰 **주문가:** `${order_px_1:.2f}`\\n\\n🔥 **[GBDT 모델 확신도]**\\n{score_block}\\n\\n🎯 **목표가:** `${tp_px:.2f}`\\n🛡️ **손절가:** `${sl_px:.2f}`\\n⏱️ **시간청산:** {time_stop_min}분"""\n')
        continue
    if '? **[DB  - ????]**' in line or '{score_block}' in line or '??**????** {time_stop_min}?""' in line:
        continue
        
    if '? **? :**' in line and '{self.broker.broker_name}' in line and i > 1100:
        new_lines.append('                    sell_msg = f"""{mode_title} 100% 매도 청산 완료\\n🚨 **브로커:** `{self.broker.broker_name} {self.broker.mode_str}`\\n💡 **사유:** `{reason_desc}`\\n📊 **종목/수량:** `{symbol} {quantity:,}주`\\n💰 **체결가:** `${latest_px:.2f}` (최종 수익률: {final_pnl_pct:+.2f}%)\\n🛡️ **사후 관리:** `100% 현금화 완료 (AI MoE 새 진입 대기)`"""\n')
        new_lines.append('                    self.dispatcher.send_telegram_message(sell_msg)\n')
        new_lines.append('                    if is_stoploss:\n')
        new_lines.append('                        self._record_stoploss()\n')
        new_lines.append('                    return True\n')
        new_lines.append('\n')
        new_lines.append('                if s_ord_no:\n')
        new_lines.append('                    self.broker.cancel_order(order_no=s_ord_no, symbol=symbol, quantity=rem_qty)\n')
        new_lines.append('                    time.sleep(0.5)\n')
        new_lines.append('\n')
        new_lines.append('        except Exception as e:\n')
        new_lines.append('            logger.error(f"매도 추격 주문 에러: {e}")\n')
        new_lines.append('            system_logger.log("ERROR", "SellChase", f"매도 추격 에러: {e}")\n')
        new_lines.append('            try:\n')
        new_lines.append('                self.experience_logger.record_error_event("LiveRunner", "SELL_CHASE_EXCEPTION", "ERROR", str(e))\n')
        new_lines.append('            except Exception:\n')
        new_lines.append('                pass\n')
        new_lines.append('            return False\n')
        new_lines.append('        finally:\n')
        new_lines.append('            with self._order_lock:\n')
        new_lines.append('                self._is_order_in_progress = False\n')
        new_lines.append('                logger.info("🔒 [AI 포지션 청산 완료] 주문 락 해제완료")\n')
        skip = True
        continue
    
    if skip:
        if '    # ??[???????? ]' in line:
            skip = False
            new_lines.append(line)
        continue

    if stripped == '# fixed' or stripped == '??????????':
        continue
    if '? **?:**' in line and '{reason_desc}' in line:
        continue
    if '? **/?' in line and '{symbol}' in line:
        continue
    if '? **? ?:**' in line and '100% ????' in line:
        continue
        
    new_lines.append(line)

with codecs.open('core/live_runner.py', 'w', 'utf-8') as f:
    f.writelines(new_lines)
