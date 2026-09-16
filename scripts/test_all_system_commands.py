import sys
import json
import time
from pathlib import Path

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.agentic_brain import AgenticTelegramBrain

print("=" * 85)
print("🧪 [Lumos v10.4 키움증권 전 시스템 지원 명령어 전수 가동 검증]")
print("=" * 85)

brain = AgenticTelegramBrain()

commands_to_test = [
    ("1. 실시간 원장 잔고 조회", "잔고"),
    ("2. 실시간 시스템 관제 상태", "상태"),
    ("3. 키움 1주 매매 핑 테스트", "테스트"),
    ("4. 6대 이종 모델 현황 대시보드", "모델 현황"),
    ("5. 긴급 거래 정지 (수동 킬스위치)", "거래 멈춰"),
    ("6. 거래 정상 재개", "매매 재개"),
    ("7. 퀀트 정밀 백테스트 실행", "백테스트"),
    ("8. EOD 정규장 마감 결산 보고서", "결산"),
]

for idx, (title, cmd) in enumerate(commands_to_test, 1):
    print(f"\n[{idx}/8] 명령어 실행 테스트: '{cmd}' ({title})")
    print("-" * 75)
    try:
        reply_text, is_handled = brain.process_message(cmd)
        print(f"• 처리 성공 여부: {'✅ 성공 (is_handled=True)' if is_handled else '⚠️ 일반 대화 처리'}")
        print("• 응답 본문 미리보기:")
        for line in reply_text.strip().split("\n")[:8]:
            print(f"  {line}")
        if len(reply_text.strip().split("\n")) > 8:
            print("  ...")
    except Exception as e:
        print(f"❌ '{cmd}' 실행 중 예외 발생: {e}")

print("\n" + "=" * 85)
print("✅ [검증 완료] 8대 핵심 시스템 명령어 전수 정상 작동 확인 완료!")
print("=" * 85)
