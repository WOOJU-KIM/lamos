import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Windows 콘솔 utf-8 설정
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

from core.weekly_tournament import WeeklyTournament
from core.system_logger import system_logger

def main():
    print("=" * 80)
    print("🏆 [Lumos OS 정기 스케줄러: 주간 3자 토너먼트 거버넌스 가동]")
    print("=" * 80)
    system_logger.log("INFO", "CronTournament", "🏆 주간 3자 토너먼트 거버넌스 스케줄러 실행 개시")

    try:
        tournament = WeeklyTournament()
        result = tournament.run_tournament()
        winner = result.get("winner", {})
        print(f"\n✅ [토너먼트 완료] 선발 챔피언: {winner.get('candidate_type')} ({winner.get('model_id')})")
        system_logger.log("INFO", "CronTournament", f"✅ 토너먼트 완료: 1위 {winner.get('candidate_type')} 선정")
    except Exception as e:
        print(f"\n❌ [토너먼트 실행 실패]: {e}")
        system_logger.log("ERROR", "CronTournament", f"⚠️ 토너먼트 실행 예외: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
