import sqlite3
from database import DB_PATH

def print_top_candidates():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT rank, ticker, z_score FROM daily_candidates WHERE date = date('now') ORDER BY rank ASC LIMIT 10")
    rows = cursor.fetchall()
    
    print("\n[오늘의 최우선 매수 후보 Top 10]")
    for rank, tk, z in rows:
        print(f"{rank}위: {tk} (Z-Score: {z:.2f})")
    conn.close()

if __name__ == "__main__":
    print_top_candidates()
