import sqlite3
from database import DB_PATH

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
cursor.execute("SELECT rank, ticker, z_score FROM daily_candidates WHERE ticker = 'PEP'")
print(cursor.fetchall())
conn.close()
