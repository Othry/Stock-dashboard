import sqlite3

DB_NAME = "mykoyfin.db"

conn = sqlite3.connect(DB_NAME)
c = conn.cursor()

try:
    c.execute("ALTER TABLE users ADD COLUMN is_approved INTEGER DEFAULT 0")
    print("Spalte 'is_approved' hinzugefügt.")
except sqlite3.OperationalError:
    print("Spalte existiert schon.")

c.execute("UPDATE users SET is_approved = 1")
print("Bestehende User wurden freigeschaltet.")

conn.commit()
conn.close()