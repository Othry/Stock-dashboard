import sqlite3
import hashlib
import json

DB_NAME = "Vektor.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password_hash TEXT,
            is_approved INTEGER DEFAULT 0 
        )
    ''')
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS portfolios (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            name TEXT,
            data TEXT,
            FOREIGN KEY(username) REFERENCES users(username)
        )
    ''')
    conn.commit()
    conn.close()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def create_user(username, password):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password_hash, is_approved) VALUES (?, ?, 0)", 
                  (username, hash_password(password)))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def check_login(username, password):
    """Prüft Passwort UND ob der User freigeschaltet ist."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT password_hash, is_approved FROM users WHERE username = ?", (username,))
    data = c.fetchone()
    conn.close()
    
    if data:
        stored_hash = data[0]
        approved = data[1]
        
        if stored_hash == hash_password(password):
            if approved == 1:
                return "OK"
            else:
                return "NOT_APPROVED" 
                
    return "FAIL"


def approve_user(username):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("UPDATE users SET is_approved = 1 WHERE username = ?", (username,))
    conn.commit()
    conn.close()

def delete_user_full(username):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("DELETE FROM portfolios WHERE username = ?", (username,))
    c.execute("DELETE FROM users WHERE username = ?", (username,))
    success = c.rowcount > 0
    conn.commit()
    conn.close()
    return success


def save_portfolio_db(username, pf_name, pf_data):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    json_data = json.dumps(pf_data)
    c.execute("SELECT id FROM portfolios WHERE username = ? AND name = ?", (username, pf_name))
    exists = c.fetchone()
    if exists:
        c.execute("UPDATE portfolios SET data = ? WHERE username = ? AND name = ?", (json_data, username, pf_name))
    else:
        c.execute("INSERT INTO portfolios (username, name, data) VALUES (?, ?, ?)", (username, pf_name, json_data))
    conn.commit()
    conn.close()

def load_portfolios_db(username):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT name, data FROM portfolios WHERE username = ?", (username,))
    rows = c.fetchall()
    conn.close()
    portfolios = {}
    for name, data_str in rows:
        portfolios[name] = json.loads(data_str)
    return portfolios

def delete_portfolio_db(username, pf_name):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("DELETE FROM portfolios WHERE username = ? AND name = ?", (username, pf_name))
    conn.commit()
    conn.close()

init_db()