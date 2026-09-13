"""
Persistent conversation memory using SQLite.

Replaces the in-memory Python list from the CLI version, which lost all
history the moment the script restarted. Each conversation gets a
session_id (a random string the frontend generates and keeps track of,
e.g. stored in the browser), so multiple separate chats don't bleed into
each other's history.

Run this once to create the database file:
    python database.py
"""

import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'chatbot_memory.db')


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # lets us access columns by name, e.g. row['content']
    return conn


def init_db():
    conn = get_connection()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,           -- 'user' or 'assistant'
            content TEXT NOT NULL,
            route TEXT,                   -- 'crisis' / 'normal' / 'low_confidence' - useful for later analysis
            label TEXT,                   -- the classifier's predicted label for this turn
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()
    print(f"Database ready at: {DB_PATH}")


def save_message(session_id: str, role: str, content: str, route: str = None, label: str = None):
    conn = get_connection()
    conn.execute(
        'INSERT INTO messages (session_id, role, content, route, label) VALUES (?, ?, ?, ?, ?)',
        (session_id, role, content, route, label)
    )
    conn.commit()
    conn.close()


def get_history(session_id: str) -> list:
    """Returns the conversation history for a session, formatted exactly
    how the Groq API expects it: [{"role": "user", "content": "..."}, ...]"""
    conn = get_connection()
    rows = conn.execute(
        'SELECT role, content FROM messages WHERE session_id = ? ORDER BY created_at ASC',
        (session_id,)
    ).fetchall()
    conn.close()
    return [{"role": row["role"], "content": row["content"]} for row in rows]


def get_all_sessions() -> list:
    """Returns a list of distinct session_ids that have any messages -
    useful later if you want to build a 'past conversations' list in the UI."""
    conn = get_connection()
    rows = conn.execute(
        'SELECT DISTINCT session_id, MIN(created_at) as started_at FROM messages GROUP BY session_id ORDER BY started_at DESC'
    ).fetchall()
    conn.close()
    return [{"session_id": row["session_id"], "started_at": row["started_at"]} for row in rows]


if __name__ == '__main__':
    init_db()