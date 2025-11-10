import os
import json
import time

SESSIONS_FILE = "sessions.json"

def _load():
    if not os.path.exists(SESSIONS_FILE):
        return {}
    with open(SESSIONS_FILE, "r") as f:
        return json.load(f)

def _save(data):
    with open(SESSIONS_FILE, "w") as f:
        json.dump(data, f, indent=2)

def create_session(title="New Chat"):
    data = _load()
    sid = f"session_{os.urandom(4).hex()}"
    now = int(time.time())
    data[sid] = {
        "id": sid,
        "title": title,
        "created_at": now,
        "last_active": now
    }
    _save(data)
    return data[sid]

def list_sessions():
    return list(_load().values())

def get_session(session_id):
    return _load().get(session_id)

def touch_session(session_id):
    data = _load()
    if session_id in data:
        data[session_id]["last_active"] = int(time.time())
        _save(data)

def delete_session_metadata(session_id):
    data = _load()
    if session_id in data:
        del data[session_id]
        _save(data)
