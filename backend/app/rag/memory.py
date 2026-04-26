from collections import deque

sessions: dict[str, deque] = {}


def get_history(session_id):
    if session_id not in sessions:
        sessions[session_id] = deque(maxlen=6)

    return sessions[session_id]


def add_message(session_id, role, content):
    history = get_history(session_id)

    history.append({"role": role, "content": content})


def delete_session(session_id):
    sessions.pop(session_id, None)
