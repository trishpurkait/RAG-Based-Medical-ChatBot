from src.config import MAX_HISTORY


chat_sessions = {}

def get_history(session_id):

    if session_id not in chat_sessions:
        chat_sessions[session_id] = []

    return chat_sessions[session_id]

def format_history(history):

    return "\n".join([
        f"User: {h['user']}\nAssistant: {h['assistant']}"
        for h in history
    ])


def update_history(session_id, question, answer):

    history = get_history(session_id)

    history.append({
        "user": question,
        "assistant": answer
    })

    chat_sessions[session_id] = history[-MAX_HISTORY:]