from flask import Flask,request,render_template ,jsonify
from src.memory import (
    get_history,
    update_history,
    chat_sessions
)
from src.retriever import hybrid_retrieve
from src.rag_chain import generate_answer


app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():

    data = request.get_json(silent=True)
    if not data or "message" not in data:
        return jsonify({"error": "Missing 'message' field"}), 400
    
    session_id = data.get("session_id", "default")

    question = data["message"]
    if not question:
        return jsonify({"error": "Empty message"}), 400
    
    try:

        history = get_history(session_id)

        docs = hybrid_retrieve(question, history)
        print ("docs retrieved:", len(docs))
        answer = generate_answer(question, docs)
        print("Generated answer:", answer)
        update_history(session_id, question, answer)


        sources = list({
            doc.metadata.get("source", "Medical Knowledge Base")
            for doc in docs
        })

        return jsonify({"answer": answer, "sources": sources})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

@app.route("/clear", methods=["POST"])
def clear_history():

    data = request.get_json(silent=True) or {}
    session_id = data.get("session_id", "default")
    chat_sessions.pop(session_id, None)
    return jsonify({"status": "History cleared"})


if __name__ == "__main__":
    app.run(debug=True)


