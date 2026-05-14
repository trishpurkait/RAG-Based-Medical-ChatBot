from flask import Flask,request,render_template ,jsonify

from src.rag_chain import qa_chain
from src.question_chain import question_chain
from src.memory import (
    get_history,
    format_history,
    update_history,
    chat_sessions
)


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

        response = qa_chain.invoke({
            "input": question_chain.invoke({"input": question, "chat_history": format_history(history)}).content
        })

        answer = response["answer"]

        sources = list({
            doc.metadata.get("source", "Medical Knowledge Base")
            for doc in response.get("context", [])
        })

        update_history(session_id, question, answer)

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



