try:
    from dotenv import load_dotenv
    import os

    load_dotenv()

    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    NVIDIA_NIM = os.getenv("NVIDIA_NIM")

    os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
    os.environ["NVIDIA_NIM"] = NVIDIA_NIM

    MODEL_NAME = "meta/llama-3.1-70b-instruct"

    MAX_HISTORY = 5

    index_name = "medical-chatbot"
except Exception as e:
    print("Error loading environment variables:", e)