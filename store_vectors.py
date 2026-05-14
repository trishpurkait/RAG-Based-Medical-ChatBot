from src.loader import load_files, spilt_text, download_embeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os


load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

extracted_data = load_files(data='Data/')
chunks = spilt_text(extracted_data)
embeddings = download_embeddings()

pc = Pinecone()
index_name = "medical-chatbot"

pc.create_index(
    name=index_name,
    dimension=384,
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)


search=PineconeVectorStore.from_documents(chunks, embeddings, index_name=index_name)