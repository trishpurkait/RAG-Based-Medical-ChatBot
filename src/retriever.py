from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from src.config import index_name

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

search=PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings)

retriever = search.as_retriever(search_type="similarity", search_kwargs={"k": 3})