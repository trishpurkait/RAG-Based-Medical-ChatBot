from langchain_openai import ChatOpenAI
from src.config import *

try:
    llm = ChatOpenAI(
        model=MODEL_NAME,
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=NVIDIA_NIM,
        temperature=0.2
    )
except Exception as e:
    print("Error initializing LLM:", e)