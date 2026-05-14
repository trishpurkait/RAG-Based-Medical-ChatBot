from langchain_core.prompts import ChatPromptTemplate

system_prompt = """You are a helpful RAG assistant for answering questions related to medical topics.
Use the following retrieved documents to provide accurate and concise answers to the user's questions. 
Try to answer in simple language as much as possible. If you don't know the answer, say you don't know. 
Act like you are the one who is answering the question, do not say "the retrieved documents says". 
Just answer the question based on the retrieved documents.

Retrieved context:
{context}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

rewriteprompt = """
Conversation:
{chat_history}

Current question:
{input}

Rewrite the question so it is self-contained.
"""

rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system", rewriteprompt)
])