from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from src.llm import llm
from src.prompt import prompt
from src.retriever import retriever



question_answering_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)

qa_chain = create_retrieval_chain(retriever, question_answering_chain)