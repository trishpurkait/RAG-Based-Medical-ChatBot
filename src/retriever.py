try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_pinecone import PineconeVectorStore
    from src.config import index_name
    from langchain_community.retrievers import BM25Retriever
    from src.reranker import rerank_documents
    from src.memory import format_history

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    search=PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings)

    def hybrid_retrieve(question, history):

        history_text = format_history(history)

        augmented_query = f"""
        Conversation:
        {history_text}

        Current Question:
        {question}
        """

        dense_docs = search.similarity_search(
            augmented_query,
            k=8
        )

        bm25 = BM25Retriever.from_documents(dense_docs)
        bm25.k = 5

        sparse_docs = bm25.invoke(question)
        merged_docs = dense_docs + sparse_docs

        unique_docs = []
        seen = set()

        for doc in merged_docs:

            text = doc.page_content

            if text not in seen:
                seen.add(text)
                unique_docs.append(doc)

        final_docs = rerank_documents(
            question,
            unique_docs,
            top_k=3
        )

        return final_docs

except Exception as e:
    print("Error initializing retriever:", e)