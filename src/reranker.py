try:
    from sentence_transformers import CrossEncoder

    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


    def rerank_documents(query, docs, top_k=3):

        pairs = [
            [query, doc.page_content]
            for doc in docs
        ]

        scores = reranker.predict(pairs)

        scored_docs = list(zip(scores, docs))

        scored_docs.sort(
            key=lambda x: x[0],
            reverse=True
        )

        return [doc for score, doc in scored_docs[:top_k]]
    
except Exception as e:
    print("Error initializing reranker:", e)