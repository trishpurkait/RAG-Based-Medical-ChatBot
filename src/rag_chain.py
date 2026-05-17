try:
    from src.llm import llm
    from src.prompt import prompt


    def generate_answer(question, docs):

        context = "\n\n".join([
            doc.page_content
            for doc in docs
        ])

        chain = prompt | llm

        response = chain.invoke({
            "context": context,
            "input": question
        })

        return response.content
    
except Exception as e:
    print("Error initializing answer generation:", e)