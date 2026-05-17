try:
    from src.prompt import rewrite_prompt
    from src.llm import llm

    question_chain = rewrite_prompt | llm
except Exception as e:
    print("Error initializing question chain:", e)
