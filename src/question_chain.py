from src.prompt import rewrite_prompt
from src.llm import llm

question_chain = rewrite_prompt | llm
