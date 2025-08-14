from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from ..config import settings
from ..utils.templates import PROMPT_TEMPLATE


class LLMService:
    def __init__(self):
        self.model = ChatGroq(model=settings.GROQ_MODEL, temperature=0)
        self.prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

        # Generate answer using LLM based on context

    def generate_answer(self, context: str, question: str, book_title: str) -> str:

        prompt = self.prompt_template.format(
            context=context, question=question, book_title=book_title
        )

        response = self.model.invoke(prompt)
        return response.content
