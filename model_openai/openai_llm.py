import logging
import os
import dotenv
from openai import OpenAI
from model.base_llm import BaseLLM


class OpenAILLM(BaseLLM):
    def __init__(self, model_name: str, base_url: str):
        # self.model_name = model_name
        super().__init__(model_name)
        dotenv.load_dotenv()
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=base_url)

    def generate(self, messages: list) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
        )
        response_content = response.choices[-1].message.content
        logging.info(f"{response_content}")
        return response_content
