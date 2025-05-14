import os

class BaseLLM:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def generate(self, messages: list) -> str:
        raise NotImplementedError("Subclasses should implement this method.")