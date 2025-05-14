from model.base_llm import BaseLLM
from model.utils import create_message, add_message_to_history
import logging

DONE_SEQUENCE = "<!DONE!>"


class ReflectionAgent:
    def __init__(self, llm: BaseLLM):
        self.llm = llm
        self.generation_system_prompt = """You are an expert content generator. Your goal is to produce the highest-quality content that fully satisfies the user's request.
- If the user provides feedback or critique, revise your previous output accordingly.
- Always output the complete, improved version based on the latest input.
- Avoid repeating previous mistakes and aim for clarity, accuracy, and relevance."""
        self.reflection_system_prompt = f"""You are a thoughtful critic tasked with reviewing the user's generated content.
- Identify any errors, inconsistencies, or areas for improvement.
- Provide a clear, concise list of critiques and actionable recommendations.
- If the content is satisfactory and requires no changes, only then respond with: {DONE_SEQUENCE}"""

    def generate(self, user_msg: str, max_steps: int = 10) -> str:
        generation_history = [
            create_message(self.generation_system_prompt, "system"),
            create_message(user_msg, "user"),
        ]

        reflection_history = [create_message(self.reflection_system_prompt, "system")]

        response = ""
        for i in range(max_steps):
            # Generate a response
            response = self.llm.generate(generation_history)
            # Add the generated response to the history as assistant
            add_message_to_history(
                generation_history, create_message(response, "assistant"), 2, 2
            )
            # ...and to reflection_history as user
            add_message_to_history(
                reflection_history, create_message(response, "user"), 1, 2
            )
            # Critique the generated response
            critique = self.llm.generate(reflection_history)
            # Check if critique was positive
            if DONE_SEQUENCE in critique:
                logging.info(f"{DONE_SEQUENCE} found, stopping reflection agent!")
                break
            # Add the messages with reverse roles
            add_message_to_history(
                generation_history, create_message(critique, "user"), 2, 2
            )
            add_message_to_history(
                reflection_history, create_message(critique, "assistant"), 1, 2
            )

        return response
