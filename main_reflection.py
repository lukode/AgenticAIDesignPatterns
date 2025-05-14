import logging

from model_openai.openai_llm import OpenAILLM
from reflection.reflection_agent import ReflectionAgent


def run_reflection_agent():
    """
    Run the reflection agent to generate content and reflect on it.

    Parameters:
    None

    Returns:
    None
    """
    llm = OpenAILLM(
        "meta-llama/llama-3.2-3b-instruct/fp-16", "https://api.inference.net/v1"
        # "meta-llama/llama-3.3-70b-instruct/fp-16", "https://api.inference.net/v1"
    )
    reflection_agent = ReflectionAgent(llm)

    user_prompt = """Generate step-by-step instructions for setting up a basic web server using Node.js"""

    response = reflection_agent.generate(user_prompt, max_steps=4)
    logging.info("The response is:")
    logging.info(response)


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_reflection_agent()
