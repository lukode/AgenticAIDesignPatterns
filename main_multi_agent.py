import logging
import datetime as dt
from model_openai.openai_llm import OpenAILLM
from multi_agent.group import Group
from multi_agent.member_agent import MemberAgent
from tool_use.llm_tool import convert_to_llm_tool as llm_tool


def submit_content_func(content: str):
    """
    Submits the content

    Parameters:
    content (str): The text content to be saved.

    Returns:
    Message confirming the content has been submitted
    """
    # timestamp the submission
    with open(f"submission_{dt.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.txt", "w") as f:
        f.write(content)
    return f"Content submitted successfully."


def run_content_moderation_system(topic: str, moderation_rules: list[str]):
    """
    Run the content moderation system with 3 agents:
    1. Post Writer Agent: Creates content about a given topic
    2. Content Moderator Agent: Rewrites content to comply with rules
    3. Content Submitter Agent: Saves the final content to a file

    Parameters:
    topic (str): The topic to write about
    moderation_rules (list[str]): List of content moderation rules to enforce
    """
    # Initialize the LLM
    llm = OpenAILLM(
        # "meta-llama/llama-3.2-3b-instruct/fp-16", "https://api.inference.net/v1"
        "meta-llama/llama-3.3-70b-instruct/fp-16", "https://api.inference.net/v1"
    )
    
    # Create tools for the content submitter agent
    llm_tools = [
        llm_tool(submit_content_func),
    ]
    
    # Format the moderation rules as a numbered list
    formatted_rules = "\n".join([f"{i+1}. {rule}" for i, rule in enumerate(moderation_rules)])
    
    # Create the group to manage the agents
    group = Group()
    
    # 1. Create the Post Writer Agent
    post_writer_agent = MemberAgent(
        llm,
        "Content Creator",
        "You are a creative content writer who can write engaging posts on any topic.",
        f"Write an brief message about the following topic: {topic}.",
        "A well-written message about the given topic in plain text.",
    )
    group.add_agent(post_writer_agent)
        
    # 2. Create the Content Moderator Agent
    content_moderator_agent = MemberAgent(
        llm,
        "Content Moderator",
        "You are an expert content editor, who can rewrite content to comply with moderation rules.",
        f"Review the content provided in the context and check if it contradicts any of the following rules:\n{formatted_rules}\n\nIf any rules are violated, rewrite the content to comply with all moderation rules, while preserving the original message as much as possible. If the content already complies with all rules, simply return the original content. Do not add any comments.",
        "A revised version of the content that complies with all moderation rules in plain text. No additional comments are needed.",
    )
    group.add_agent(content_moderator_agent)
    post_writer_agent.add_dependent(content_moderator_agent)
    
    # 3. Create the Content Submitter Agent
    content_submitter_agent = MemberAgent(
        llm,
        "Content Submitter",
        "You are responsible for submitting approved content to the appropriate channels.",
        f"Take the moderated content from the context and successfully submit it.",
        "Confirmation that the content has been successfully submitted.",
        llm_tools,
    )
    group.add_agent(content_submitter_agent)
    content_moderator_agent.add_dependent(content_submitter_agent)
    
    # Run the agent workflow
    result = group.generate()
    logging.info("Content moderation workflow complete.")
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    topic = "You are a nobleman in pre-Copernicus Europe. Answer in first person. According to your teachings, does the Earth rotate around the Sun or is the Sun rotating around the Earth?"
    topic = "Write a theorem about the Sun orbiting the Earth as in 10th century teachings - without any modern science knowledge. Don't mention who you are or where or when you live."
    moderation_rules = [
        "Earth orbits around the Sun",
        "The Sun is at the center of the solar system",
    ]
    
    run_content_moderation_system(topic, moderation_rules)
