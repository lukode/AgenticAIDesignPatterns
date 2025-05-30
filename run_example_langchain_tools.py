"""
Helper script to run LLMWrapper integration tests directly.
"""

import asyncio
import json
import logging
import os

# Add the parent directory to the path so we can import from app
import uuid
from pathlib import Path
from typing import Any, Dict, List, Union

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

OPENAI_AGENT_API_KEY = "INFERENCE_API_KEY"
AGENT_MODEL = "meta-llama/llama-3.2-3b-instruct/fp-16"  # "meta-llama/llama-3.3-70b-instruct/fp-16"
AGENT_BASE_URL = "https://api.inference.net/v1"

# OPENAI_AGENT_API_KEY = "SAMBANOVA_API_KEY"
# AGENT_MODEL = "Meta-Llama-3.1-8B-Instruct"
# AGENT_BASE_URL = "https://api.sambanova.ai/v1/"

API_KEY_VALUE = os.environ.get(OPENAI_AGENT_API_KEY)


def deserialize_messages(
    serialized: Dict[str, List[Dict[str, Any]]],
) -> List[BaseMessage]:
    """
    Convert a list of serialized message dictionaries back to LangChain message objects.

    Args:
        serialized: Dictionary containing a "messages" key with a list of serialized messages

    Returns:
        List of LangChain message objects
    """
    message_map = {
        "HumanMessage": HumanMessage,
        "ToolMessage": ToolMessage,
        "AIMessage": AIMessage,
        "SystemMessage": SystemMessage,
    }

    messages = []
    for msg_dict in serialized.get("messages", []):
        msg_type = msg_dict["type"]
        if msg_type not in message_map:
            raise ValueError(f"Unknown message type: {msg_type}")

        msg_class = message_map[msg_type]
        content = msg_dict["content"]
        additional_kwargs = msg_dict.get("additional_kwargs", {})

        # Create message with type-specific parameters
        if msg_type == "ToolMessage":
            tool_call_id = msg_dict.get("tool_call_id", "")
            msg = msg_class(
                content=content, tool_call_id=tool_call_id, **additional_kwargs
            )
        else:
            if "tool_calls" in msg_dict:
                additional_kwargs["tool_calls"] = msg_dict["tool_calls"]
            msg = msg_class(content=content, **additional_kwargs)

        messages.append(msg)

    return messages


def load_messages_from_file(filepath: Union[str, Path]) -> List[BaseMessage]:
    """
    Load LangChain messages from a JSON file.

    Args:
        filepath: Path to the input file

    Returns:
        List of LangChain message objects
    """
    with open(filepath, "r", encoding="utf-8") as f:
        serialized = json.load(f)
    return deserialize_messages(serialized)


@tool
def get_current_temperature_func(location: str) -> str:
    """Get the current temperature in the specified location.

    Args:
        location: The location to get the temperature for

    Returns:
        The current temperature in the specified location
    """
    return "20 degrees celsius"


async def run_langchain_integration(messages_filepath: str) -> str:
    """
    Run a test with LangChain directly.

    Args:
        query: The query string to test with the LLM wrapper

    Returns:
        The response from the LLM wrapper
    """

    # define config, mode, user_info, tools and messages
    config = {
        "configurable": {
            "thread_id": uuid.uuid4(),
        }
    }
    messages = {"messages": load_messages_from_file(messages_filepath)}
    # define the model and react agent
    chat_llm = ChatOpenAI(
        model=AGENT_MODEL,
        base_url=AGENT_BASE_URL,
        api_key=API_KEY_VALUE,
        temperature=0,
        # response_format={"type": "json_object"}, # causes ERROR knowledgebase_search_tool is not strict. Only strict function tools can be auto-parsed
    )
    agent = create_react_agent(
        model=chat_llm,
        tools=[get_current_temperature_func],  # tools,
        prompt="",  # system_message,
    )

    try:
        result = ""
        async for response2 in agent.astream(messages, config=config):
            logging.info(f"{response2}")
            result = response2
        logging.info(f"Result: {result}")
        return result
    except Exception as e:
        logging.error(f"Error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    # basic logging setup
    logging.basicConfig(level=logging.INFO)
    # Example usage
    messages_filepath = "msgs_inference_tool_fails.txt"

    print("Running LLM wrapper test with messages:")
    print(f"'{messages_filepath}'")
    response = asyncio.run(run_langchain_integration(messages_filepath))
    print(f"'{response}'")
