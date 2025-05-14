from model.base_llm import BaseLLM
from model.utils import create_message, add_message_to_history, TYPE_DICTIONARY

from tool_use.llm_tool import LLMTool
import json
import re

from tool_use.utils import (
    TOOLS_DEFINITIONS_TAG,
    TOOLS_DEFINITIONS_TAG_END,
    TOOLS_INVOCATIONS_TAG,
    TOOLS_INVOCATIONS_TAG_END,
    TOOLS_RESULTS_TAG,
    TOOLS_RESULTS_TAG_END,
)


class ToolUseAgent:
    def __init__(
        self,
        llm: BaseLLM,
        tools: list[LLMTool],
    ):
        self.llm = llm
        self.tools = tools
        self.tools_dict = {tool.name: tool for tool in tools}
        self.agent_system_prompt = f"""You are a function-calling AI model.
Function signatures are provided within {TOOLS_DEFINITIONS_TAG}{TOOLS_DEFINITIONS_TAG_END} XML tags. Call one or more functions to assist with the user query without making assumptions about argument values.
Pay close attention to the name and type of each parameter. Return each function call as a JSON object within {TOOLS_INVOCATIONS_TAG}{TOOLS_INVOCATIONS_TAG_END} XML tags, formatted as follows:
{TOOLS_INVOCATIONS_TAG}
{{"name": <function-name>, "arguments": <arguments-dict>}}
{TOOLS_INVOCATIONS_TAG_END}

Here are the available functions:
"""
        self.tool_results_prompt = f"Always check if the function has already been called and the results are in the {TOOLS_RESULTS_TAG}{TOOLS_RESULTS_TAG_END} XML tags. If so, you must answer the user without referring to any functions!"

    def _extract_tool_calls(self, text: str):
        # Find content between the tags using regex
        tag_pattern = rf"{TOOLS_INVOCATIONS_TAG}(.*?){TOOLS_INVOCATIONS_TAG_END}"
        matched_contents = re.findall(tag_pattern, text, re.DOTALL)
        # Strip leading and trailing whitespace or return "" if empty
        content = [content.strip() for content in matched_contents]
        return content if content else ""

    def _convert_tool_arguments(self, tool_call: dict, tool_signature: dict) -> dict:
        # Iterate over all arguments in tool call
        for arg_name, arg_value in tool_call["arguments"].items():
            param_type = tool_signature["parameters"][arg_name]
            # Convert the argument value to the correct type if needed
            if not isinstance(arg_value, TYPE_DICTIONARY[param_type]):
                param_value = TYPE_DICTIONARY[param_type](arg_value)
                tool_call["arguments"][arg_name] = param_value

        return tool_call

    def _handle_tool_calls(self, tool_calls: list) -> dict:
        tool_results = {}
        for tc in tool_calls:
            tool_call_dict = json.loads(tc)
            # Get the tool from the dictionary
            tool = self.tools_dict[tool_call_dict["name"]]
            # Convert any arguments to the correct type
            tool_call = self._convert_tool_arguments(
                tool_call_dict, json.loads(tool.description)
            )
            # Invoke the tool using the tool call data
            result = tool.invoke(**tool_call["arguments"])
            # Store the result for the tool call
            tool_results[tc] = result

        return tool_results

    def generate(self, user_msg: str) -> str:
        tool_definitions = "\n".join(
            [
                TOOLS_DEFINITIONS_TAG,
                ",\n\n".join(
                    [
                        tool.description.encode().decode("unicode_escape")
                        for tool in self.tools
                    ]
                ),
                TOOLS_DEFINITIONS_TAG_END,
            ]
        )
        # Assemble the full prompt
        complete_tool_agent_prompt = f"{self.agent_system_prompt}\n{tool_definitions}\n{self.tool_results_prompt}"
        # Initialize the chat history with tool definitions
        tool_chat_history = [
            create_message(complete_tool_agent_prompt, "system"),
            create_message(user_msg, "user"),
        ]
        # Generate a response (with tool invocations)
        tool_call_response = self.llm.generate(tool_chat_history)
        # Find tool calls
        content = self._extract_tool_calls(tool_call_response)
        # Handle tool calls
        if content:
            tool_results = self._handle_tool_calls(content)
            tool_message = create_message(
                f"{TOOLS_RESULTS_TAG}\n{tool_results}\n{TOOLS_RESULTS_TAG_END}",
                "assistant",
            )
            add_message_to_history(tool_chat_history, tool_message, 1, 100)
        # Generate a final response based on the additional information from the tool call
        final_response = self.llm.generate(tool_chat_history)
        return final_response
