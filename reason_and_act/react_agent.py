from model.base_llm import BaseLLM
from model.utils import create_message, add_message_to_history, TYPE_DICTIONARY
from tool_use.llm_tool import LLMTool
from tool_use.utils import (
    TOOLS_DEFINITIONS_TAG,
    TOOLS_DEFINITIONS_TAG_END,
    TOOLS_INVOCATIONS_TAG,
    TOOLS_INVOCATIONS_TAG_END,
)
from reason_and_act.utils import (
    RESPONSE_TAG,
    RESPONSE_TAG_END,
    OBSERVATION_TAG,
    OBSERVATION_TAG_END,
    THOUGHT_TAG,
    THOUGHT_TAG_END,
    QUERY_TAG,
    QUERY_TAG_END,
    sanitize_json_string,
)
import json
import re
import ast


class ReactAgent:
    def __init__(
        self,
        llm: BaseLLM,
        tools: list[LLMTool],
        backstory_prompt: str = "",
    ):
        self.llm = llm
        self.tools = tools
        self.tools_dict = {tool.name: tool for tool in tools}
        self.backstory_prompt = backstory_prompt
        self.agent_system_prompt = f"""You are a planning and function-calling AI model.
You can only generate the following steps:
- thought: use the {THOUGHT_TAG}{THOUGHT_TAG_END} XML tags to plan the next steps and make function calls with values you have available, so you can obtain more data to call other functions with
- function calls: use the {TOOLS_INVOCATIONS_TAG}{TOOLS_INVOCATIONS_TAG_END} XML tags to request more information, which will be given to you as observation in {OBSERVATION_TAG}{OBSERVATION_TAG_END} XML tags
- answer: use the {RESPONSE_TAG}{RESPONSE_TAG_END} XML tags to deliver the final answer
You operate in a loop between the thought, function calls and observation steps - advancing one step at a time. When you have enough data to provide the final answer you can do so at any step.

Function signatures are provided within {TOOLS_DEFINITIONS_TAG}{TOOLS_DEFINITIONS_TAG_END} XML tags. Call one or more functions to assist with the user query without making assumptions about argument values.
Pay close attention to the name and type of each parameter. Return each function call as a JSON object within {TOOLS_INVOCATIONS_TAG}{TOOLS_INVOCATIONS_TAG_END} XML tags, formatted as follows:
{TOOLS_INVOCATIONS_TAG}
{{"name": <function-name>, "arguments": <arguments-dict>}}
{TOOLS_INVOCATIONS_TAG_END}

Here are the available functions:
"""
        self.tool_results_prompt = f"""Always check if the function has already been called and the results are in the {OBSERVATION_TAG}{OBSERVATION_TAG_END} XML tags.
If you have enough information to answer the user query, you must do so within {RESPONSE_TAG}{RESPONSE_TAG_END} XML tags, without referring to any functions!"""
        self.one_shot_prompt = f"""
Example user query:
{QUERY_TAG}What's the temperature in London?{QUERY_TAG_END}

You can output a thought:
{THOUGHT_TAG}I need to get the temperature in London{THOUGHT_TAG_END}

You can output function calls with available arguments, each like this:
{TOOLS_INVOCATIONS_TAG}{{"name": "get_current_temperature_func", "arguments": {{"location": "London", "unit": "celsius"}}}}{TOOLS_INVOCATIONS_TAG_END}

You will receive an observation for the function calls:
{OBSERVATION_TAG}{{{{"name": "get_current_temperature_func", "arguments": {{"location": "London", "unit": "celsius"}}}}: {{"temperature": 15, "unit": "celsius"}}}}{OBSERVATION_TAG_END}

You have enough information and must provide the final answer:
{RESPONSE_TAG}The temperature in London is 15 degrees celsius{RESPONSE_TAG_END}

Additional instructions:
Always aim to answer the user query fully, but if the user query cannot be answered with provided tools, respond freely within {RESPONSE_TAG}{RESPONSE_TAG_END} XML tags.
"""

    def _extract_response_content(
        self, text: str, tag: str, tag_end: str, allow_no_tags: bool = False
    ) -> list:
        # Find content between the tags using regex
        tag_pattern = rf"{tag}(.*?){tag_end}"
        matched_contents = re.findall(tag_pattern, text, re.DOTALL)
        if not matched_contents and allow_no_tags:
            return [text]
        # Strip leading and trailing whitespace or return "" if empty
        content = [content.strip() for content in matched_contents]
        return content

    def _convert_tool_arguments(self, tool_call: dict, tool_signature: dict) -> dict:
        # Iterate over all arguments in tool call
        for arg_name, arg_value in tool_call["arguments"].items():
            # if tool_signature["parameters"] does not contain arg_name, raise exception
            if arg_name not in tool_signature["parameters"]:
                raise Exception(
                    f"Function {tool_call['name']} does not have argument {arg_name}. Call it with those arguments: {list(tool_signature['parameters'].keys())}"
                )
            param_type = tool_signature["parameters"][arg_name]
            # Convert the argument value to the correct type if needed
            if not isinstance(arg_value, TYPE_DICTIONARY[param_type]):
                param_value = TYPE_DICTIONARY[param_type](arg_value)
                tool_call["arguments"][arg_name] = param_value

        return tool_call

    def _handle_tool_calls(self, tool_calls: list) -> dict:
        tool_results = {}
        tool_calls_list_of_lists = [
            ast.literal_eval(t) if t.startswith("[") else [t] for t in tool_calls
        ]
        tool_calls_flat = [
            item for sublist in tool_calls_list_of_lists for item in sublist
        ]
        counter = 0
        for tc in tool_calls_flat:
            counter += 1
            sanitised_tc = sanitize_json_string(tc)
            tool_call_dict = json.loads(sanitised_tc)
            try:
                # Get the tool from the dictionary
                if tool_call_dict["name"] not in self.tools_dict:
                    raise Exception(
                        f"Function {tool_call_dict['name']} does not exist. Call another function of check if you have enough data to provide an answer."
                    )
                tool = self.tools_dict[tool_call_dict["name"]]
                # Convert any arguments to the correct type
                tool_call = self._convert_tool_arguments(
                    tool_call_dict, json.loads(tool.description)
                )
                # Invoke the tool using the tool call data
                result = tool.invoke(**tool_call["arguments"])
            except Exception as e:
                # get message from exception
                result = str(e)
            # Store the result for the tool call
            # tool_results[tc] = result
            tool_results[counter] = result

        return tool_results

    def generate(self, user_msg: str, max_steps: int = 10) -> str:
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
        complete_tool_agent_prompt = f"{self.backstory_prompt}\n{self.agent_system_prompt}\n{tool_definitions}\n{self.tool_results_prompt}\n{self.one_shot_prompt}"
        # Initialize the chat history with tool definitions
        react_chat_history = [
            create_message(complete_tool_agent_prompt, "system"),
            create_message(f"{QUERY_TAG}{user_msg}{QUERY_TAG_END}", "user"),
        ]
        counter = 0
        while self.tools and counter < max_steps:
            counter += 1
            # Generate a response
            response = self.llm.generate(react_chat_history)
            # If we got tool calls then handle them
            tool_call_content = self._extract_response_content(
                response, TOOLS_INVOCATIONS_TAG, TOOLS_INVOCATIONS_TAG_END
            )
            if tool_call_content:
                # tool_call_msg = create_message(
                #     f"{TOOLS_INVOCATIONS_TAG}\n{tool_call_content}\n{TOOLS_INVOCATIONS_TAG_END}",
                #     "assistant",
                # )
                # Not adding the tool call itself can save tokens
                # add_message_to_history(react_chat_history, tool_call_msg, 2, 100)
                # Handle the tool calls
                tool_results = self._handle_tool_calls(tool_call_content)
                # Sometimes more humanised responses result in more accurate answers
                tool_results_humanised = "\n".join(tool_results.values())
                tool_message = create_message(
                    f"{OBSERVATION_TAG}\n{tool_results_humanised}\n{OBSERVATION_TAG_END}",
                    "user",
                )
                add_message_to_history(react_chat_history, tool_message, 2, 100)
            # If we got a response then return it
            response_content = self._extract_response_content(
                response, RESPONSE_TAG, RESPONSE_TAG_END
            )
            if response_content:
                return response_content[-1]
            # If we got a thought then add it to chat history
            thought_content = self._extract_response_content(
                response, THOUGHT_TAG, THOUGHT_TAG_END
            )
            if thought_content:
                thought_msg = create_message(
                    f"{THOUGHT_TAG}\n{thought_content}\n{THOUGHT_TAG_END}", "assistant"
                )
                add_message_to_history(react_chat_history, thought_msg, 2, 100)

        # Generate a final response
        add_message_to_history(
            react_chat_history,
            create_message(
                "You now have to provide a final response based on all the information provided without the use of any functions or thoughts.",
                "user",
            ),
            2,
            100,
        )
        final_response = self.llm.generate(react_chat_history)
        final_response_content = self._extract_response_content(
            final_response, RESPONSE_TAG, RESPONSE_TAG_END, True
        )
        return final_response_content[-1]
