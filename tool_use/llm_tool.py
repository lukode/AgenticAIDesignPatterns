import json
from typing import Callable


class LLMTool:
    def __init__(self, name: str, description: str, function: Callable):
        self.name = name
        self.description = description
        self.function = function

    def invoke(self, **kwargs):
        return self.function(**kwargs)


def convert_to_llm_tool(function: Callable):
    # Get the schema of the function
    function_schema = {
        name: typ.__name__
        for name, typ in function.__annotations__.items()
        if name != "return"
    }
    # Create a function signature
    function_signature: dict = {
        "name": function.__name__,
        "description": function.__doc__,
        "parameters": function_schema,
    }
    # return a LLMTool instance
    ret = LLMTool(
        name=function_signature.get("name"),
        description=json.dumps(function_signature),
        function=function,
    )
    return ret
