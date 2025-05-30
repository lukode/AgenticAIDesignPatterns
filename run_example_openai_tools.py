from openai import OpenAI
import os

OPENAI_AGENT_API_KEY = "INFERENCE_API_KEY"

AGENT_MODEL = "meta-llama/llama-3.2-3b-instruct/fp-16"  # "meta-llama/llama-3.3-70b-instruct/fp-16"
AGENT_BASE_URL = "https://api.inference.net/v1"
API_KEY_VALUE = os.environ.get(OPENAI_AGENT_API_KEY)

client = OpenAI(
    base_url=AGENT_BASE_URL,
    api_key=API_KEY_VALUE,
)

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current temperature for a given location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City and country e.g. Bogotá, Colombia",
                    }
                },
                "required": ["location"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }
]

completion = client.chat.completions.create(
    model=AGENT_MODEL,
    messages=[{"role": "user", "content": "What is the weather like in Paris today?"}],
    tools=tools,
)

print(completion.choices[0].message.tool_calls)
for tool_call in completion.choices[0].message.tool_calls:
    print(tool_call.function)
