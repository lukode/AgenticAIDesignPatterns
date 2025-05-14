from model.base_llm import BaseLLM
from multi_agent.utils import (
    CONTEXT_TAG,
    CONTEXT_TAG_END,
)
from reason_and_act.react_agent import ReactAgent
from tool_use.llm_tool import LLMTool


class MemberAgent:
    def __init__(
        self,
        llm: BaseLLM,
        name: str,
        backstory: str,
        task_description: str,
        task_expected_output: str = "",
        tools: list[LLMTool] | None = None,
    ):
        self.react_agent = ReactAgent(llm, tools or [], backstory)
        self.name = name
        self.backstory = backstory
        self.task_description = task_description
        self.task_expected_output = task_expected_output
        # 2-way dependency tracking
        self.dependencies: list[MemberAgent] = []
        self.dependents: list[MemberAgent] = []
        self.dependencies_context = ""
        self.member_agent_prompt = f"""
You are the {self.name}, collaborating with a team in a workflow.
{self.backstory}

Instructions:
1. Carefully analyze the context provided in the {CONTEXT_TAG}{CONTEXT_TAG_END} XML tags (if available)
2. Focus on completing your specific task as described
3. Provide your response in the expected output format
4. Be thorough and precise in your work
5. Remember that your output will be passed to the next agent in the workflow (if any)

Your task is to:
{self.task_description}

Your task should result in the following output:
{self.task_expected_output}

Process the following context:
{CONTEXT_TAG}
%s
{CONTEXT_TAG_END}

"""

    def add_dependency(self, other):
        self.dependencies.append(other)
        other.dependents.append(self)

    def add_dependent(self, other):
        other.dependencies.append(self)
        self.dependents.append(other)

    def add_context(self, new_data):
        self.dependencies_context += f"\n{new_data}\n"

    def generate(self):
        # Generate the result
        result = self.react_agent.generate(
            self.member_agent_prompt % (self.dependencies_context)
        )
        # Add the result to the context of agents depending on this agent
        for d in self.dependents:
            d.add_context(result)
        return result
