from collections import deque
import logging

from multi_agent.member_agent import MemberAgent


class Group:
    def __init__(self):
        self.members: list[MemberAgent] = []

    def add_agent(self, agent):
        self.members.append(agent)

    def topological_sort(self):
        in_degree = {agent: len(agent.dependencies) for agent in self.members}
        queue = deque([agent for agent in self.members if in_degree[agent] == 0])

        sorted_members = []

        while queue:
            current_agent = queue.popleft()
            sorted_members.append(current_agent)

            for dependent in current_agent.dependents:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        if len(sorted_members) != len(self.members):
            raise ValueError(
                "Circular dependencies detected among agents, preventing a valid topological sort"
            )

        return sorted_members

    def generate_old(self):
        ret = ""
        members_sorted = self.topological_sort()
        for member in members_sorted:
            logging.info(f"Asking member {member}")
            ret = member.generate()
            logging.info(f"Member {member} responded with:\n{ret}")
        return ret

    def generate(self, max_steps: int = 10):
        ret = ""
        counter = 0

        # define a recursive function to generate and call dependents
        def member_generate(current_member: MemberAgent):
            nonlocal counter
            if counter > max_steps:
                return None
            counter += 1
            # generate response and add context to dependents
            latest_ret = current_member.generate()
            # recursively call member dependents
            for dependent in current_member.dependents:
                if dependent in self.members and counter <= max_steps:
                    latest_ret = member_generate(dependent) or latest_ret
            return latest_ret

        # invoke the first member recursively and return the response
        ret = member_generate(self.members[0])
        return ret
