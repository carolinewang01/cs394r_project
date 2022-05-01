import random

class AgentPool(object):
    def __init__(self):
        self.policies = []
        self.len = 0

    def __len__(self):
        return self.len

    def add(self, policy):
        """
        Add a policy to the agent pool
        Size of the agent pool increases
        """
        self.policies.append(policy)
        self.len = len(self.policies)

    def replace(self, policy):
        """
        Add a policy to the agent pool
        Delete one old policy from the agent pool
        """
        self.policies.append(policy)
        # TODO add replace mechanism
        self.len = len(self.policies)

    def action(self):
        """
        Return a action picked by a joint policy
        """
        return None

    def sample(self):
        """
        Select and return policy from agent pool
        """
        index = random.randint(0, self.len)
        policy = self.policies[index]
        return policy

    def reset(self):
        """
        Clean up the agent pool
        """
        self.policies = []
        self.len = 0

