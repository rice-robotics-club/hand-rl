''' 
Clearly defined reward functions live here independent of weights
'''

class BaseReward:
    def __init__(self):
        pass

    def compute_reward(self, state, action):
        raise NotImplementedError

    def __call__(self, state, action):
        return self.compute_reward(state, action)

class 