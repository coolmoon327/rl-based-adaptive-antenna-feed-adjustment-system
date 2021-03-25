from MADDPG.RL_brain import MADDPG


class Algorithm(object):
    def __init__(self, n_agents, dim_act, dim_obs, batch_size, capacity, episodes_before_train):
        self.maddpg = MADDPG(n_agents=n_agents,
                             dim_act=dim_act,
                             dim_obs=dim_obs,
                             batch_size=batch_size,
                             capacity=capacity,
                             episodes_before_train=episodes_before_train)

