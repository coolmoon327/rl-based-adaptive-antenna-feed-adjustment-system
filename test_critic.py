from Parameter import Parameter
from Environment import Environment
from Algorithm import Algorithm
from DataProcessing import DataProcessing
import numpy as np
import torch as th
import time
import sys
import os

M = 5
interval = 10
xSize = 30
ySize = 30

maxEpisode = 1000
maxStep = 20

param = Parameter(M=M, interval=interval, xSize=xSize, ySize=ySize)
dp = DataProcessing(param=param)

n_agents = param.M * 3

# FloatTensor = th.cuda.FloatTensor if th.cuda.is_available() else th.FloatTensor
FloatTensor = th.FloatTensor

if __name__ == "__main__":

    # 进行仿真
    # filename = f'map_{M}_{xSize}_{ySize}_{interval}.npz'
    # param.loadMap(filename=filename)
    env = Environment(param=param, isRender=False)

    for ap_num in range(1, 30):
        alg = Algorithm(n_agents=ap_num*3,
                        dim_act=env.n_azimuth_actions + env.n_pitch_actions,
                        dim_obs=dp.n_features,
                        dim_obs_critic=param.xSize*param.ySize+param.M*3*2,
                        batch_size=100,
                        capacity=100000,
                        episodes_before_train=-1)
        # print(sys.getsizeof(alg.maddpg_0.critics)+sys.getsizeof(alg.maddpg_0.actors), sys.getsizeof(alg.maddpg_1.critics)+sys.getsizeof(alg.maddpg_1.actors))
        # actor_data = []
        # critic_data = []
        # for net in alg.maddpg_0.actors:
        #     state_dict = net.state_dict()
        #     actor_data.append(state_dict)
        # for net in alg.maddpg_0.critics:
        #     state_dict = net.state_dict()
        #     critic_data.append(state_dict)
        # size_0 = actor_data.__sizeof__()+critic_data.__sizeof__()
        # actor_data = []
        # critic_data = []
        # for net in alg.maddpg_1.actors:
        #     state_dict = net.state_dict()
        #     actor_data.append(state_dict)
        # for net in alg.maddpg_1.critics:
        #     state_dict = net.state_dict()
        #     critic_data.append(state_dict)
        # size_1 = actor_data.__sizeof__() + critic_data.__sizeof__()
        # print(size_0, size_1)
        alg.maddpg_0.save_networks()
        filePath = './data/actor_net_params.pkl'
        fsize_0 = os.path.getsize(filePath)
        filePath = './data/critic_net_params.pkl'
        fsize_0 += os.path.getsize(filePath)

        alg.maddpg_1.save_networks(mode=1)
        filePath = './data/actor_net_params_test.pkl'
        fsize_1 = os.path.getsize(filePath)
        filePath = './data/critic_net_params_test.pkl'
        fsize_1 += os.path.getsize(filePath)

        print(round(fsize_0, 2), round(fsize_1, 2))
        del alg
