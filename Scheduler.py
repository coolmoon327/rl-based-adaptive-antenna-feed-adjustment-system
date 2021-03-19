from Parameter import Parameter
from Environment import Environment
from MADDPG import MADDPG
from DataProcessing import DataProcessing
import numpy as np
import torch as th

M = 15
interval = 10
xSize = 50
ySize = 50

maxEpisode = 1000
maxStep = 10

param = Parameter(M=M, interval=interval, xSize=xSize, ySize=ySize)
dp = DataProcessing(param=param)

n_agents = param.M * 3

def run_simulation():
    for episode in range(maxEpisode):
        # 0. 初始化观测值
        env.reset()
        obs = []
        for ap in range(param.M):
            for antenna in range(3):
                dp.set_agent(ap=ap, antenna=antenna)
                agent_obs = dp.normalize_potential_coverage_observation()
                obs.append(agent_obs)
        obs = np.stack(obs)
        # if isinstance(obs, np.ndarray):
        obs = th.from_numpy(obs).float()
        if obs.dim() == 1:
            obs = obs.unsqueeze(dim=0)

        FloatTensor = th.cuda.FloatTensor if RL.use_cuda else th.FloatTensor
        for step in range(maxStep):
            if episode >= 0:
                env.isRender = True
                env.render()

            # 1. 预测出所有agents的actions
            obs = obs.type(FloatTensor)
            act = RL.select_action(obs).data.cpu()

            # 2. 执行actions
            obs_ = []
            reward = []
            for ap in range(param.M):
                for antenna in range(3):
                    index = ap * 3 + antenna
                    agent_act = act[index]
                    azimuth_index = th.argmax(agent_act[0: env.n_azimuth_actions]).numpy()
                    azimuth_act = env.azimuth_action_space[azimuth_index]
                    pitch_index = th.argmax(agent_act[env.n_azimuth_actions:
                                                    env.n_azimuth_actions + env.n_pitch_actions]).numpy() - env.n_azimuth_actions
                    pitch_act = env.pitch_action_space[pitch_index]

                    env.step(ap=ap, antenna=antenna, azimuth_act=azimuth_act, pitch_act=pitch_act)

                    dp.set_agent(ap=ap, antenna=antenna)
                    agent_obs = dp.normalize_potential_coverage_observation()
                    obs_.append(agent_obs)
                    agent_reward = dp.cal_reward()
                    reward.append(agent_reward)

            reward = np.stack(reward)
            reward = th.from_numpy(reward).float()
            # reward = th.FloatTensor(reward).type(FloatTensor)
            if reward.dim() == 1:
                reward = reward.unsqueeze(dim=0)

            obs_ = np.stack(obs_)
            obs_ = th.from_numpy(obs_).float()
            if obs_.dim() == 1:
                obs_ = obs_.unsqueeze(dim=0)

            RL.memory.push(obs, act, obs_, reward)
            RL.update_policy()

            step += 1
            print(f"Episode {RL.episode_done} Setp {step} Total RSRP {dp.cal_total_reward()}")

        RL.save_networks()
        RL.episode_done += 1

    print("game over")
    env.destroy()


if __name__ == "__main__":
    # 生成地图
    # maxx = -1e8
    # for _ in range(125):
    #     param.init_parameters()
    #     param.generate_AP_Map()
    #     env = Environment(param=param, isRender=False)
    #     ttRSRP = np.sum(param.rsrp_map)
    #     if maxx < ttRSRP:
    #         maxx = ttRSRP
    #         param.saveMap(filename=filename)

    # 进行仿真
    filename = f'map_{M}_{xSize}_{ySize}_{interval}.npz'
    param.loadMap(filename=filename)
    env = Environment(param=param, isRender=True)

    # 行为空间是二者之和：前一部分中的最大值为azimuth的行为，后一部分的最大值为pitch的行为
    # 观测值空间从数据预处理过程中取得
    RL = MADDPG(n_agents=n_agents,
                dim_act=env.n_azimuth_actions + env.n_pitch_actions,
                dim_obs=dp.n_features,
                batch_size=100,
                capacity=1000,
                episodes_before_train=1)

    RL.load_networks()
    env.after(1, run_simulation)
    env.mainloop()
