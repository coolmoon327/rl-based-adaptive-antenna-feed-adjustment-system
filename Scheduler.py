from Parameter import Parameter
from Environment import Environment
from Algorithm import Algorithm
from DataProcessing import DataProcessing
import numpy as np
import torch as th

M = 5
interval = 10
xSize = 30
ySize = 30

maxEpisode = 1000
maxStep = 100

param = Parameter(M=M, interval=interval, xSize=xSize, ySize=ySize)
dp = DataProcessing(param=param)

n_agents = param.M * 3

FloatTensor = th.cuda.FloatTensor if th.cuda.is_available() else th.FloatTensor


def get_obs_list():
    obs_ = []
    for ap in range(param.M):
        for antenna in range(3):
            dp.set_agent(ap=ap, antenna=antenna)
            obs = dp.normalize_potential_coverage_observation()
            obs_.append(obs)
    return obs_


def get_reward_list():
    reward = []
    for ap in range(param.M):
        for antenna in range(3):
            dp.set_agent(ap=ap, antenna=antenna)
            agent_reward = dp.cal_reward() - 10 * dp.cal_uncovered_num()
            reward.append(agent_reward)
    return reward


def floatTensor_from_list(x):
    x = np.stack(x)
    x = th.FloatTensor(x).type(FloatTensor)
    return x


def choose_max_index(acts):
    acts = acts.numpy()
    # 随机选择value最高的一个元素，返回其index
    index_list = []
    maxx = -1.e6
    for i in range(len(acts)):
        act = acts[i]
        if maxx == act:
            index_list += [i]
        elif maxx < act:
            maxx = act
            index_list = [i]
    i = np.random.randint(0, len(index_list))
    return index_list[i]

def run_simulation(RL, algId):
    RL.load_networks()
    for episode in range(maxEpisode):
        # 0. 初始化观测值
        env.reset()
        obs = get_obs_list()
        obs = floatTensor_from_list(obs)

        for step in range(maxStep):
            if episode >= 0:
                env.isRender = True
                env.render()

            last_uncovered_count = dp.cal_total_uncovered_num()
            last_map = param.rsrp_map
            last_antenna_angles = dp.get_total_antenna_angles()

            # 1. 预测出所有agents的actions
            act = RL.select_action(obs).data.cpu()

            # 2. 执行actions
            param.set_point()
            for cc in range(100):
                for ap in range(param.M):
                    for antenna in range(3):
                        index = ap * 3 + antenna
                        if np.random.randint(0, 5) == 0:
                            # 五分之一的概率不执行任何操作
                            act[index][10] = 1.
                            act[index][10+env.n_azimuth_actions] = 1.
                            continue

                        agent_act = act[index]
                        azimuth_act = env.azimuth_action_space[choose_max_index(agent_act[0: env.n_azimuth_actions])]
                        pitch_act = env.pitch_action_space[choose_max_index(agent_act[env.n_azimuth_actions:
                                                                            env.n_azimuth_actions + env.n_pitch_actions])]
                        env.step(ap=ap, antenna=antenna, azimuth_act=azimuth_act, pitch_act=pitch_act)

                max_uncovered_num = max(5, 150-episode)
                if dp.cal_total_uncovered_num() <= max(max_uncovered_num, last_uncovered_count):
                    break
                else:
                    param.go_back_to_point()

            reward = floatTensor_from_list(get_reward_list())
            obs_ = floatTensor_from_list(get_obs_list())

            if algId == 0:
                RL.memory.push(obs.cpu(), act, obs_.cpu(), reward.cpu())
            elif algId == 1:
                obs_critic = floatTensor_from_list(np.append(last_map.reshape(-1), last_antenna_angles))

                obs_critic_ = floatTensor_from_list(np.append(param.rsrp_map.reshape(-1), dp.get_total_antenna_angles()))
                RL.memory.push(obs.cpu(), obs_critic, act, obs_.cpu(), obs_critic_, reward.cpu())
                # map中其实包含所有天面的俯仰角和方位角，和地图信息一起在一个一维空间里

            RL.update_policy()

            step += 1
            print(f"Episode {RL.episode_done} Setp {step} Total RSRP {dp.cal_total_reward()} Uncovered Number {dp.cal_total_uncovered_num()}")

        RL.episode_done += 1
        if RL.episode_done % 3 == 2:
            RL.save_networks()

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
    #         filename = f'map_{M}_{xSize}_{ySize}_{interval}.npz'
    #         param.saveMap(filename=filename)

    # 进行仿真
    filename = f'map_{M}_{xSize}_{ySize}_{interval}.npz'
    param.loadMap(filename=filename)
    env = Environment(param=param, isRender=False)

    # 行为空间是二者之和：前一部分中的最大值为azimuth的行为，后一部分的最大值为pitch的行为
    # 观测值空间从数据预处理过程中取得
    alg = Algorithm(n_agents=n_agents,
                    dim_act=env.n_azimuth_actions + env.n_pitch_actions,
                    dim_obs=dp.n_features,
                    dim_obs_critic=param.xSize*param.ySize+param.M*3*2,
                    batch_size=1000,
                    capacity=100000,
                    episodes_before_train=-1)

    # env.after(1, run_simulation(alg.maddpg_0, 0))
    env.after(1, run_simulation(alg.maddpg_1, 1))
    env.mainloop()
