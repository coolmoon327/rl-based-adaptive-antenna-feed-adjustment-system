from Parameter import Parameter
from Environment import Environment
from Algorithm import Algorithm
from DataProcessing import DataProcessing
import numpy as np
import torch as th
import time

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
            agent_reward = dp.cal_reward() - 30 * dp.cal_uncovered_num()
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
    # 记录
    # reward_list = np.load('./data/reward_list.npy')
    # target_list = np.load('./data/target_list.npy')
    # uncovered_list = np.load('./data/uncovered_list.npy')
    # reward_list = []
    # target_list = []
    # uncovered_list = []
    #
    # print(reward_list)
    #
    # f = open('./data/episode.txt', 'r')
    # RL.episode_done = int(f.read())
    # f.close()
    # f = open('./data/var.txt', 'r')
    # var = float(f.read())
    # for ii in range(len(RL.var)):
    #     RL.var[ii] = var
    # f.close()

    # env.isRender = True
    # env.render()

    RL.load_networks()
    # RL.load_networks(mode=1)
    logs_time = []
    for episode in range(20):
    # for episode in range(maxEpisode):
        # 0. 初始化观测值
        env.reset()
        obs = get_obs_list()
        obs = floatTensor_from_list(obs)

        # rr = get_reward_list()
        # ta = dp.cal_total_reward()
        # un = dp.cal_total_uncovered_num()
        # print(f"Begin: Average reward {np.mean(rr)} Total RSRP {ta * 900} Uncovered Number {un / 9.}%")

        # var = max(0.05, 1. - episode/10.)
        var = 0.1
        for ii in range(len(RL.var)):
            RL.var[ii] = var

        # rewards = []
        # targets = []
        # uncovereds = []

        # max_uncovered_num = max(5, 100 - episode / 10)
        # if episode<10:
        #     max_uncovered_num = 900 - 50*episode
        # else:
        #     max_uncovered_num = max(5, 200 - episode*10)\
        # max_uncovered_num = max(0, 900 - episode * 10)
        max_uncovered_num = 10

        start = time.time()
        for step in range(10):
        # for step in range(maxStep):
            # env.render()

            last_uncovered_count = dp.cal_total_uncovered_num()
            last_map = param.rsrp_map
            last_antenna_angles = dp.get_total_antenna_angles()

            param.set_point()
            for cc in range(1000):
                # 1. 预测出所有agents的actions
                act = RL.select_action(obs).data.cpu()

                # 2. 执行actions
                for ap in range(param.M):
                    for antenna in range(3):
                        index = ap * 3 + antenna
                        dp.set_agent(ap, antenna)
                        # if np.random.randint(0, 5) == 0: # or dp.cal_uncovered_num() == 0:
                        #     # 五分之一的概率不执行任何操作
                        #     act[index][10] = 1.
                        #     act[index][10+env.n_azimuth_actions] = 1.
                        #     continue

                        agent_act = act[index]
                        azimuth_act = env.azimuth_action_space[choose_max_index(agent_act[0: env.n_azimuth_actions])]
                        pitch_act = env.pitch_action_space[choose_max_index(agent_act[env.n_azimuth_actions:
                                                                            env.n_azimuth_actions + env.n_pitch_actions])]
                        env.step(ap=ap, antenna=antenna, azimuth_act=azimuth_act, pitch_act=pitch_act)

                if dp.cal_total_uncovered_num() <= max(max_uncovered_num, last_uncovered_count + 5):
                    break
                else:
                    param.go_back_to_point()

            # rr = get_reward_list()
            # reward = floatTensor_from_list(rr)
            # obs_ = floatTensor_from_list(get_obs_list())

            # if algId == 0:
            #     RL.memory.push(obs, act, obs_, reward)
            # elif algId == 1:
            #     obs_critic = floatTensor_from_list(np.append(last_map.reshape(-1), last_antenna_angles))
            #
            #     obs_critic_ = floatTensor_from_list(np.append(param.rsrp_map.reshape(-1), dp.get_total_antenna_angles()))
            #     RL.memory.push(obs, obs_critic, act, obs_, obs_critic_, reward)
            #     # map中其实包含所有天面的俯仰角和方位角，和地图信息一起在一个一维空间里
            #
            # RL.update_policy()

            step += 1
            # ta = dp.cal_total_reward()
            # un = dp.cal_total_uncovered_num()

            # if step % 10 == 0:
            #     print(f"Episode {RL.episode_done} Setp {step} Average reward {np.mean(rr)} Total RSRP {ta*900} Uncovered Number {un/9.}%")

            # if step % 10 == 0:
            #     end = time.time()
            #     print(f"{end - start}")
                # print(f"Setp {step} Reward {np.mean(rr)} Target {ta * 900} Uncovered {un / 9.}% Time {end-start}")

            # if step>maxStep/2:
            #     rewards += [np.mean(rr)]
            #     targets += [ta]
            #     uncovereds += [un]

        # RL.episode_done += 1
        # if RL.episode_done:
        #     RL.save_networks(mode=1)

        #  print(f"---{episode}\nAverage reward {np.mean(rewards)} Total RSRP {np.mean(targets) * 900} Uncovered Number {np.mean(uncovereds)/9.}%\n---")
        # print(f"Episode {RL.episode_done} Max_uncovered_num {max_uncovered_num} Average reward {np.mean(rewards)} Total RSRP {np.mean(targets) * 900} Uncovered Number {np.mean(uncovereds)/9.}%")

        # reward_list += [np.mean(rewards)]
        # target_list += [np.mean(targets)]
        # uncovered_list += [np.mean(uncovereds)]
        # # if RL.episode_done % 100 == 0:
        # np.save('./data/reward_list.npy', reward_list)
        # np.save('./data/target_list.npy', target_list)
        # np.save('./data/uncovered_list.npy', uncovered_list)
        # f = open('./data/episode.txt', 'w')
        # f.write(str(RL.episode_done))
        # f.close()
        # f = open('./data/var.txt', 'w')
        # f.write(str(RL.var[0]))
        # f.close()

        end = time.time()
        logs_time.append(end - start)

    print(logs_time)

    print("game over")
    env.destroy()


if __name__ == "__main__":

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
                    batch_size=100,
                    capacity=100000,
                    episodes_before_train=-1)

    env.after(1, run_simulation(alg.maddpg_1, 1))
    env.mainloop()
