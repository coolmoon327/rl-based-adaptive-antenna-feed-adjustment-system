from .model import Critic, Actor
import torch as th
from copy import deepcopy
from .memory import ReplayMemory, Experience
from torch.optim import Adam
# from randomProcess import OrnsteinUhlenbeckProcess
import torch.nn as nn
import numpy as np
from .params import scale_reward


def soft_update(target, source, t):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - t) * target_param.data + t * source_param.data)


def hard_update(target, source):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(source_param.data)


class MADDPG:
    def __init__(self, n_agents, dim_obs, dim_obs_critic, dim_act, batch_size,
                 capacity, episodes_before_train):
        self.actors = [Actor(dim_obs, dim_act) for i in range(n_agents)]
        self.critics = [Critic(n_agents, dim_obs_critic, dim_act) for i in range(n_agents)]
        self.actors_target = deepcopy(self.actors)
        self.critics_target = deepcopy(self.critics)

        self.n_agents = n_agents
        self.n_states = dim_obs
        self.n_states_critic = dim_obs_critic           # 也就是整张地图的xSize*ySize
        self.n_actions = dim_act
        self.memory = ReplayMemory(capacity)
        self.batch_size = batch_size
        # self.use_cuda = th.cuda.is_available()
        self.use_cuda = False
        self.episodes_before_train = episodes_before_train

        self.GAMMA = 0.95
        self.tau = 0.01

        self.var = [1.0 for i in range(n_agents)]
        self.critic_optimizer = [Adam(x.parameters(),
                                      lr=0.001) for x in self.critics]
        self.actor_optimizer = [Adam(x.parameters(),
                                     lr=0.0001) for x in self.actors]

        if self.use_cuda:
            for x in self.actors:
                x.cuda()
            for x in self.critics:
                x.cuda()
            for x in self.actors_target:
                x.cuda()
            for x in self.critics_target:
                x.cuda()

        self.steps_done = 0
        self.episode_done = 0

    def update_policy(self):
        # do not train until exploration is enough
        if self.episode_done <= self.episodes_before_train or len(self.memory) <= self.batch_size:
            return None, None

        ByteTensor = th.cuda.ByteTensor if self.use_cuda else th.ByteTensor
        FloatTensor = th.cuda.FloatTensor if self.use_cuda else th.FloatTensor

        c_loss = []
        a_loss = []
        for agent in range(self.n_agents):
            # 1. 采样出一批记忆
            transitions = self.memory.sample(self.batch_size)
            batch = Experience(*zip(*transitions))
            non_final_mask = ByteTensor(list(map(lambda s: s is not None, batch.next_states))).bool()
            # 从记忆batch中提取出5类数据
            # state_batch: batch_size x n_agents x dim_obs
            state_batch = th.stack(batch.states).type(FloatTensor)
            action_batch = th.stack(batch.actions).type(FloatTensor)
            reward_batch = th.stack(batch.rewards).type(FloatTensor)
            map_batch = th.stack(batch.map_states).type(FloatTensor)        # map后面的数据包含所有天面的角度信息
            # : (batch_size_non_final) x n_agents x dim_obs
            non_final_next_states = th.stack([s for s in batch.next_states if s is not None]).type(FloatTensor)
            non_final_next_map = th.stack([s for s in batch.next_map_states if s is not None]).type(FloatTensor)

            # 2. 用Critic现实网络（新）计算该map下采取whole_action的R_1（使用map作为Critic的状态）
            # for current agent
            whole_state = state_batch.view(self.batch_size, -1)
            whole_action = action_batch.view(self.batch_size, -1)
            whole_map = map_batch.view(self.batch_size, -1)
            self.critic_optimizer[agent].zero_grad()
            current_Q = self.critics[agent](whole_map, whole_action)

            # 3. 用Actor目标网络（旧）计算下一个状态下应该采取的Action（使用Observation作为Actor的状态）
            non_final_next_actions = [self.actors_target[i](non_final_next_states[:, i, :]) for i in range(self.n_agents)]
            non_final_next_actions = th.stack(non_final_next_actions)
            non_final_next_actions = (non_final_next_actions.transpose(0, 1).contiguous())
            # 0和1转置，即两维度互换，agents是第0维

            # 4. 用Critic目标网络（旧）计算下一个map下采取上述action的R_2（使用map作为Critic的状态）
            target_Q = th.zeros(self.batch_size).type(FloatTensor)
            target_Q[non_final_mask] = self.critics_target[agent](
                non_final_next_map.view(-1, self.n_states_critic),
                non_final_next_actions.view(-1, self.n_agents * self.n_actions)
            ).squeeze()
            # scale_reward: to scale reward in Q functions

            # 5. 用基于值的TD思想得到该map下采取whole_action该有的R_true（gamma * R_2 + reward）
            target_Q = (target_Q.unsqueeze(1) * self.GAMMA) + (reward_batch[:, agent].unsqueeze(1) * scale_reward)

            # 6. 反向传递：让Q现实网络的预测值R_1逼近R_true
            loss_Q = nn.MSELoss()(current_Q, target_Q.detach())
            loss_Q.backward()
            self.critic_optimizer[agent].step()

            # 7. 通过最大化Critic网络的输出来更新Actor网络
            self.actor_optimizer[agent].zero_grad()
            state_i = state_batch[:, agent, :]
            # 7.1 用Actor现实网络（新）计算当前状态下的Action_i
            action_i = self.actors[agent](state_i)
            # 7.2 用新计算的Action_i替换记忆batch中行为体i旧的action
            ac = action_batch.clone()
            ac[:, agent, :] = action_i
            whole_action = ac.view(self.batch_size, -1)
            # 7.3 用Critic现实网络（新）计算最新的R作为Actor的优化目标（最大化R）
            actor_loss = -self.critics[agent](whole_map, whole_action)
            actor_loss = actor_loss.mean()
            actor_loss.backward()
            self.actor_optimizer[agent].step()
            c_loss.append(loss_Q)
            a_loss.append(actor_loss)

        if self.steps_done % 100 == 0 and self.steps_done > 0:
            for i in range(self.n_agents):
                soft_update(self.critics_target[i], self.critics[i], self.tau)
                soft_update(self.actors_target[i], self.actors[i], self.tau)

        return c_loss, a_loss

    def select_action(self, state_batch):
        # state_batch: n_agents x state_dim
        actions = th.zeros(
            self.n_agents,
            self.n_actions)
        FloatTensor = th.cuda.FloatTensor if self.use_cuda else th.FloatTensor
        for i in range(self.n_agents):
            sb = state_batch[i, :].detach()
            act = self.actors[i](sb.unsqueeze(0)).squeeze()

            act += th.from_numpy(
                np.random.randn(self.n_actions) * self.var[i]).type(FloatTensor)

            if self.episode_done > self.episodes_before_train and self.var[i] > 0.05:
                self.var[i] *= 0.999995
            act = th.clamp(act, -1.0, 1.0)

            actions[i, :] = act
        self.steps_done += 1

        return actions

    def load_networks(self, mode=0):
        try:
            if mode == 0:
                # actor_data = th.load('./data/actor_net_params_1.pkl')
                # critic_data = th.load('./data/critic_net_params_1.pkl')
                actor_data = th.load('./data/actor_net_params_1.pkl', map_location=th.device('cpu'))
                critic_data = th.load('./data/critic_net_params_1.pkl', map_location=th.device('cpu'))
            else:
                actor_data = th.load('./data/actor_net_params_test.pkl', map_location=th.device('cpu'))
                critic_data = th.load('./data/critic_net_params_test.pkl', map_location=th.device('cpu'))
        except IOError:
            print("Error: 没有找到文件或读取文件失败")
        else:
            for i in range(self.n_agents):
                self.actors[i].load_state_dict(actor_data[i])
                self.critics[i].load_state_dict(critic_data[i])
                hard_update(self.actors_target[i], self.actors[i])
                hard_update(self.critics_target[i], self.critics[i])
            print("网络参数加载成功")

    def soft_load_networks(self):
        # 用于分布式同时训练
        try:
            # actor_data = th.load('./data/actor_net_params_1.pkl')
            # critic_data = th.load('./data/critic_net_params_1.pkl')
            actor_data = th.load('./data/actor_net_params_1.pkl', map_location=th.device('cpu'))
            critic_data = th.load('./data/critic_net_params_1.pkl', map_location=th.device('cpu'))
        except IOError:
            print("Error: 没有找到文件或读取文件失败")
        else:
            for i in range(self.n_agents):
                actor = Actor(self.n_states, self.n_actions)
                critic = Critic(self.n_agents, self.n_states_critic, self.n_actions)
                actor.load_state_dict(actor_data[i])
                critic.load_state_dict(critic_data[i])
                soft_update(self.actors[i], actor, 0.5)
                soft_update(self.critics[i], critic, 0.5)
                soft_update(self.critics_target[i], self.critics[i], self.tau)
                soft_update(self.actors_target[i], self.actors[i], self.tau)
            print("网络参数软加载成功")

    def save_networks(self, mode=0):
        actor_data = []
        critic_data = []
        for net in self.actors:
            state_dict = net.state_dict()
            actor_data.append(state_dict)
        for net in self.critics:
            state_dict = net.state_dict()
            critic_data.append(state_dict)
        if mode == 0:
            th.save(actor_data, './data/actor_net_params_1.pkl', _use_new_zipfile_serialization=False)
            th.save(critic_data, './data/critic_net_params_1.pkl', _use_new_zipfile_serialization=False)
        else:
            th.save(actor_data, './data/actor_net_params_test.pkl', _use_new_zipfile_serialization=False)
            th.save(critic_data, './data/critic_net_params_test.pkl', _use_new_zipfile_serialization=False)
        # print("网络参数保存成功")