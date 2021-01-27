import numpy as np
import torch

np.random.seed(1)


class Net(torch.nn.Module):
    def __init__(self, n_features, n_hidden, n_actions):
        super().__init__()
        self.fc1 = torch.nn.Linear(n_features, n_hidden)
        self.fc1.weight.data.normal_(0, 0.1)  # initialization
        self.out = torch.nn.Linear(n_hidden, n_actions)
        self.out.weight.data.normal_(0, 0.1)  # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        actions_value = self.out(x)
        return actions_value


# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=None,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        self.learn_step_counter = 0
        self.cost_his = []
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))
        self.memory_counter = 0

        self._build_net()
        self.loss_func = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.evaluate_net.parameters(), lr=learning_rate)

    def _build_net(self):
        n_hidden = 256
        # ------------------ build evaluate_net ------------------
        self.evaluate_net = Net(self.n_features, n_hidden, self.n_actions)
        # ------------------ build target_net ------------------
        self.target_net = Net(self.n_features, n_hidden, self.n_actions)

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = torch.FloatTensor(observation[np.newaxis, :])

        if np.random.uniform() < self.epsilon:
            actions_value = self.evaluate_net(observation)
            action = torch.squeeze(torch.max(actions_value, 1)[1]).numpy()
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.target_net.load_state_dict(self.evaluate_net.state_dict())
            # print(f'\ntarget_params_replaced: {self.learn_step_counter}\n')

        # sample batch memory from all memory
        mem_size = min(self.memory_size, self.memory_counter)
        sample_index = np.random.choice(mem_size, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        s = torch.FloatTensor(batch_memory[:, :self.n_features])
        a = torch.LongTensor(batch_memory[:, self.n_features: self.n_features + 1])
        r = torch.FloatTensor(batch_memory[:, self.n_features + 1: self.n_features + 2])
        s_ = torch.FloatTensor(batch_memory[:, -self.n_features:])

        q_next = self.target_net(s_).detach()
        q_eval = self.evaluate_net(s).gather(1, a)
        q_target = r + self.gamma * torch.max(q_next, 1)[0].view(self.batch_size, 1)

        # train eval network
        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

        self.cost_his.append(loss.data.detach().numpy())

    def save(self, path="state_dict.pkl"):
        torch.save(self.evaluate_net.state_dict(), path)
        print("Saved state dict.")

    def load(self, path="state_dict.pkl"):
        import os
        if os.path.exists(path):
            state_dict = torch.load(path)
            if state_dict is not None:
                self.evaluate_net.load_state_dict(state_dict)
                self.target_net.load_state_dict(state_dict)
                print("Loaded state dict.")

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()
