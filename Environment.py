import numpy as np
import time
import sys
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk

ENCODE_NUM = 10     # 自编码神经网络的编码大小（建议实现完自编码后，此处从自编码神经网络中获取结果

UNIT = 4   # pixels
MAZE_H = 50  # grid height
MAZE_W = 50  # grid width


class Environment(tk.Tk, object):
    def __init__(self):
        super(Environment, self).__init__()
        self.action_space = [i for i in range(-10, 10, 1)]    # 可逆时针调整10度 到顺时针调整10度
        self.n_actions = len(self.action_space)
        self.n_features = ENCODE_NUM
        self.title('RSRP Map')
        self.geometry('{0}x{1}'.format(MAZE_H * UNIT, MAZE_H * UNIT))
        self.initMap()

    def initMap(self):
        self._build_map()

    def _build_map(self):
        self.canvas = tk.Canvas(self, bg='white',
                                height=MAZE_H * UNIT,
                                width=MAZE_W * UNIT)
        # create grids
        for c in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_W * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        # create origin
        origin = np.array([UNIT/2, UNIT/2])

        # 设置点的过程详见DQN学习例程的maze_env

        # pack all
        self.canvas.pack()

    def reset(self):
        pass

    def step(self, action):
        pass

    def render(self):
        # time.sleep(0.01)
        self.update()
