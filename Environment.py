import numpy as np
import time
import sys
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk

from Parameter import Parameter

UNIT = 4   # pixels


class Environment(tk.Tk, object):
    def __init__(self, param: Parameter):
        super(Environment, self).__init__()

        self.param = param
        self.action_space = [i for i in range(-10, 10, 1)]      # 可逆时针调整10度 到顺时针调整10度
        self.n_actions = len(self.action_space)                 # 总的行为数
        self.n_features = self.param.xSize * self.param.ySize   # 输出的尺寸（以一维数据的形式输出整张地图，因此尺寸为x*y）
        self.title('RSRP Map of 5G Access Network')
        self.geometry(f'{self.param.xSize * UNIT}x{self.param.ySize * UNIT}')
        self._init_map()

    def _init_map(self):
        # 初始化参数
        pass
        # 构建地图
        self._build_map()

    def _build_map(self):
        MAZE_H = self.param.ySize
        MAZE_W = self.param.xSize
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

        pass

        # pack all
        self.canvas.pack()

    def reset(self):
        pass

    def step(self, action, ap: int, antenna: int):
        # 需要指定该环境下执行决策的AP编号与天面编号
        pass

    def render(self):
        # time.sleep(0.01)
        self.update()
