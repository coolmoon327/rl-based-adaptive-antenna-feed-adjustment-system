import numpy as np
import time
import copy
import sys
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk

from Parameter import Parameter

UNIT = 10   # pixels


"""虚拟环境
---
:func reset: 重置环境
:func step: 某AP的某天面执行某操作后进入下一个环境状态，返回features（以一维数据的形式输出整张RSRP地图）
:func render: 输出渲染的canvas地图
:func generate_AP_Map loadMap saveMap: 生成、加载、保存AP分布地图
---
:param isRender: 是否自动渲染canvas
:param action_space: 行为空间
:param n_actions: 行为空间尺寸
:param n_features: 输出尺寸
---
注意，canvas的(0, 0)是左上角，而我们计算中默认原点在左下角，调试中需要留意
"""
class Environment(tk.Tk, object):
    def __init__(self, param: Parameter, isRender: False):
        super(Environment, self).__init__()

        self.param = param
        self.isRender = isRender

        self.action_space = [i for i in range(-10, 10, 1)]      # 可逆时针调整10度 到顺时针调整10度
        self.n_actions = len(self.action_space)                 # 总的行为数
        self.n_features = self.param.xSize * self.param.ySize   # 输出的尺寸（以一维数据的形式输出整张地图，因此尺寸为x*y）
        self.title('RSRP Map of 5G Access Network')
        self.geometry(f'{self.param.xSize * UNIT}x{self.param.ySize * UNIT}')
        self.ap_ovals = []      # 保存画布上绘制的ap（圆点）
        self.rssi_rects = []    # 保存画布上绘制的RSRP（方格）

        self._init_map()

    def _init_map(self):
        # 初始化参数
        if np.sum(self.param.AP_loc, axis=0)[0] == 0:
            print("Warning: 未初始化AP分布地图，将进行随机生成！")
            self.param.generate_AP_Map()
        # 初始化覆盖地图
        covered_map, self.param.rsrp_map = self.cal_covered_map()
        # 构建地图
        self._build_map_canvas()

    def _build_map_canvas(self):
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

        if self.isRender:
            # 绘制每个点的信号强度
            self.draw_RSSI_graph()
            # 绘制基站
            self.draw_AP_Loc(ap=-1)

            self.canvas.pack()
            self.mainloop()

    '''绘制地图上每个方格的RSSI颜色
    [-65, +∞):    蓝色   #0005F6
    [-75, -65):   湛蓝   #0064EF
    [-85, -75):   深天蓝  #00B3FB
    [-95, -85):   亮天蓝  #72DEFF 
    [-105, -95):  yellow
    [-115, -105): orange
    [-∞, -115):   red
    '''
    def draw_RSSI_graph(self):
        # 清空图上绘制的点
        for rect in self.rssi_rects:
            self.canvas.delete(rect)
            self.rssi_rects.remove(rect)
        # 在图上绘制新点
        for x in range(self.param.xSize):
            for y in range(self.param.ySize):
                rsrp_level = self.param.rsrp_level(x, y)
                if rsrp_level == 1:
                    color = '#0005F6'
                elif rsrp_level == 2:
                    color = '#0064EF'
                elif rsrp_level == 3:
                    color = '#00B3FB'
                elif rsrp_level == 4:
                    color = '#72DEFF'
                elif rsrp_level == 5:
                    color = 'yellow'
                elif rsrp_level == 6:
                    color = 'orange'
                else:
                    color = 'red'
                rect = self.canvas.create_rectangle(x*UNIT, y*UNIT, (x+1)*UNIT, (y+1)*UNIT, fill=color)
                self.rssi_rects.append(rect)

    '''绘制AP的位置
    在图上用原点标出所有的AP
    :param ap: 传入当前执行决策的AP编号，若不为空，则将对应的点高亮
    '''
    def draw_AP_Loc(self, ap: -1):
        # 清空图上绘制的点
        for oval in self.ap_ovals:
            self.canvas.delete(oval)
            self.ap_ovals.remove(oval)
        # 在图上绘制新点
        for j in range(self.param.M):
            if j == ap:
                color = 'white'
            else:
                color = 'black'
            x, y = self.param.AP_loc[j]
            oval = self.canvas.create_oval(x*UNIT, y*UNIT, (x+1)*UNIT, (y+1)*UNIT, fill=color)
            self.ap_ovals.append(oval)

    def reset(self):
        self.param.init_parameters()
        self._init_map()

    '''进入下一个状态
    :param ap: 执行当前动作的AP
    :param antenna: 执行当前动作的天面(0~2)
    :param azimuth_act: 方位角调整策略（从action_space选择一个角度进行调整）
    :param pitch_act: 俯仰角调整策略（从action_space选择一个角度进行调整）
    '''
    def step(self, ap: int, antenna: int, azimuth_act: int, pitch_act: int):
        # 执行动作
        ha = azimuth_act + self.param.antenna_horizontal_angle[ap][antenna]
        va = pitch_act + self.param.antenna_vertical_angle[ap][antenna]
        hab = self.param.antenna_horizontal_angle_bound[ap][antenna]
        vab = self.param.antenna_veritical_angle_bound[ap][antenna]
        if hab[0] <= ha <= hab[1]:
            self.param.antenna_horizontal_angle[ap][antenna] = ha
        if vab[0] <= va <= vab[1]:
            self.param.antenna_vertical_angle[ap][antenna] = va
        # 计算rsrp地图
        covered_map, rsrp_map = self.cal_covered_map()
        # 渲染
        if self.isRender:
            self.render(ap=ap)

        self.param.rsrp_mapb =copy.deepcopy(rsrp_map)
        return rsrp_map.reshape((1, -1))[0]

    def render(self, ap=-1):
        # time.sleep(0.01)
        self.draw_RSSI_graph()
        self.draw_AP_Loc(ap=ap)
        self.update()

    '''计算覆盖地图
    :return [covered_map, rsrp_map]: 返回每个点被辐射到的基站列表，以及每个点接收信号的RSRP值
    '''
    def cal_covered_map(self):
        param = self.param
        M = param.M
        # 记录每个点有多少基站覆盖，covered_map[x][y]记录一个点上能够接收到信号的所有基站编号
        # 只需要记录基站而不需要天面，因为只用得到基站要与点之间的距离参数
        covered_map = [[[] for _ in range(param.ySize)] for _ in range(param.xSize)]
        # 记录每个点的信号质量，rsrp_map[x][y]记录一个点上当前信道的RSRP值
        rsrp_map = np.array([[0. for _ in range(param.ySize)] for _ in range(param.xSize)])

        # 1. 枚举每个基站的每个天面，找到它们现在辐射到的所有区域，并给对应点的covered_map列表中添加上基站j
        for j in range(M):
            for i in range(3):
                hr = param.antenna_horizontal_range[j][i] % 360
                vr = param.antenna_vertical_range[j][i] % 360
                ha = param.antenna_horizontal_angle[j][i]
                va = param.antenna_vertical_angle[j][i]
                h = param.AP_height[j]
                x_j, y_j = param.AP_loc[j]

                # 1.1 计算打到地面上的最小半径与最大半径，单位interval
                inner_angle = (va - vr/2.) % 360 / 180 * np.pi
                inner_radius = h * np.tan(inner_angle) / param.interval
                outer_angle = (va + vr/2.) % 360 / 180 * np.pi
                outer_radius = min(h * np.tan(outer_angle), 300) / param.interval

                # 1.2 计算打到地面上的圆弧两边角度（以AP为圆心，正北为0度）
                left_angle = (ha - hr/2.) % 360 / 180 * np.pi
                right_angle = (ha + hr/2.) % 360 / 180 * np.pi

                # 1.3 以AP为圆心，枚举一个以最大直径为边的矩形区域，判断其中的点是否在1.2与1.3计算的范围内（可优化）
                x_min = int(np.ceil(max(0, x_j - outer_radius)))
                x_max = int(np.floor(min(param.xSize - 1, x_j + outer_radius)))
                y_min = int(np.ceil(max(0, y_j - outer_radius)))
                y_max = int(np.floor(min(param.ySize - 1, y_j + outer_radius)))
                for x in range(x_min, x_max + 1):
                    for y in range(y_min, y_max + 1):
                        # 计算(x, y)与(x_j, y_j)连线的角度（以AP为圆心，正北为0度）
                        if y == y_j:
                            angle = np.pi/2 if x > x_j else 3*np.pi/2
                        else:
                            angle = np.arctan((x - x_j) / (y - y_j))
                        # 计算(x, y)与(x_j, y_j)连线的距离，单位interval
                        dist = np.sqrt((x-x_j)**2 + (y-y_j)**2)
                        if y >= y_j:
                            angle = angle % (2 * np.pi)
                        else:
                            angle = (angle + np.pi) % (2 * np.pi)
                        if inner_radius <= dist <= outer_radius:
                            # 当left_angle<right_angle时，处理位于[left_angle, right_angle]之间的点
                            if left_angle < right_angle:
                                if left_angle <= angle <= right_angle:
                                    covered_map[x][y].append(j)
                            # 当left_angle>right_angle时，处理位于[left_angle, 2*np.pi] or [0, right_angle]之间的点
                            else:
                                if left_angle <= angle <= 2*np.pi or 0 <= angle <= right_angle:
                                    covered_map[x][y].append(j)

        # 2. 枚举地图上的每个点，生成它们的RSRP值
        for x in range(param.xSize):
            for y in range(param.ySize):
                # 2.1 估算覆盖强度等级，每有一个基站信号就以1/8的概率降低覆盖强度
                covered_num = len(covered_map[x][y])
                if covered_num == 0:
                    level = 7
                else:
                    level = 1   # 最低1: -65, 最高6: -115
                    for k in range(max(0, covered_num-1)):
                        rand = np.random.randint(0, 8)
                        if rand == 0:
                            ladd = np.random.randint(0, max(0, 7-level))
                            level += ladd if level < 6 else 0

                # 2.2 高斯采样出RSRP值
                mu = -65. - 10.*level + np.random.randint(0, 10)
                sigma = 3. * covered_num
                rand = np.random.normal(loc=mu, scale=sigma, size=1)
                rsrp_map[x][y] = max(rand, -125.)

        return covered_map, rsrp_map