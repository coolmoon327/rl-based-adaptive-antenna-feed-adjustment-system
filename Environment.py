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

        self.ap_ovals = []      # 保存画布上绘制的ap
        self.rssi_rects = []    # 保存画布上绘制的每个格子

        self._init_map()

    def _init_map(self):
        # 初始化参数
        if np.sum(self.param.AP_loc, axis=0)[0] == 0:
            print("Warning: 未初始化AP分布地图，将进行随机生成！")
            self.generate_AP_Map()
        # 初始化覆盖地图
        self.cal_covered_map()
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

        # 绘制每个点的信号强度
        self.draw_RSSI_graph()

        # 绘制基站
        self.draw_AP_Loc()

        # pack all
        self.canvas.pack()

    '''绘制地图上每个方格的RSSI颜色
    [-65, +∞):    蓝色   0,4,247
    [-75, -65):   湛蓝   0,103,232
    [-85, -75):   道奇蓝 74,137,254
    [-95, -85):   天蓝   122,209,244  
    [-105, -95):  黄色   255,248,12
    [-115, -105): 橙色   255,138,22
    [-∞, -115):   红色   255,6,0
    '''
    def draw_RSSI_graph(self):
        # 清空图上绘制的点
        for rect in self.rssi_rects:
            self.canvas.delete(rect)
            self.rssi_rects.remove(rect)
        # 在图上绘制新点
        pass

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
        pass

    def reset(self):
        pass

    def step(self, action, ap: int, antenna: int):
        # 需要指定该环境下执行决策的AP编号与天面编号
        pass

    def render(self):
        # time.sleep(0.01)
        self.update()

    '''生成AP的分布地图'''
    def generate_AP_Map(self):
        xSize = self.param.xSize
        ySize = self.param.ySize
        interval = self.param.interval
        spacing = np.ceil(np.sqrt(min(xSize, ySize)))   # 要与各边留一个间距

        if min(xSize, ySize) <= 2*spacing:
            print("地图尺寸过小，生成AP分布图失败。")
            exit(0)

        # Generate APs
        for j in range(self.param.M):
            while 1:
                xj = np.random.randint(spacing, xSize - spacing)
                yj = np.random.randint(spacing, ySize - spacing)
                isExist = False
                for [x, y] in self.param.AP_loc:
                    if x == xj and y == yj:
                        isExist = True
                        break
                if not isExist:
                    self.param.AP_loc[j] = [xj, yj]
                    break
        print('Success to generate map!\n')

    '''从文件中读取AP与UE的分布地图'''
    def loadMap(self, filename: str):
        npzfile = np.load('data/'+filename)
        self.param.AP_loc = npzfile['arr_0']
        print('Success to load map!')

    '''将AP与UE的分布地图保存到文件'''
    def saveMap(self, filename: str):
        np.savez('data/'+filename, self.param.AP_loc)
        print('Success to save map!\n')

    '''计算覆盖地图
    :return [covered_map, rsrp_map]: 返回每个点被辐射到的基站列表，以及每个点接收信号的RSRP值
    '''
    def cal_covered_map(self):
        param = self.param
        M = param.M
        # 记录每个点有多少基站覆盖，covered_map[x][y]记录一个点上能够接收到信号的所有基站编号
        # 只需要记录基站而不需要天面，因为只用得到基站要与点之间的距离参数
        covered_map = np.array([[[] for _ in range(self.ySize)] for _ in range(self.xSize)])
        # 记录每个点的信号质量，rsrp_map[x][y]记录一个点上当前信道的RSRP值
        rsrp_map = np.array([[0. for _ in range(self.ySize)] for _ in range(self.xSize)])

        # 1. 枚举每个基站的每个天面，找到它们现在辐射到的所有区域，并给对应点的covered_map列表中添加上基站j
        for j in range(M):
            for i in range(3):
                hr = param.antenna_horizontal_range[j][i] % 360
                vr = param.antenna_vertical_range[j][i] % 360
                ha = param.antenna_horizontal_angle[j][i]
                va = param.antenna_vertical_angle[j][i]
                h = param.AP_height[j]
                x_j, y_j = param.AP_loc[j]

                # 1.1 计算打到地面上的最小半径与最大半径
                inner_angle = (va - vr/2.) % 360 / 180 * np.pi
                inner_radius = h * np.tan(inner_angle)
                outer_angle = (va + vr/2.) % 360 / 180 * np.pi
                outer_radius = h * np.tan(outer_angle)

                # 1.2 计算打到地面上的圆弧两边角度（以AP为圆心，正北为0度）
                left_angle = (ha - hr/2.) % 360 / 180 * np.pi
                right_angle = (ha + hr/2.) % 360 / 180 * np.pi

                # 1.3 以AP为圆心，枚举一个以最大直径为边的矩形区域，判断其中的点是否在1.2与1.3计算的范围内（可优化）
                x_min = np.max(0, x_j - outer_radius)
                x_max = np.min(param.xSize - 1, x_j + outer_radius)
                y_min = np.max(0, y_j - outer_radius)
                y_max = np.min(param.ySize - 1, y_j + outer_radius)
                for x in range(x_min, x_max + 1):
                    for y in range(y_min, y_max + 1):
                        # 计算(x, y)与(x_j, y_j)连线的角度（以AP为圆心，正北为0度）
                        angle = np.arctan((x - x_j) / (y - y_j))
                        # 计算(x, y)与(x_j, y_j)连线的距离
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
                    level = 6
                else:
                    level = 1   # 最低1: -65, 最高6: -115
                    for k in range(covered_num):
                        rand = np.random.randint(0, 8)
                        if rand == 0:
                            level += 1 if level < 6 else 0

                # 2.2 高斯采样出RSRP值
                mu = -65. - 10.*level + np.random.randint(0, 10)
                sigma = 3. * covered_num
                rand = np.random.normal(loc=mu, scale=sigma, size=1)
                rsrp_map[x][y] = max(min(rand, -65.), -115.)

        return covered_map, rsrp_map