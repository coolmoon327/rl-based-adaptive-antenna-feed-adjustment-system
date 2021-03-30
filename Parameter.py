import numpy as np
import copy

'''注意点：
1. AP的位置在生成的时候需要注意离边界有充足的距离，以便缓解状态统一性问题
2. 我们假定正北为0度，antenna 0可在[-60, 60]调整，antenna 1可在[60, 180]调整， antenna 2可在[180, 300]调整
3. 暂定天线的平均水平覆盖范围90度，平均站高30米，平均竖直覆盖范围60度（这些粗略数值需要根据仿真效果进行调整）
4. 所有距离相关的地方以interval作为单位，在计算信道增益等数据的时候需要用interval换算成米的单位
5. 处理角度时，需要归一化到[0, 360]之间，但在存储时用的是以1度为单位，以方便进行强化学习的调整
6. 5G基站的覆盖范围一般在100m-300m内，这里我们取一个覆盖上限300/interval
'''


class Parameter(object):
    def __init__(self, M: 10, interval: 1, xSize: 50, ySize: 50):
        super(Parameter, self).__init__()

        self.interval = interval                                    # 单位长度的大小（换算成米）,如xSize*interval表示横轴大小
        self.xSize = xSize                                          # 地图的宽度（以一个矩形区域为单位进行训练）
        self.ySize = ySize                                          # 地图的高度
        self.M = M                                                  # AP数量
        self.maxRange = 300                                         # 覆盖距离上限

        '''天馈参数'''
        self.AP_loc = np.array([[0, 0] for _ in range(self.M)])   # 基站j在图中的横纵坐标
        self.AP_height = np.array([0. for _ in range(self.M)])      # 基站j的高度

        self.antenna_horizontal_range = np.array([[0. for _ in range(3)] for _ in range(self.M)])
        # 基站j的0～2号天面的水平覆盖范围antenna_horizontal_range[ap][antenna]
        self.antenna_horizontal_angle = np.array([[0. for _ in range(3)] for _ in range(self.M)])
        # 基站j的0～2号天面的水平方位角antenna_horizontal_angle[ap][antenna]
        self.antenna_vertical_range = np.array([[0. for _ in range(3)] for _ in range(self.M)])
        # 基站j的0～2号天面的竖直覆盖范围antenna_vertical_range[ap][antenna]
        self.antenna_vertical_angle = np.array([[0. for _ in range(3)] for _ in range(self.M)])
        # 基站j的0～2号天面的竖直俯仰角antenna_vertical_angle[ap][antenna]

        self.antenna_horizontal_angle_bound = np.array([[[0., 0.] for _ in range(3)] for _ in range(self.M)])
        # antenna_horizontal_angle_bound[ap][antenna]是一个二元组[left, right]，代表天线最小和最大可以调整到的角度边界
        self.antenna_veritical_angle_bound = np.array([[[0., 0.] for _ in range(3)] for _ in range(self.M)])

        '''覆盖数据'''
        self.rsrp_map = np.array([[0. for _ in range(self.ySize)] for _ in range(self.xSize)])
        # 地图上某一点的RSRP值rsrpMap[x][y]
        self.covered_map = [[[] for _ in range(self.ySize)] for _ in range(self.xSize)]

        self.point = []     # 保存点，用于保存某个时刻的天面状态

        self.init_parameters()

    def init_parameters(self):
        for j in range(self.M):
            self.AP_height[j] = 30.
            for i in range(3):
                vr = 60.
                ha = (i - 1) * 120.
                self.antenna_horizontal_range[j][i] = 90.
                self.antenna_vertical_range[j][i] = vr
                self.antenna_horizontal_angle[j][i] = ha
                self.antenna_vertical_angle[j][i] = 45.
                self.antenna_horizontal_angle_bound[j][i] = [ha - 60., ha + 60.]
                max_vab = min(np.arctan(self.maxRange/self.AP_height[j])*180/np.pi, 90.) - vr/2
                self.antenna_veritical_angle_bound[j][i] = [vr/2, max_vab]

    '''获取RSRP评级
    :param x,y: 坐标
    [-65, +∞):    level 1
    [-75, -65):   level 2
    [-85, -75):   level 3
    [-95, -85):   level 4
    [-105, -95):  level 5
    [-115, -105): level 6
    [-∞, -115):   level 7
    '''
    def rsrp_level(self, x: int, y: int):
        rssi = self.rsrp_map[x][y]
        rssi += 65
        level = 1
        while rssi < 0:
            rssi += 10
            level += 1
        return level

    '''获取指定天面的潜在覆盖范围
    :param ap: 指定的AP
    :param antenna: 指定的天面(0~2)
    :return pcovers: 返回一个list，里面包含该天面覆盖的所有点[x, y]
    '''
    def cal_potential_coverage(self, ap: int, antenna: int):
        pcovers = []
        vab = self.antenna_veritical_angle_bound[ap][antenna]
        vr = self.antenna_vertical_range[ap][antenna]
        hab = self.antenna_horizontal_angle_bound[ap][antenna]
        hr = self.antenna_horizontal_range[ap][antenna]
        h = self.AP_height[ap]
        x_j, y_j = self.AP_loc[ap]

        # 1.1 计算打到地面上的最小半径与最大半径，单位interval
        inner_angle = 0.
        inner_radius = 0.
        outer_angle = (vab[1] + vr / 2.) % 360 / 180 * np.pi
        outer_radius = min(h * np.tan(outer_angle), self.maxRange) / self.interval

        # 1.2 计算打到地面上的圆弧两边角度（以AP为圆心，正北为0度）
        left_angle = (hab[0] - hr / 2.) % 360 / 180 * np.pi
        right_angle = (hab[1] + hr / 2.) % 360 / 180 * np.pi

        # 1.3 以AP为圆心，枚举一个以最大直径为边的矩形区域，判断其中的点是否在1.2与1.3计算的范围内（可优化）
        x_min = int(np.ceil(max(0, x_j - outer_radius)))
        x_max = int(np.floor(min(self.xSize - 1, x_j + outer_radius)))
        y_min = int(np.ceil(max(0, y_j - outer_radius)))
        y_max = int(np.floor(min(self.ySize - 1, y_j + outer_radius)))
        for x in range(x_min, x_max + 1):
            for y in range(y_min, y_max + 1):
                # 计算(x, y)与(x_j, y_j)连线的角度（以AP为圆心，正北为0度）
                if y == y_j:
                    angle = np.pi / 2 if x > x_j else 3 * np.pi / 2
                else:
                    angle = np.arctan((x - x_j) / (y - y_j))
                # 计算(x, y)与(x_j, y_j)连线的距离，单位interval
                dist = np.sqrt((x - x_j) ** 2 + (y - y_j) ** 2)
                if y >= y_j:
                    angle = angle % (2 * np.pi)
                else:
                    angle = (angle + np.pi) % (2 * np.pi)
                if inner_radius <= dist <= outer_radius:
                    # 当left_angle<right_angle时，处理位于[left_angle, right_angle]之间的点
                    if left_angle < right_angle:
                        if left_angle <= angle <= right_angle:
                            pcovers.append([x, y])
                    # 当left_angle>right_angle时，处理位于[left_angle, 2*np.pi] or [0, right_angle]之间的点
                    else:
                        if left_angle <= angle <= 2 * np.pi or 0 <= angle <= right_angle:
                            pcovers.append([x, y])
        return pcovers

    '''生成AP的分布地图'''
    def generate_AP_Map(self):
        xSize = self.xSize
        ySize = self.ySize
        interval = self.interval
        spacing = np.ceil(np.sqrt(min(xSize, ySize)))   # 要与各边留一个间距

        if min(xSize, ySize) <= 2*spacing:
            print("地图尺寸过小，生成AP分布图失败。")
            exit(0)

        # Generate APs
        for j in range(self.M):
            while 1:
                xj = np.random.randint(spacing, xSize - spacing)
                yj = np.random.randint(spacing, ySize - spacing)
                isExist = False
                for [x, y] in self.AP_loc:
                    if x == xj and y == yj:
                        isExist = True
                        break
                if not isExist:
                    self.AP_loc[j] = [xj, yj]
                    break
        print('Success to generate map!\n')

    '''从文件中读取AP与UE的分布地图'''
    def loadMap(self, filename: str):
        try:
            npzfile = np.load('data/'+filename)
        except IOError:
            print("Error: 没有找到文件或读取文件失败")
        else:
            self.AP_loc = npzfile['arr_0']
            print('Success to load map!')

    '''将AP与UE的分布地图保存到文件'''
    def saveMap(self, filename: str):
        np.savez('data/'+filename, self.AP_loc)
        print('Success to save map!\n')

    '''设置一个保存点'''
    def set_point(self):
        self.point = [copy.deepcopy(self.antenna_horizontal_angle), copy.deepcopy(self.antenna_vertical_angle)]

    '''从保存点中加载'''
    def go_back_to_point(self):
        if len(self.point) == 0:
            print("there is no point")
        else:
            self.antenna_horizontal_angle  = self.point[0]
            self.antenna_vertical_angle = self.point[1]
