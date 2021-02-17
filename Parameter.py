import numpy as np

'''注意点：
1. AP的位置在生成的时候需要注意离边界有充足的距离，以便缓解状态统一性问题
2. 我们假定正北为0度，antenna 0可在[-60, 60]调整，antenna 1可在[60, 180]调整， antenna 2可在[180, 300]调整
3. 暂定天线的平均水平覆盖范围90度，平均站高30米，平均竖直覆盖范围60度（这些粗略数值需要根据仿真效果进行调整）
4. 所有距离相关的地方以interval作为单位，在计算信道增益等数据的时候需要用interval换算成米的单位
5. 处理角度时，需要归一化到[0, 360]之间，但在存储时用的是以1度为单位，以方便进行强化学习的调整
'''


class Parameter(object):
    def __init__(self, M: 10, interval: 1, xSize: 50, ySize: 50):
        super(Parameter, self).__init__()

        self.interval = interval                                    # 单位长度的大小（换算成米）,如xSize*interval表示横轴大小
        self.xSize = xSize                                          # 地图的宽度（以一个矩形区域为单位进行训练）
        self.ySize = ySize                                          # 地图的高度
        self.M = M                                                  # AP数量

        '''天馈参数'''
        self.AP_loc = np.array([[0., 0.] for _ in range(self.M)])   # 基站j在图中的横纵坐标
        self.AP_height = np.array([0. for _ in range(self.M)])      # 基站j的高度

        self.antenna_horizontal_range = np.array([[0. for _ in range(3)] for _ in range(self.M)])
        # 基站j的0～2号天面的水平覆盖范围antenna_horizontal_range[j][0~2]
        self.antenna_horizontal_angle = np.array([[0. for _ in range(3)] for _ in range(self.M)])
        # 基站j的0～2号天面的水平方位角antenna_horizontal_angle[j][0~2]
        self.antenna_vertical_range = np.array([[0. for _ in range(3)] for _ in range(self.M)])
        # 基站j的0～2号天面的竖直覆盖范围antenna_vertical_range[j][0~2]
        self.antenna_vertical_angle = np.array([[0. for _ in range(3)] for _ in range(self.M)])
        # 基站j的0～2号天面的竖直俯仰角antenna_vertical_angle[j][0~2]

        self.antenna_horizontal_angle_bound = np.array([[[0., 0.] for _ in range(3)] for _ in range(self.M)])
        # antenna_horizontal_angle_bound[j][0~2]是一个二元组[left, right]，代表天线最小和最大可以调整到的角度边界
        self.antenna_veritical_angle_bound = np.array([[[0., 0.] for _ in range(3)] for _ in range(self.M)])

        '''覆盖数据'''
        self.rsrp_map = np.array([[0. for _ in range(self.ySize)] for _ in range(self.xSize)])
        # 地图上某一点的RSRP值rsrpMap[x][y]

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
                self.antenna_veritical_angle_bound[j][i] = [vr/2, 90. - vr/2]
