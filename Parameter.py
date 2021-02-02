import numpy as np

'''注意点：
1. AP的位置在生成的时候需要注意离边界有充足的距离，以便缓解状态统一性问题
2. 我们假定正北为0度，antenna 0可在[-60, 60]调整，antenna 1可在[60, 180]调整， antenna 2可在[180, 240]调整
3. 暂定天线的平均水平覆盖范围90度，平均站高30米，平均竖直覆盖范围60度（这些粗略数值需要根据仿真效果进行调整）
'''

class Parameter(object):
    def __init__(self, M: 10, xSize: 50, ySize: 50):
        super(Parameter, self).__init__()

        self.xSize = xSize                                          # 地图的宽度（以一个矩形区域为单位进行训练）
        self.ySize = ySize                                          # 地图的高度
        self.M = M                                                  # AP数量

        '''天馈参数'''
        self.AP_loc = np.array([[0., 0.] for _ in range(self.M)])   # 基站j在图中的横纵坐标
        self.AP_height = np.array([0. for _ in range(self.M)])      # 基站j的高度

        self.antenna_horizontal_range = np.array([[0. for _ in range(3)] for _ in range(self.M)])
        # 基站j的0～2号天面的水平覆盖范围antenna_horizontal_range[j][0~2]
        self.antenna_vertical_range = np.array([[0. for _ in range(3)] for _ in range(self.M)])
        # 基站j的0～2号天面的竖直覆盖范围antenna_vertical_range[j][0~2]
        self.antenna_angle = np.array([[[0., 0.] for _ in range(3)] for _ in range(self.M)])
        # 基站j的0～2号天面的 方位角antenna_angle[j][0~2][0] 俯仰角antenna_angle[j][0~2][1]

        '''覆盖数据'''
        self.rsrpMap = np.array([[0. for _ in range(self.ySize)] for _ in range(self.xSize)])
        # 地图上某一点的RSRP值rsrpMap[x][y]

