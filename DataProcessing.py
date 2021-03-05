import numpy as np
from Parameter import Parameter

'''数据处理工具
预处理虚拟环境生成的RSRP地图，得到归一化后的observation
'''
class DataProcessing(object):
    def __init__(self, param: Parameter):
        self.param = param
        self.ap = 0
        self.antenna = 0

    '''设置当前服务的天面'''
    def chose_target(self, ap: int, antenna: int):
        self.ap = ap
        self.antenna = antenna

    '''归一化潜在覆盖地图
        :param map: RSRP地图
        :return ans: 返回归一化后的潜在覆盖地图
        '''
    def normalize_potential_coverage_observation(self, map: np.ndarray):
        xSize = self.param.xSize
        ySize = self.param.ySize
        if map.shape == 1:
            np.reshape(map, (xSize, ySize))

        # 1. 计算出采样的半径列表
        # 暂定采样20个半径，按照一定规则切割俯仰角，划分成四个区，角度分配比例为1：2：3：4，每个区里等间距采样5个角度
        # 1.1 计算总俯仰角度范围
        # 1.2 划分俯仰角的四个区域
        # 1.3 区域内细分采样角度
        # 1.4 每个采样角度映射成半径
        pass
        # 2. 计算出采样的水平角度列表
        # 同样采样20个角度，等比例划分
        # 2.1 计算总方位角范围
        # 2.2 划分采样角度
        pass
        # 3. 从潜在覆盖区域中，找出需要采样的点
        # 以方位角为横轴（顺时针增长），俯仰角为纵轴（半径从小到大增长）、
        # 3.1 依次算出每个方位角对应的线性函数
        # 3.1.1 0 < = a < np.pi/2:          y = y_j + (x - x_j) / np.tan(a)
        # 3.1.2 np.pi/2 < = a < np.pi:      y = y_j - (x - x_j) * np.tan(a - np.pi/2)
        # 3.1.3 np.pi < = a < np.pi*1.5:    y = y_j + (x - x_j) / np.tan(a - np.pi)
        # 3.1.4 np.pi*1.5 < = a < np.pi*2:  y = y_j - (x - x_j) * np.tan(a - np.pi*1.5)、
        # 3.2 找出线性函数上的每一个整数坐标对(x, y)，依次按照从小到达的采样半径进行采样，采样RSRP保存进矩阵ans
