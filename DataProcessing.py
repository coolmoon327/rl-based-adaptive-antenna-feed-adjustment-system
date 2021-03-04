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
        pass
        # 2. 计算出采样的水平角度列表
        pass
        # 3. 从潜在覆盖区域中，找出需要采样的点
        pass
        # 4. 取出所有采样点的RSRP，并整理成新的矩阵ans
        pass
