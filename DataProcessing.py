import numpy as np
from Parameter import Parameter

'''数据处理工具
预处理虚拟环境生成的RSRP地图，得到归一化后的observation
'''
class DataProcessing(object):
    def __init__(self, param: Parameter, ap: int, antenna: int):
        self.param = param
        self.ap = 0
        self.antenna = 0
        self.pcovers = []

        self.set_agent(ap=ap, antenna=antenna)

    '''设置当前服务的天面'''
    def set_agent(self, ap: int, antenna: int):
        self.ap = ap
        self.antenna = antenna
        self.pcovers = self.param.cal_potential_coverage(ap=ap, antenna=antenna)

    '''归一化潜在覆盖地图
        :return ans: 返回归一化后的潜在覆盖地图
        '''
    def normalize_potential_coverage_observation(self):
        param = self.param
        map = param.rsrp_map

        # 1. 计算出采样的半径列表
        # 暂定采样20个半径，按照一定规则切割俯仰角，划分成四个区，角度分配比例为1：2：3：4，每个区里等间距采样5个角度
        pitch_groups_num = 4          # 组数
        pitch_group_samples_num = 5   # 每组样本数
        pitch_samples_num = pitch_groups_num * pitch_group_samples_num    # 总样本点数

        # 1.1 计算总俯仰角度范围
        ap = self.ap
        antenna = self.antenna
        vr = self.param.antenna_vertical_range[ap][antenna]
        vab = param.antenna_veritical_angle_bound[ap][antenna]
        pitch_total_angle = vab[1] - vab[0] + vr

        # 1.2 划分俯仰角的四个区域（半径从小到大，所以区域大小从大到小）
        denominator = 0.
        for i in range(pitch_groups_num):
            denominator += i + 1.
        pitch_groups_angle = [0. for _ in range(pitch_groups_num)]      # 每组分配到的角度
        for i in range(pitch_groups_num):
            pitch_groups_angle[i] = (pitch_groups_num - i) / denominator * pitch_total_angle

        # 1.3 区域内细分采样角度
        pitch_samples_angle = [0. for _ in range(pitch_samples_num)]    # 每个样本点分配到的角度
        temp_angle = 0.
        for i in range(pitch_groups_num):
            for j in range(pitch_group_samples_num):
                temp_angle += pitch_groups_angle[i] / pitch_group_samples_num
                pitch_samples_angle[i * pitch_group_samples_num + j] = temp_angle

        # 1.4 每个采样角度映射成半径
        h = param.AP_height[ap]
        pitch_samples_radius = [0. for _ in range(pitch_samples_num)]
        for i in range(pitch_samples_num):
            pitch_samples_radius[i] = h * np.tan(pitch_samples_angle[i] / 180 * np.pi)

        # 2. 计算出采样的水平角度列表
        # 同样采样20个角度，等比例划分
        azimuth_sample_num = 20

        # 2.1 计算总方位角范围
        hr = param.antenna_horizontal_range[ap][antenna]
        hab = param.antenna_horizontal_angle_bound[ap][antenna]
        azimuth_total_angle = (360. + hab[1] - hab[0] + hr) % 360.

        # 2.2 划分采样角度
        init_angle = hab[0] - hr/2
        unit_angle = azimuth_total_angle / azimuth_sample_num
        azimuth_samples_angle = [init_angle + i * unit_angle for i in range(azimuth_sample_num)]

        # 3. 从潜在覆盖区域中，找出需要采样的点
        # 以方位角为横轴（顺时针增长），俯仰角为纵轴（半径从小到大增长）
        x_j, y_j = param.AP_loc[ap]
        ans = np.array([[-125. for _ in range(pitch_samples_num)] for _ in range(azimuth_sample_num)])
        for i in range(azimuth_sample_num):
            a = azimuth_samples_angle[i] % 360. / 180 * np.pi
            # 3.1 依次算出每个方位角对应的线性函数
            y = lambda x: x
            dosage = 1
            # 3.1.1 0 <= a < np.pi/2:          y = y_j + (x - x_j) / np.tan(a)
            if 0 <= a < np.pi/2:
                y = lambda x: y_j + (x - x_j) / np.tan(a)
            # 3.1.2 np.pi/2 <= a < np.pi:      y = y_j - (x - x_j) * np.tan(a - np.pi/2)
            if np.pi/2 <= a < np.pi:
                y = lambda x: y_j - (x - x_j) * np.tan(a - np.pi/2)
                dosage = -1
            # 3.1.3 np.pi <= a < np.pi*1.5:    y = y_j + (x - x_j) / np.tan(a - np.pi)
            if np.pi <= a < np.pi*1.5:
                y = lambda x: y_j + (x - x_j) / np.tan(a - np.pi)
            # 3.1.4 np.pi*1.5 <= a < np.pi*2:  y = y_j - (x - x_j) * np.tan(a - np.pi*1.5)
            if np.pi*1.5 <= a < np.pi*2:
                y = lambda x: y_j - (x - x_j) * np.tan(a - np.pi*1.5)
                dosage = -1
            # 3.2 找出线性函数上的每一个整数坐标对(x, y)，依次按照从小到大的采样半径进行采样，采样RSRP保存进矩阵ans
            temp_x = x_j
            dist = lambda x1, y1, x2, y2: np.sqrt((x1-x2)**2 + (y1-y2)**2)
            dist_from_ap = lambda x, y: dist(x, y, x_j, y_j)
            for j in range(pitch_samples_num):
                temp_y = y(temp_x)
                next_x = temp_x + dosage
                next_y = y(next_x)
                # 3.2.1 先让(temp_x, temp_y)~(next_x, next_y)包含采样点
                while dist_from_ap(next_x, next_y) <= pitch_samples_radius[j]:
                    temp_x = next_x
                    temp_y = y(temp_x)
                    next_x = temp_x + dosage
                    next_y = y(next_x)
                # 3.2.2 在(temp_x, temp_y)~(next_x, next_y)中找到采样点
                # 从range(int(np.floor(temp_y)), int(np.ceil(next_y)))中枚举所有整数y，找到dist小于采样radius，且最近的target_y
                target_y = np.floor(temp_y)
                for k in range(int(np.floor(temp_y)), int(np.ceil(next_y))):
                    if dist_from_ap(temp_x, k) > pitch_samples_radius[j]:
                        break
                    target_y = k
                # 取rsrp_map(temp_x, target_y)填充到ans[i][j]中
                ans[i][j] = param.rsrp_map[temp_x][target_y]

            # 4. 处理ans矩阵，用近邻+线性预测的方法补全
            # 以要补的点为中心，找到周围的一圈点，然后做一个三维的线性预测：离目标点距离为x轴、离基站距离为y轴，预测z轴的值
            pass

        return ans

    '''计算潜在覆盖地图的reward
        :return ans: 返回归一化后的潜在覆盖地图
        '''
    def cal_reward(self):
        ans = 0.
        for x, y in self.pcovers:
            # 1. 根据pcovers从map中抽取出一个新的表，该表存放潜在覆盖范围内的RSRP评级（1-7）
            rsrp = self.param.rsrp_level(x=x, y=y)
            # 2. 按照一定规则，将配评级进行量化，并加权平均
            # 1: +4; 2: +3; 3: +2; 4: +1; 5: -2; 6: -4; 7: -8.
            if rsrp <= 4:
                ans += 5 - rsrp
            elif rsrp == 5:
                ans += -2
            elif rsrp == 6:
                ans += -4
            elif rsrp == 7:
                ans += -8
        return ans
