from Parameter import Parameter
from Environment import Environment
import numpy as np

M = 15
interval = 10
xSize = 50
ySize = 50
param = Parameter(M=M, interval=interval, xSize=xSize, ySize=ySize)

potential_coverage = [[[] for _ in range(3)] for _ in range(param.M)]
# potential_coverage[ap][antenna]是天面的潜在覆盖范围内所有的坐标[x, y]集合
for j in range(param.M):
    for i in range(3):
        potential_coverage[j][i] = param.cal_potential_coverage(ap=j, antenna=i)

filename = f'map_{M}_{xSize}_{ySize}_{interval}.npz'
param.loadMap(filename=filename)
env = Environment(param=param, isRender=True)

# maxx = -1e8
# for _ in range(125):
#     param.init_parameters()
#     param.generate_AP_Map()
#     env = Environment(param=param, isRender=False)
#     ttRSRP = np.sum(param.rsrp_map)
#     if maxx < ttRSRP:
#         maxx = ttRSRP
#         param.saveMap(filename=filename)