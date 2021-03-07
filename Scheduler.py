from Parameter import Parameter
from Environment import Environment
import numpy as np

M = 15
interval = 10
xSize = 50
ySize = 50
param = Parameter(M=M, interval=interval, xSize=xSize, ySize=ySize)

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