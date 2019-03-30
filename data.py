import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

data = [
        [[32,24,44],[24,29,47],[24,41,35],[27,26,48],[31,29,40],[29,39,32],[26,32,41],[32,37,30],[26,35,39]],
        [[31,26,44],[24,29,47],[23,40,36],[27.1,26.9,46],[29,36,35],[29,32,39],[38,29,33],[28,29,43],[23,44,33]],
        [[34,23,43],[23,28,48],[23,38,39],[36,27,38],[23,42,35],[23,33,44],[34,33.1,32.9],[22,36,42],[26,39,35]]
        ]
data = np.array(data)
input_data = data[:, :-1]
y = data[:, 1:]
# 输入为前n-1天的，预测对应后一天的概率
atte_key = torch.Tensor([[input_data[1], input_data[2], input_data[0]], [input_data[2], input_data[0], input_data[1]]])
input_datas = torch.Tensor(input_data)
ys = torch.Tensor(y)