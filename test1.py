import torch
import torch.nn as nn
import numpy as np

# IOS 安卓官服 B服数据
data = [
        [[32,24,44],[24,29,47],[24,41,35],[27,26,48],[31,29,40],[29,39,32],[26,32,41]],
        [[31,26,44],[24,29,47],[23,40,36],[27,27,46],[29,36,35],[29,32,39],[38,29,33]],
        [[34,23,43],[23,28,48],[23,38,39],[36,27,38],[23,42,35],[23,33,44],[34,33,33]]
        ]

data = np.array(data)
input_data = data[:,:-1]
y = data[:,1:]
# 输入为前n-1天的，预测对应后一天的概率

# 定义网络
class testNet(nn.Module):
    def __init__(self):
        super(testNet,self).__init__()

        self.fc1 = nn.Linear(3,16)
        self.lstm = nn.LSTM(16,32,1,batch_first=True)
        self.fc2 = nn.Linear(32,3)

    def forward(self, x):

        x = self.fc1(x)
        #print(x.shape)
        #x = x.view(batch_size,-1,16)
        x, _ = self.lstm(x)
        x = self.fc2(x)
        return x

net = testNet()
# optimizer = torch.optim.SGD(net.parameters(),lr = 0.01,momentum=0.9,weight_decay=1E-7)
optimizer = torch.optim.Adam(net.parameters(),lr = 0.01,weight_decay=1E-7)
loss1 = torch.nn.MSELoss()


def train(batch,label):
    optimizer.zero_grad()
    reslut = net(batch)
    #loss = net.loss(reslut,label)
    loss = loss1(reslut,label) # 和gt做MSE
    loss2 = torch.sum(torch.abs(torch.sum(reslut,-1) -100)) #约束概率的和为100.。。。normaliza

    loss = loss + 0.1 * loss2
    loss.backward()
    optimizer.step()
    return loss.detach()

save_name = "temp_model.pkl"
load_name = save_name # 读取模型文件的名字，设为NULL重新训练
if load_name :
    net.load_state_dict(torch.load(load_name))


for epoch in range(0): # 训练的轮数 一般几百次次就收敛了
    input_datas = torch.Tensor(input_data)
    ys = torch.Tensor(y)
    step_loss = train(input_datas,ys)
    print(step_loss)

if save_name:
    torch.save(net.state_dict(),save_name)


#test_data = [[38,21,41],[26,27,47],[24,36,40],[34,30,36]] #渠道服
test_data = data[0] #输出结果 0:IOS 1:安卓官服 2:安卓b服
if 1:
    data = np.array(test_data)
    input_datas = torch.Tensor([data])
    #print(input_datas)
    reslut = net(input_datas)
    print(reslut)