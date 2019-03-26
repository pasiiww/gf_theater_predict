import torch
import torch.nn as nn
import numpy as np

data = [
        [[32,24,44],[24,29,47],[24,41,35],[27,26,48],[31,29,40]],
        [[31,26,44],[24,29,47],[23,40,36],[27,27,46],[29,36,35]],
        [[34,23,43],[23,28,48],[23,38,39],[36,27,38],[23,42,53]]
        ]

data = np.array(data)
input_data = data[:,:-1]
y = data[:,1:]

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

    def loss(self,logits,target): # 没有用到的cross entry loss with logits
        loss = torch.sum(- target * nn.functional.log_softmax(logits, -1), -1)
        mean_loss = loss.mean()
        return mean_loss


net = testNet()

#optimizer = torch.optim.SGD(net.parameters(),lr = 0.01,momentum=0.9,weight_decay=1E-7)
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
load_name = save_name
if load_name :
    net.load_state_dict(torch.load(load_name))

for epoch in range(0):

    input_datas = torch.Tensor(input_data)
    ys = torch.Tensor(y)
    step_loss = train(input_datas,ys)
    print(step_loss)

if save_name:
    torch.save(net.state_dict(),save_name)


test_data = [[[38,21,41],[26,27,47],[24,36,40],[34,30,36]]] #渠道服
if 1:
    data = np.array(test_data)
    input_data = data[:, :]
    input_datas = torch.Tensor([input_data[0]])
    #print(input_datas)
    reslut = net(input_datas)
    print(reslut)