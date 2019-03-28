import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

# IOS 安卓官服 B服数据
data = [
        [[32,24,44],[24,29,47],[24,41,35],[27,26,48],[31,29,40],[29,39,32],[26,32,41]],
        [[31,26,44],[24,29,47],[23,40,36],[27,27,46],[29,36,35],[29,32,39],[38,29,33]],
        [[34,23,43],[23,28,48],[23,38,39],[36,27,38],[23,42,35],[23,33,44],[34,33,33]]
        ]

data = np.array(data)


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.attn = nn.Linear(6, 1)

    def forward(self, x, key , value):
        #  x:  1 * seq_len1 * 3
        #  key: 1 * seq_len2 * 3

        batch_size = x.size(0)
        seq_len1 = x.size(1)
        seq_len2 = key.size(0)
        x = x.view(-1,seq_len1,1,3).repeat(1,1,seq_len2,1)
        key = key.view(1,1,seq_len2,3).repeat(batch_size,seq_len1,1,1)

        x = torch.cat([x,key],-1)
        x = self.attn(x)
        x = F.softmax(x.squeeze(-1),dim=-1)
        # x: batch x seq1 x seq2 x 3
        # seq2 x 3
        value = value.view(1,seq_len2,3).repeat(batch_size,1,1)

        x = torch.bmm(x,value)
        return x


# 定义网络
class testNet(nn.Module):
    def __init__(self):
        super(testNet,self).__init__()

        self.fc1 = nn.Linear(9,32)
        self.lstm = nn.LSTM(32,32,1,batch_first=True)
        self.fc2 = nn.Linear(32,3)
        self.atte = Attention()

    def forward(self, x , atte_parameter):

        cat_dict = [x]
        for v in atte_parameter:
            value = self.atte(x,v,v)
            cat_dict.append(value)
        x = torch.cat(cat_dict,-1)
        x = self.fc1(x)
        #print(x.shape)
        #x = x.view(batch_size,-1,16)
        x, _ = self.lstm(x)
        x = self.fc2(x)
        return x

net = testNet()
#optimizer = torch.optim.SGD(net.parameters(),lr = 0.001,momentum=0.9,weight_decay=1E-7)
optimizer = torch.optim.Adam(net.parameters(),lr = 0.005,weight_decay=1E-7)
loss1 = torch.nn.MSELoss()


def train(batch,label,atte_key):
    optimizer.zero_grad()
    reslut = net(batch,atte_key)
    #loss = net.loss(reslut,label)
    loss = loss1(reslut,label) # 和gt做MSE
    loss2 = torch.sum(torch.abs(torch.sum(reslut,-1) -100)) #约束概率的和为100.。。。normaliza

    loss = loss + 0.1 * loss2
    loss.backward()
    optimizer.step()
    return loss.detach()

def test(batch,label,atte_key):
    net.eval()
    optimizer.zero_grad()
    reslut = net(batch,atte_key)
    loss = loss1(reslut,label) # 和gt做MSE
    loss2 = torch.sum(torch.abs(torch.sum(reslut,-1) -100)) #约束概率的和为100.。。。normaliza
    loss = loss + 0.1 * loss2
    return loss.detach()

save_name = "temp_model2.pkl"
load_name = "temp_model2.pkl" # 读取模型文件的名字，设为NULL重新训练
if load_name :
    net.load_state_dict(torch.load(load_name))

input_data = data[:,:-1]
y = data[:,1:]
# 输入为前n-1天的，预测对应后一天的概率

for epoch in range(0): # 训练的轮数 一般几百次次就收敛了
    for i in range(3):
        #input_datas = torch.Tensor([input_data[i]])
        #ys = torch.Tensor([y[i]])
        #atte_key = torch.Tensor([input_data[(i+1)%3],input_data[(i+2)%3]])

        input_datas = torch.Tensor(input_data)
        ys = torch.Tensor(y)
        atte_key = torch.Tensor([input_data[(i+1)%3],input_data[(i+2)%3]])

        step_loss = train(input_datas,ys,atte_key)
        print(step_loss)

if save_name:
    torch.save(net.state_dict(),save_name)

input_datas = torch.Tensor(input_data)
ys = torch.Tensor(y)
atte_key = torch.Tensor([input_data[0],input_data[1]])
print("test loss ", test(input_datas,ys,atte_key))

#test_data = [[38,21,41],[26,27,47],[24,36,40],[34,30,36]] #渠道服
#输出结果 0:IOS 1:安卓官服 2:安卓b服

name = ["IOS ","安官","安B "]
for i in range(3):
    test_data = data[i]

    data_ = np.array(test_data)
    input_datas = torch.Tensor([data_])
    #print(input_datas)
    atte_key = torch.Tensor([input_data[(i + 1) % 3], input_data[(i + 2) % 3]])
    reslut = net(input_datas,atte_key).detach().numpy()
    print(name[i],reslut[0][-1])