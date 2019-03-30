from data import *

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
# optimizer = torch.optim.SGD(net.parameters(),lr = 0.0001,momentum=0.9,weight_decay=1E-7)
optimizer = torch.optim.Adam(net.parameters(),lr = 0.005,weight_decay=1E-7)
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
    step_loss = train(input_datas,ys)
    print(step_loss)

if save_name:
    torch.save(net.state_dict(),save_name)


def get_net():
    return net


if __name__ == '__main__':
    #test_data = [[38,21,41],[26,27,47],[24,36,40],[34,30,36],[24,37,39],[32,35,33],[30,26,44],[31.9,36,32.1],[24,32,44]] #渠道服
    #test_data = [[30,25,44],[23,29,48],[25,26,38],[35,31,34],[22,34,44],[26,41,34],[28,31,41],[32,34,33]]
    name = ["IOS ","安官","安B "]
    for i in range(3):
        test_data = data[i]

        data_ = np.array(test_data)
        input_datas = torch.Tensor([data_])
        #print(input_datas)
        reslut = net(input_datas).detach().numpy()
        print(name[i],reslut[0][-1])
        #print(name[i], reslut[0])
        #a = reslut[0][:-1]-test_data[1:]
        #print(np.sum(np.abs(a)))