from data import *

class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.attn = nn.Linear(6, 1)

    def forward(self, x, key , value):
        #  x:  batch_size * seq_len1 * 3
        #  key: batch_size * seq_len2 * 3

        seq_len1 = x.size(1)
        seq_len2 = key.size(1)

        x = x.view(-1,seq_len1,1,3).repeat(1,1,seq_len2,1)
        key = key.view(-1,1,seq_len2,3).repeat(1,seq_len1,1,1)
        x = torch.cat([x,key],-1)
        x = self.attn(x)

        x = F.softmax(x.squeeze(-1),dim=-1)
        # x: batch x seq1 x seq2 x 3
        # seq2 x 3
        x = torch.bmm(x,value)
        return x


# 定义网络
class testNet(nn.Module):
    def __init__(self):
        super(testNet,self).__init__()

        self.fc1 = nn.Linear(3,16)
        self.lstm = nn.LSTM(16,32,1,batch_first=True)
        self.fc2 = nn.Linear(38,16)
        self.fc3 = nn.Linear(16, 3)
        self.atte = Attention()

    def forward(self, x , atte_parameter):

        cat_dict = []
        for v in atte_parameter:
            value = self.atte(x,v[:,:-1],v[:,1:])
            cat_dict.append(value)

        x = self.fc1(x)
        #print(x.shape)
        #x = x.view(batch_size,-1,16)
        x, _ = self.lstm(x)
        cat_dict.append(x)

        x = torch.cat(cat_dict,dim=-1)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

net = testNet()
#optimizer = torch.optim.SGD(net.parameters(),lr = 0.00001,momentum=0.9)
optimizer = torch.optim.Adam(net.parameters(),lr = 0.01,weight_decay=1E-7)
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
    reslut = net(batch,atte_key)
    loss = loss1(reslut,label)
    return loss.detach()

save_name = "temp_model2.pkl"
load_name = "temp_model2.pkl" # 读取模型文件的名字，设为NULL重新训练
if load_name :
    net.load_state_dict(torch.load(load_name))

for epoch in range(0):
    step_loss = train(input_datas,ys,atte_key)
    print(step_loss)
    if step_loss < 0.1:
        break

if save_name:
    torch.save(net.state_dict(),save_name)

print("test loss ", test(input_datas,ys,atte_key))

def get_net():
    return net

if __name__ == '__main__':
    #test_data = [[38,21,41],[26,27,47],[24,36,40],[34,30,36],[24,37,39],[32,35,33],[30,26,44],[31.9,36,32.1],[24,32,44]] #渠道服
    #test_data = [[30,25,44],[23,29,48],[25,26,38],[35,31,34],[22,34,44],[26,41,34],[28,31,41],[32,34,33]]
    #输出结果 0:IOS 1:安卓官服 2:安卓b服

    name = ["IOS ","安官","安B "]
    for i in range(3):
        test_data = data[i]
        data_ = np.array(test_data)
        input_datas = torch.Tensor([data_])
        #print(input_datas)
        atte_key = torch.Tensor(
            [[input_data[(i+1)%3]], [input_data[(i+2)%3]]])
        reslut = net(input_datas,atte_key).detach().numpy()
        print(name[i],reslut[0][-1])
        #print(name[i], reslut[0])
        #a = reslut[0][:-1]-test_data[1:]
        #print(np.sum(np.abs(a)))