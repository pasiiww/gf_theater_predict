from test1 import get_net as oldmodel
from model_with_attention import get_net as newmodel


from data import *

model = 0
if model == 0:
    net = newmodel()
else:
    net = oldmodel()

if __name__ == '__main__':
    #test_data = [[38,21,41],[26,27,47],[24,36,40],[34,30,36],[24,37,39],[32,35,33],[30,26,44],[31.9,36,32.1],[24,32,44]] #渠道服
    test_data = [[30,25,44],[23,29,48],[25,26,38],[35,31,34],[22,34,44],[26,41,34],[28,31,41],[32,34,33]]
    #输出结果 0:IOS 1:安卓官服 2:安卓b服

    name = ["IOS ","安官","安B "]
    for i in range(3):
        test_data = data[i]
        data_ = np.array(test_data)
        input_datas = torch.Tensor([data_])
        #print(input_datas)
        atte_key = torch.Tensor(
            [[input_data[(i+1)%3]], [input_data[(i+2)%3]]])
        if model == 0:
            reslut = net(input_datas,atte_key).detach().numpy()
        else:
            reslut = net(input_datas).detach().numpy()
        print(name[i],reslut[0][-1])
        #print(name[i], reslut[0])
        #a = reslut[0][:-1]-test_data[1:]
        #print(np.sum(np.abs(a)))