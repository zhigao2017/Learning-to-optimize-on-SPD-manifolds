import torch
import torch.nn as nn
from torch.autograd import Variable as V
from torch.autograd import Function


class M_Log(nn.Module):
    def __init__(self):
        super(M_Log, self).__init__()
        self.beta=0.000001

    def forward(self, input1):
        n=input1.shape[0]
        dim=input1.shape[1]

        espison=torch.eye(dim)*self.beta
        espison=espison.cuda()
        espison=torch.unsqueeze(espison,0)
        input2=torch.where(input1-espison < 0, espison,input1)

        one=torch.ones(input2.shape).cuda()
        e=torch.eye(dim).cuda()


        output=torch.log(input2+one-e)

        return output