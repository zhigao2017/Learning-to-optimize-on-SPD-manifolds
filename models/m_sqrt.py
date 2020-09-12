import torch
import torch.nn as nn
from torch.autograd import Variable as V
from torch.autograd import Function


class M_Sqrt(nn.Module):
    def __init__(self,sign):
        super(M_Sqrt, self).__init__()
        self.sign=sign
        #self.epsilon=0.000001
        self.epsilon=0.000001


    def forward(self, input1):
        n=input1.shape[0]
        dim=input1.shape[1]

        one=torch.ones(input1.shape).cuda()
        e=torch.eye(dim).cuda()

        output=input1+one-e
        output=torch.where(output > 0, output, self.epsilon*one)

        #print('output1',output)
        output=torch.sqrt(output)
        if self.sign==-1:
            #output=output+self.epsilon*e
            output=1/output
        output=output-one+e
        #print('sqrt output',output)

        #print('M_Sqrt finish')

        return output