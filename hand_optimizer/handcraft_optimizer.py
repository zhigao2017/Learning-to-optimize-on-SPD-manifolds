import torch
import torch.nn as nn
from torch.autograd import Variable as V
from MatrixLSTM.MatrixLSTM import MatrixLSTM
from hand_optimizer.retraction import Retraction

class Hand_Optimizee_Model(nn.Module): 
    def __init__(self,lr):
        super(Hand_Optimizee_Model,self).__init__()
        self.lr=lr
        self.retraction=Retraction(self.lr)
        self.w=nn.Parameter(torch.randn(2,2),requires_grad =True)

    def forward(self,grad,M,state):
        grad_R = torch.matmul( torch.matmul(M, (grad+grad.permute(0,2,1))/2),M )
        M = self.retraction(M,grad_R)

        return M,state

