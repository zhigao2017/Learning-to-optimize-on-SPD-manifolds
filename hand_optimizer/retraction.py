import torch
import torch.nn as nn
from torch.autograd import Variable as V
from torch.autograd import Function

from models.EigLayer import EigLayer
from models.m_sqrt import M_Sqrt
from models.m_exp import M_Exp

class Retraction(nn.Module):
    def __init__(self,lr):
        super(Retraction, self).__init__()

        self.beta=lr

        self.eiglayer1=EigLayer()
        self.eiglayer2=EigLayer()
        self.msqrt1=M_Sqrt(1)
        self.msqrt2=M_Sqrt(-1)
        self.mexp=M_Exp()


    def forward(self, inputs, grad):

        #print('inputs',torch.sum(inputs))
        #print('grad',torch.sum(grad))
        #print('here------------1-------------')
        M_S,M_U=self.eiglayer1(inputs)
        #print('M_S',torch.sum(M_S))
        #print('M_U',torch.sum(M_U))

        #print('here------------2-------------')
        M_S1=self.msqrt1(M_S)
        M_S2=self.msqrt2(M_S)
        #print('M_S1 sum',torch.sum(M_S1))
        #print('M_S2 sum',torch.sum(M_S2))

        #print('here------------3-------------')

        M_1=torch.matmul( torch.matmul( M_U, M_S1 ), M_U.permute(0,2,1) )
        M_2=torch.matmul( torch.matmul( M_U, M_S2 ), M_U.permute(0,2,1) )
        #print('M_1',torch.sum(M_1))
        #print('M_2',torch.sum(M_2))

        #print('here------------4-------------')
        M_e=torch.matmul( torch.matmul( M_2, grad ), M_2 )
        M_e=-self.beta*M_e
        #print('here------------4.5-------------')
        #print('M_e',torch.sum(M_e))
        #M_eS,M_eU=self.eiglayer2(-1*self.beta*M_e)
        M_eS,M_eU=self.eiglayer2(M_e)
        #print('M_eS',torch.sum(M_eS))

        #print('here------------5-------------')
        flag, MeSe=self.mexp(M_eS)
        #print('M_eSe',torch.sum(MeSe))
        M_e=torch.matmul( torch.matmul( M_eU,MeSe ), M_eU.permute(0,2,1) )

        #print('here------------6-------------')
        M=torch.matmul( torch.matmul( M_1, M_e ), M_1 )
        #M=M+self.epsilon*torch.eye(M.shape[1]).cuda()
        #print('M',M)

        return M