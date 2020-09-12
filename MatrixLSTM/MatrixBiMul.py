import torch
import torch.nn as nn
from torch.autograd import Variable as V



class MatrixBiMul(nn.Module):
    def __init__(self, input_size, output_size):
        super(MatrixBiMul, self).__init__()
        self.input_size=input_size
        self.output_size=output_size
        self.w=nn.Parameter(torch.randn(self.input_size, self.output_size),requires_grad =True)
        nn.init.orthogonal_(self.w)
        #self.b=nn.Parameter(torch.randn(self.output_size, self.output_size),requires_grad =True)
        #self.w=nn.Parameter(manifold=nn.Stiefel(self.input_size, self.output_size))

    def forward(self, input):
        #print('MatrixBiMul inputs',input)

        n=input.shape[0]
        Y=torch.matmul(torch.matmul(self.w.t(),input),self.w)
        
        '''
        Y=torch.zeros(n,self.output_size,self.output_size).cuda()
        for i in range(n):
            Y[i]=torch.mm(torch.mm(self.w.t(),input[i]),self.w)
        '''

        #print('---MatrixBiMul layer finish---')
        return Y
    
