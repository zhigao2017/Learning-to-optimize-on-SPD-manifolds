import torch
import torch.nn as nn
from torch.autograd import Variable as V
from MatrixLSTM_lr.MatrixLSTMCell import MatrixLSTMCell
from MatrixLSTM_lr.MatrixBiMul import MatrixBiMul
import torch.nn.functional as F

class MatrixLSTM_lr(nn.Module):
    def __init__(self, input_size, hidden_size,output_size):
        super(MatrixLSTM_lr, self).__init__()

        self.lstm1=MatrixLSTMCell(input_size, hidden_size,hidden_size)
        self.lstm2=MatrixLSTMCell(hidden_size, hidden_size,hidden_size)

        self.proj=nn.Linear(hidden_size*hidden_size,25)
        self.proj2=nn.Linear(25,1)



    def forward(self,input,state):

        h1,c1=self.lstm1(input,state[0],state[1])
        h2,c2=self.lstm2(h1,state[2],state[3])


        #output=self.proj(h2)


        h3=h2.view(h2.shape[0],-1)
        output=self.proj(h3)
        output=F.relu(output)
        output=self.proj2(output)
        output=torch.unsqueeze(output,2)


        return output, (h1,c1,h2,c2)

