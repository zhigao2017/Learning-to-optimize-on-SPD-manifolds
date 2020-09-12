import torch
import torch.nn as nn
from torch.autograd import Variable as V
from MatrixLSTM.MatrixLSTMCell import MatrixLSTMCell
from MatrixLSTM.MatrixBiMul import MatrixBiMul

class MatrixLSTM(nn.Module):
    def __init__(self, input_size, hidden_size,output_size):
        super(MatrixLSTM, self).__init__()

        self.lstm1=MatrixLSTMCell(input_size, hidden_size)
        self.lstm2=MatrixLSTMCell(hidden_size, hidden_size)
        #self.lstm3=MatrixLSTMCell(hidden_size, hidden_size)
        self.proj=MatrixBiMul(hidden_size,output_size)
        #self.sigmoid = nn.Sigmoid()


    def forward(self,input,state):

        h1,c1=self.lstm1(input,state[0],state[1])
        h2,c2=self.lstm2(h1,state[2],state[3])
        #h3,c3=self.lstm3(h1,state[4],state[5])
        #print('h2',h2)

        output=self.proj(h2)
        #output=self.sigmoid(output)

    	#return output, (h1,c1,h2,c2,h3,c3)
        return output, (h1,c1,h2,c2)
        #return output, (h1,c1)
