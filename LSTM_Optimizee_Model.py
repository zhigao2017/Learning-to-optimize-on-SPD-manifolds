import torch
import torch.nn as nn
from torch.autograd import Variable as V
import torch.nn.functional as F

from MatrixLSTM.MatrixLSTM import MatrixLSTM
from MatrixLSTM_lr.MatrixLSTM import MatrixLSTM_lr


class LSTM_Optimizee_Model(nn.Module):    
    def __init__(self,opt,input_size, hidden_size, output_size, batchsize_data,batchsize_para):
        super(LSTM_Optimizee_Model,self).__init__()
        self.lstm=MatrixLSTM(input_size, hidden_size, output_size)
        self.lstm_lr=MatrixLSTM_lr(input_size, hidden_size, output_size)


        self.input_size=input_size
        self.hidden_size=hidden_size
        self.output_size=output_size
        self.batchsize_data=batchsize_data
        self.batchsize_para=batchsize_para

        self.scale=1

    
    def forward(self, input_gradients, prev_state):

        input_gradients = input_gradients.cuda()
        dim=input_gradients.shape[1]
        

        if prev_state is None: 
            prev_state = (torch.zeros(self.batchsize_para,self.hidden_size,self.hidden_size).cuda(),
                            torch.zeros(self.batchsize_para,self.hidden_size,self.hidden_size).cuda(),
                            torch.zeros(self.batchsize_para,self.hidden_size,self.hidden_size).cuda(),
                            torch.zeros(self.batchsize_para,self.hidden_size,self.hidden_size).cuda(),
                            )        
        
        update_dir , next_state_dir = self.lstm(input_gradients, prev_state)
        update_lr , next_state_lr= self.lstm_lr(input_gradients, prev_state)


        next_state=( torch.mul(next_state_dir[0],next_state_lr[0]), torch.mul(next_state_dir[1],next_state_lr[1]), torch.mul(next_state_dir[2],next_state_lr[2]), torch.mul(next_state_dir[3],next_state_lr[3])  )


        return update_lr, update_dir , next_state   