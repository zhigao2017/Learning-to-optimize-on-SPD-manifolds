import torch
import torch.nn as nn
from torch.autograd import Variable as V
from MatrixLSTM_lr.MatrixBiMul import MatrixBiMul


class MatrixLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size,output_size):
        super(MatrixLSTMCell, self).__init__()
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.f_gate_x=MatrixBiMul(input_size,hidden_size)
        self.i_gate_x=MatrixBiMul(input_size,hidden_size)
        self.o_gate_x=MatrixBiMul(input_size,output_size)
        self.f_gate_h=MatrixBiMul(hidden_size,hidden_size)
        self.i_gate_h=MatrixBiMul(hidden_size,hidden_size)
        self.o_gate_h=MatrixBiMul(hidden_size,output_size)

        self.o_gate_c=MatrixBiMul(hidden_size,output_size)

        self.g_gate_x=MatrixBiMul(input_size,hidden_size)
        self.g_gate_h=MatrixBiMul(hidden_size,hidden_size)

        #self.sigmoid1 = nn.Sigmoid()
        #self.sigmoid2 = nn.Sigmoid()
        #self.sigmoid3 = nn.Sigmoid()

        #self.sigmoid1 = nn.Tanh()
        #self.sigmoid2 = nn.Tanh()
        #self.sigmoid3 = nn.Tanh()

        self.sigmoid1 = nn.ReLU()
        self.sigmoid2 = nn.ReLU()
        self.sigmoid3 = nn.ReLU()

        #self.tanh1 = nn.Tanh()
        #self.tanh2 = nn.Tanh()

        self.tanh1 = nn.ReLU()
        self.tanh2 = nn.ReLU()

    def forward(self,input,hidden,cell):
        #print('MatrixLSTMCell inputs',input)

        fgatex=self.f_gate_x(input)
        igatex=self.i_gate_x(input)
        ogatex=self.o_gate_x(input)

        fgateh=self.f_gate_h(hidden)
        igateh=self.i_gate_h(hidden)
        ogateh=self.o_gate_h(hidden)

        #print('fgatex',fgatex)
        #print('fgateh',fgateh)

        fgate=self.sigmoid1(fgatex+fgateh)
        igate=self.sigmoid2(igatex+igateh)
        ogate=self.sigmoid3(ogatex+ogateh)

        #print('fgate',fgate)
        #print('igate',igate)
        #print('ogate',ogate)

        ggatex=self.g_gate_x(input)
        ggateh=self.g_gate_h(hidden)
        ggate=self.tanh1(ggatex+ggateh)

        #print('ggate',ggate)

        #c=torch.mul(fgate,cell)+torch.mul(igate,ggate)
        c=fgate+cell+igate+ggate
        #h=torch.mul(ogate,self.tanh2(self.o_gate_c(c)))
        h=ogate+self.tanh2(self.o_gate_c(c))


        return h, c
    
