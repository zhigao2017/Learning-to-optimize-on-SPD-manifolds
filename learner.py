import torch
import torch.nn as nn
from torch.autograd import Variable 
import time


from models.EigLayer import EigLayer
from models.m_log import M_Log

from retraction import Retraction

eiglayer1=EigLayer()
mlog=M_Log()
def f(inputs,M,label):

    n=inputs.shape[0]
    loss=0
    for i in range(n):
        l=label[i]-1
        AIM=torch.mm(torch.mm(inputs[i],M[l]),inputs[i])
        AIM=torch.unsqueeze(AIM,0)
        M_S,M_U=eiglayer1(AIM)
        M_Sl=mlog(M_S)

        AIM=torch.matmul(torch.matmul(M_U,M_Sl),M_U.permute(0,2,1))

        p=torch.sum(torch.sum(torch.pow(AIM,2),2),1)
        loss=loss+torch.mean(p)
    loss=loss/n
    return loss




class Learner( object ):
    def __init__(self, DIM, batchsize_para,  optimizee, train_steps ,  
                                            retain_graph_flag=False,
                                            reset_theta = False ,
                                            reset_function_from_IID_distirbution = True):
        self.optimizee = optimizee
        self.beta=1
        self.train_steps = train_steps
        self.retain_graph_flag = retain_graph_flag
        self.reset_theta = reset_theta
        self.reset_function_from_IID_distirbution = reset_function_from_IID_distirbution  
        self.state = None


        self.DIM=DIM
        self.batchsize_para=batchsize_para
        self.retraction=Retraction(1)

        for parameters in optimizee.parameters():
            print(torch.sum(parameters))

        self.M=torch.randn(self.batchsize_para,self.DIM, self.DIM)
        for i in range(self.batchsize_para):
   
            self.M[i]=torch.eye(self.DIM)

        self.M=self.M.cuda()
        self.M.requires_grad=True

        self.P_tangent=torch.zeros(self.batchsize_para,self.DIM, self.DIM).cuda()



        
            
    def Reset_Or_Reuse(self ,num_roll, M, P_tangent, state):


        reset_theta =self.reset_theta
        reset_function_from_IID_distirbution = self.reset_function_from_IID_distirbution

       
        if num_roll == 0 and reset_theta == True:
            M=torch.randn(self.batchsize_para,self.DIM, self.DIM)
            for i in range(self.batchsize_para):
                M[i]=torch.eye(self.DIM)
            
        if num_roll == 0:
            state = None
            P_tangent=torch.zeros(self.batchsize_para,self.DIM, self.DIM).cuda()
            
        M = M.cuda()
        M.requires_grad=True

        return M, P_tangent,state
          
            
    def __call__(self, train_loader,num_roll=0) : 


        M, P_tangent, state =  self.Reset_Or_Reuse( num_roll, self.M, self.P_tangent, self.state )
        self.global_loss_graph = 0
        optimizee = self.optimizee



        count=0
        break_flag=False
        flag=False
        while(1):
            for j, data in enumerate(train_loader, 0):


                inputs, labels = data
                inputs = Variable(inputs.cuda())
                labels = Variable(labels).cuda()
                loss = f(inputs,M,labels)
                localtime = time.asctime( time.localtime(time.time()) )
                loss.backward() 

                lr, update, state = optimizee(M.grad, state)
                update=update+M.grad
                M.grad.data.zero_()
                update_R=torch.matmul(torch.matmul(M, (update+update.permute(0,2,1))/2),M)
                flag, M = self.retraction(M,update_R, lr)

                state = (state[0].detach(),state[1].detach(),state[2].detach(),state[3].detach())
                M = M.detach()
                M.requires_grad = True
                M.retain_grad()


                count=count+1
                total_loss=0
                for k, subdata in enumerate(train_loader, 0):
                    inputs, labels = subdata
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels).cuda()

                    loss = f(inputs,M,labels)
                    total_loss += loss.detach()
                print('count',count,'total loss',total_loss)  




