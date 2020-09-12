import torch
import torch.nn as nn
from torch.autograd import Variable
from timeit import default_timer as timer
import time
import math, random

from ReplyBuffer import ReplayBuffer
from models.EigLayer import EigLayer
from models.m_log import M_Log
from retraction import Retraction

eiglayer1=EigLayer()
mlog=M_Log()
retraction=Retraction(1)

def f(inputs,M,label,sample_num):

    n=inputs.shape[0]
    loss=0
    for i in range(n):
        l=label[i]-1
        for j in range(sample_num):
            
            AIM=torch.mm(torch.mm(inputs[i],M[l*sample_num+j]),inputs[i])
            AIM=torch.unsqueeze(AIM,0)
            M_S,M_U=eiglayer1(AIM)
            M_Sl=mlog(M_S)
            AIM=torch.matmul(torch.matmul(M_U,M_Sl),M_U.permute(0,2,1))
            

            p=torch.sum(torch.sum(torch.pow(AIM,2),2),1)
            loss=loss+torch.mean(p)
    loss=loss/(n*sample_num)
    return loss


def adjust_learning_rate(optimizer, decay_rate):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate


def Learning_to_learn_global_training(opt,hand_optimizee,optimizee,train_loader):

    
    DIM=opt.DIM
    batchsize_para=opt.batchsize_para
    Observe=opt.Observe
    Epochs=opt.Epochs
    Optimizee_Train_Steps=opt.Optimizee_Train_Steps
    optimizer_lr=opt.optimizer_lr
    Decay=opt.Decay
    Decay_rate=opt.Decay_rate

    adam_global_optimizer = torch.optim.Adamax(optimizee.parameters(),lr = optimizer_lr)

    RB_list=[]
    for i in range(opt.category_num):
        RB=ReplayBuffer(opt.Content*opt.sample_num)  
        RB_list.append(RB)


    for i in range(Observe):

        if i ==0:
            M=torch.randn(batchsize_para,DIM, DIM).cuda()
            for k in range(batchsize_para):
                M[k]=torch.eye(DIM).cuda()

            state = (torch.zeros(batchsize_para,DIM,DIM).cuda(),
                                     torch.zeros(batchsize_para,DIM,DIM).cuda(),
                                     torch.zeros(batchsize_para,DIM,DIM).cuda(),
                                     torch.zeros(batchsize_para,DIM,DIM).cuda(),
                                     ) 
            iteration=torch.zeros(batchsize_para)
            M.requires_grad=True

            num=0
            for k in range(opt.category_num):
                for l in range(opt.sample_num):
                    RB_list[k].push((state[0][num],state[1][num],state[2][num],state[3][num]),M[num],iteration[num])
                    num=num+1

            count=1
            print ('observe finish',count)
            localtime = time.asctime( time.localtime(time.time()) )
            print ("local time", localtime)
            
        break_flag=False
        for j, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs = Variable(inputs.cuda())
            labels = Variable(labels).cuda()
            
            loss = f(inputs,M,labels,opt.sample_num)
            loss.backward()

            M, state = hand_optimizee(M.grad, M, state)
            iteration=iteration+1
            
            for k in range(batchsize_para):
                if iteration[k]>=Optimizee_Train_Steps-opt.train_steps:       
                    M[k]=torch.eye(DIM).cuda()
                    state[0][k]=torch.zeros(DIM,DIM).cuda()
                    state[1][k]=torch.zeros(DIM,DIM).cuda()
                    state[2][k]=torch.zeros(DIM,DIM).cuda()
                    state[3][k]=torch.zeros(DIM,DIM).cuda()   
                    iteration[k]=0
            

            state = (state[0].detach(),state[1].detach(),state[2].detach(),state[3].detach())
            M=M.detach()
            M.requires_grad=True

            num=0
            for k in range(opt.category_num):
                for l in range(opt.sample_num):
                    RB_list[k].push((state[0][num],state[1][num],state[2][num],state[3][num]),M[num],iteration[num])
                    num=num+1

            count=count+1
            print ('loss',loss)
            print ('observe',count)
            localtime = time.asctime( time.localtime(time.time()) )
            print ("local time", localtime)

            if count==Observe:
                break_flag=True
                break
        if break_flag==True:
            break                         

    for i in range(opt.category_num):
        RB_list[i].shuffle()



    check_point=optimizee.state_dict()
    check_point2=optimizee.state_dict()
    for i in range(Epochs): 
        print('\n=======> global training steps: {}'.format(i))
        if i % Decay==0 and i != 0:
            count=count+1
            adjust_learning_rate(adam_global_optimizer, Decay_rate)

        if i % opt.savemodel==0 and i != 0:
            if opt.loadpretrain == True:
                torch.save(optimizee.state_dict(), 'snapshot/itera'+str(i)+'_'+str(opt.optimizer_lr*1000)+'_hand_optimizer_lr'+str(opt.hand_optimizer_lr)+'_Observe'+str(opt.Observe)+'_Epochs'+str(opt.Epochs)+'_Optimizee_Train_Steps'+str(opt.Optimizee_Train_Steps)+'_train_steps'+str(opt.train_steps)+'.pth')
        if i==0:
            global_loss_graph=0
            train_loss=0
        else:
            global_loss_graph=global_loss_graph.detach()
            global_loss_graph=0
            train_loss=train_loss.detach()
            train_loss=0

        M=(torch.randn(batchsize_para,DIM, DIM)).cuda()
        state = (torch.zeros(batchsize_para,DIM,DIM).cuda(),
                                torch.zeros(batchsize_para,DIM,DIM).cuda(),
                                torch.zeros(batchsize_para,DIM,DIM).cuda(),
                                torch.zeros(batchsize_para,DIM,DIM).cuda(),
                                ) 
        iteration=torch.zeros(batchsize_para)

        num=0
        for k in range(opt.category_num):
            state_s, M_s, iteration_s= RB_list[k].sample(opt.sample_num)
            for l in range(opt.sample_num):
                M[num]=M_s[l].detach()
                iteration[num]=iteration_s[l].detach()
                state[0][num]=state_s[l][0].detach()
                state[1][num]=state_s[l][1].detach()
                state[2][num]=state_s[l][2].detach()
                state[3][num]=state_s[l][3].detach()
                num=num+1
        M.requires_grad = True
        M.retain_grad()
        

        break_flag=False
        flag=False
        count=0
        adam_global_optimizer.zero_grad()
        while(1):
            for j, data in enumerate(train_loader, 0):

                print('---------------------------------------------------------------------------')
                inputs, labels = data
                inputs = Variable(inputs.cuda())
                labels = Variable(labels).cuda()

                loss = f(inputs,M,labels,opt.sample_num)
                train_loss=train_loss+loss
                loss.backward(retain_graph=True)
                global_loss_graph=global_loss_graph+loss

                print('count',count)
                lr, update, state = optimizee(M.grad, state)

                M.grad.data.zero_()

                update_R=torch.matmul(torch.matmul(M, (update+update.permute(0,2,1))/2),M)
                flag,M = retraction(M,update_R,lr)
                

                print('old loss',loss)
                localtime = time.asctime( time.localtime(time.time()) )
                print ("local time", localtime)

                iteration=iteration+1

                M.retain_grad()
                update.retain_grad()

                count=count+1
                if count==opt.train_steps:
                    break_flag=True
                    break   

                if flag == True:
                    break_flag=True
                    break

            if break_flag==True:
                break 

        adam_global_optimizer.zero_grad()
        global_loss_graph.backward()
        if flag==False:           
            adam_global_optimizer.step()

            for k in range(batchsize_para):
                if iteration[k]>=Optimizee_Train_Steps-opt.train_steps:
                    M[k]=torch.eye(DIM)
                    state[0][k]=torch.zeros(DIM,DIM).cuda()
                    state[1][k]=torch.zeros(DIM,DIM).cuda()
                    state[2][k]=torch.zeros(DIM,DIM).cuda()
                    state[3][k]=torch.zeros(DIM,DIM).cuda()     
                    iteration[k]=0
            
            num=0
            for k in range(opt.category_num):
                for l in range(opt.sample_num):
                    RB_list[k].push((state[0][num],state[1][num],state[2][num],state[3][num]),M[num],iteration[num])
                    num=num+1

            check_point=check_point2
            check_point2=optimizee.state_dict()
        else:
            print('=====>eigenvalue break, reloading check_point')
            optimizee.load_state_dict(check_point)

        print('=======>global_loss_graph',global_loss_graph)
