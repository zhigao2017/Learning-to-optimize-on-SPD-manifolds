import torch
import torch.nn as nn
from torch.autograd import Variable as V
from timeit import default_timer as timer

from learning_to_learn import Learning_to_learn_global_training
from LSTM_Optimizee_Model import LSTM_Optimizee_Model
from DataSet.KYLBERG import KYLBERG

import config_evaluation
from learner import Learner






opt = config_evaluation.parse_opt()
print(opt)
LSTM_Optimizee = LSTM_Optimizee_Model(opt, opt.DIM, opt.DIM, opt.DIM, batchsize_data=opt.batchsize_data, batchsize_para=opt.batchsize_para).cuda()

checkpoint = torch.load(opt.prepath)
LSTM_Optimizee.load_state_dict(checkpoint)


train_mnist = KYLBERG(opt.datapath, train=True)
train_loader = torch.utils.data.DataLoader(
        train_mnist, batch_size=opt.batchsize_data,shuffle=True, drop_last=False, num_workers=0)

LSTM_Learner = Learner(opt.DIM, opt.batchsize_para, LSTM_Optimizee, opt.UnRoll_STEPS, retain_graph_flag=True, reset_theta=True,reset_function_from_IID_distirbution = True)
LSTM_Learner(train_loader)   



