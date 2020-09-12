import argparse

def parse_opt():
    parser = argparse.ArgumentParser()
    # Data input settings
    parser.add_argument('--DIM', type=int, default=5)
    parser.add_argument('--batchsize_para', type=int, default=20)
    parser.add_argument('--batchsize_data', type=int, default=512)
    parser.add_argument('--datapath', type=str, default='data')

    parser.add_argument('--prepath', type=str, default='models/d5_twoLSTMtwo_updatestate_Optimizeelr_3_5000.pth')     
    parser.add_argument('--prepath2', type=str, default='models/d5_twoLSTM_Optimizee_lr0.001_iteration1000_nomoment_30_3.pth') 
    
    parser.add_argument('--Decay', type=int,default=5000)
    parser.add_argument('--savemodel', type=int,default=200)
    parser.add_argument('--Decay_rate', type=float,default=0.5)

    parser.add_argument('--category_num', type=int,default=20)
    parser.add_argument('--sample_num', type=int,default=1)
    parser.add_argument('--loadpretrain', type=bool,default=True)
    parser.add_argument('--Content', type=int, default=300)
    parser.add_argument('--Observe', type=int, default=300)
    parser.add_argument('--Epochs', type=int, default=20000)
    parser.add_argument('--Optimizee_Train_Steps', type=int, default=100)
    parser.add_argument('--train_steps', type=int, default=5)

    parser.add_argument('--optimizer_lr', type=float, default=0.0001)
    parser.add_argument('--hand_optimizer_lr', type=float, default=1)
    args = parser.parse_args()
    return args