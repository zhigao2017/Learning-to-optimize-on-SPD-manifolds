import argparse

def parse_opt():
    parser = argparse.ArgumentParser()
    # Data input settings
    parser.add_argument('--DIM', type=int, default=5)
    parser.add_argument('--batchsize_para', type=int, default=28)
    parser.add_argument('--batchsize_data', type=int, default=512)
    parser.add_argument('--datapath', type=str, default='data')
    parser.add_argument('--prepath', type=str, default='snapshot/itera2400.pth')     
    parser.add_argument('--UnRoll_STEPS', type=int, default=5)
    args = parser.parse_args()
    return args