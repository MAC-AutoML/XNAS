import os
import torch

def test_local():
    root = '/home/xfey/XNAS/exp/search/OFA_trial_25/kernel_1/checkpoints/'
    filename_prefix = 'model_epoch_'
    filename_postfix = '.pyth'

    for i in range(101, 110):
        ckpt = torch.load(root+filename_prefix+'0'+str(i)+filename_postfix)
        for k,v in ckpt['model_state'].items():
            # print(k)
            # if k.endswith('5to3_matrix'):
            if k == 'blocks.6.conv.depth_conv.conv.5to3_matrix':
                print(k, v[0:3, 0:3])
                break
        for k,v in ckpt['model_state'].items():
            # print(k)
            # if k.endswith('7to5_matrix'):
            if k == 'blocks.6.conv.depth_conv.conv.7to5_matrix':
                print(k, v[0:3, 0:3])
                break

def test_original():
    root = "/home/xfey/XNAS/tests/weights/"
    ckpt = torch.load(root+"ofa_D4_E6_K357")
    for k,v in ckpt['state_dict'].items():
        # print(k)
        if k.endswith('5to3_matrix') and k.startswith('blocks.6'):
            print(k, v[0:3, 0:3])
            break
    for k,v in ckpt['state_dict'].items():
        # print(k)
        if k.endswith('7to5_matrix') and k.startswith('blocks.6'):
            print(k, v[0:3, 0:3])
            break

if __name__ == '__main__':
    # test_local()
    test_original()
