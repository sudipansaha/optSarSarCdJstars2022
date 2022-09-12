## Code was developed in different phases by Sudipan Saha and Patrick Ebel
## Additionally, some functions were taken from the repository of OSCD dataset

# PyTorch
import torch
import torch.nn as nn
from torch.autograd import Variable

# Data
from dataLoader import multiCD

# Models
from models.siamunetConc import SiamUnet_conc_multi

# Other
import os
import numpy as np
import random
from skimage import io
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from pandas import read_csv
from math import floor, ceil, sqrt, exp
from IPython import display
import time
import warnings



# Global Variables' Definitions
PATH_TO_DATASET = "../../data/ONERA_s1_s2/"
FP_MODIFIER = 5  # Tuning parameter, use 1 if unsure
BATCH_SIZE = 32  # number of elements in a batch
NUM_THREADS = 2  # number of parallel threads in data loader
NET = 'SiamUnet_conc'  # 'Unet', 'SiamUnet_conc-simple', 'SiamUnet_conc', 'SiamUnet_diff', 'FresUNet'
N_EPOCHS = 200   # number of epochs to train the network
TYPE = 4  #  type of input to the network: 0-RGB | 1-RGBIr | 2-All bands s.t. resulution <= 20m | 3-All bands | 4-All bands Sentinel-2 & Sentinel-1 | 5-RGB bands Sentinel-2 & Sentinel-1
DATA_AUG = True  # whether to apply data augmentation (mirroring, rotating) or not
ONERA_PATCHES = True  # whether to train on patches sliced on the original Onera images or not
NORMALISE_IMGS = False  # z-standardizing on full-image basis, note: only implemented for online slicing!


dimWiseSlice = 2  # along each dimension, number of slices during test time (helps to conserve memory)

augm_str  = 'Augm' if DATA_AUG else 'noAugm'
save_path = os.path.join('FolderWhereModelIsStored')  ##Specify the folder where the model is stored




def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



# run network on full-scene ROI and evaluate performance,
# this expects
def computeEvidentialStatistics(dset):
    net.eval()
    totCount = 0
    evidenceSum = 0

    # iterate over all ROI, load modalities and
    for img_index in dset.names:
        print(f"Testing for ROI {img_index}")
        # inserted multi-modality here
        #S2_1_full, S2_2_full, cm_full = dset.get_img(img_index)
        fullImgs = dset.get_img(img_index)
        S2_1_full, S2_2_full, cm_full = fullImgs['time_1']['S2'], fullImgs['time_2']['S2'], fullImgs['label']
        s = cm_full.shape

        if TYPE in [4, 5]:
            S1_1_full, S1_2_full = fullImgs['time_1']['S1'], fullImgs['time_2']['S1']

        steps0 = np.arange(0, s[0], ceil(s[0] / dimWiseSlice))
        steps1 = np.arange(0, s[1], ceil(s[1] / dimWiseSlice))
        for ii in range(dimWiseSlice):
            for jj in range(dimWiseSlice):
                xmin = steps0[ii]
                if ii == dimWiseSlice - 1:
                    xmax = s[0]
                else:
                    xmax = steps0[ii + 1]
                ymin = steps1[jj]
                if jj == dimWiseSlice - 1:
                    ymax = s[1]
                else:
                    ymax = steps1[jj + 1]
                # inserted multi-modality here
                S2_1 = S2_1_full[:, xmin:xmax, ymin:ymax]
                S2_2 = S2_2_full[:, xmin:xmax, ymin:ymax]
                cm = cm_full[xmin:xmax, ymin:ymax]

                S2_1 = Variable(torch.unsqueeze(S2_1, 0).float()).cuda()
                S2_2 = Variable(torch.unsqueeze(S2_2, 0).float()).cuda()
                cm = Variable(torch.unsqueeze(torch.from_numpy(1.0 * cm), 0).float()).cuda()

                if TYPE in [4, 5]:
                    S1_1 = S1_1_full[:, xmin:xmax, ymin:ymax]
                    S1_2 = S1_2_full[:, xmin:xmax, ymin:ymax]
                    S1_1 = Variable(torch.unsqueeze(S1_1, 0).float()).cuda()
                    S1_2 = Variable(torch.unsqueeze(S1_2, 0).float()).cuda()

                # predict output via network and compute losses
                
                if TYPE in [4, 5]:
                    output = net(S2_1, S2_2, S1_1, S1_2)
                else:
                    output = net(S2_1, S2_2)
                totCount   += np.prod(cm.size())


                print(evidenceSum)
                print((output[0,0,:,:]).sum())
                evidenceSum=evidenceSum+((output[0,0,:,:]).sum()).detach().cpu().numpy()
    evlidentialStatistics = evidenceSum/totCount
    return evlidentialStatistics







if __name__ == '__main__':


    ## test data
    testDataset = multiCD(PATH_TO_DATASET, split="train", run_on_onera_patches=ONERA_PATCHES, normalize=NORMALISE_IMGS)
    


    # 0-RGB | 1-RGBIr | 2-All bands s.t. resolution <= 20m | 3-All bands | 4-All bands Sentinel-2 & Sentinel-1 | 5-RGB bands Sentinel-2 & Sentinel-1


    if TYPE == 4:
        if NET == 'Unet': net, net_name = Unet(2*13+2*2, 2), 'FC-EF-multi' # same architecture as the other network
        if NET == 'SiamUnet_conc-simple': net, net_name = SiamUnet_conc(13+2, 2), 'FC-Siam-conc-simple'
        if NET == 'SiamUnet_conc': net, net_name = SiamUnet_conc_multi((13, 2), 2), 'FC-Siam-conc-complex'
        if NET == 'SiamUnet_diff': net, net_name = SiamUnet_diff_multi(13, 2, 2), 'FC-Siam-diff'
        if NET == 'FresUNet': net, net_name = FresUNet_multi(2 * 13 + 2*13, 2), 'FresUNet'
    elif TYPE == 5:
        if NET == 'Unet': net, net_name = Unet(2*3+2*2, 2), 'FC-EF-multi'  # same architecture as the other network
        if NET == 'SiamUnet_conc-simple': net, net_name = SiamUnet_conc(3+2, 2), 'FC-Siam-conc-simple'
        if NET == 'SiamUnet_conc': net, net_name = SiamUnet_conc_multi((3, 2), 2), 'FC-Siam-conc-complex'
        if NET == 'SiamUnet_diff': net, net_name = SiamUnet_diff_multi(3, 2, 2), 'FC-Siam-diff'
        if NET == 'FresUNet': net, net_name = FresUNet_multi(2 * 3 + 2*3, 2), 'FresUNet'
    net.cuda()


    
    ## After training, load the best model and test on it
    # train the network
    print('We now proceed to compute the statistics')
    net.load_state_dict(torch.load(os.path.join(save_path, 'checkpoints', 'netBest.pth.tar')))
    print('Best model loaded for computing statistics')
    evlidentialStatistics = computeEvidentialStatistics(testDataset)
    
    print('Computed evidential statistics is: '+str(evlidentialStatistics)) 