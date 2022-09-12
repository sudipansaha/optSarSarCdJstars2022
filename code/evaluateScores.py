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

from skimage import morphology


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
EVIDENTIAL_STATISTICS = 1.498 ## Computed from the training images, Seed 19: 1.498 (please change it as per the seed used)


dimWiseSlice = 5  # along each dimension, number of slices during test time (helps to conserve memory)
trainedModelLoadPath = os.path.join('FolderWhereTheModelIsStored')  ##Path to the folder where trained model is stored

resultFiguresSavePath = os.path.join('./outputFigures/')




def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



# run network on full-scene ROI and evaluate performance,
# this expects
def test(dset):
    net.eval()
    totLoss = 0
    totCount = 0

    classCorrect = list(0. for i in range(2))
    classTotal = list(0. for i in range(2))
    classAccuracy = list(0. for i in range(2))

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    
    # iterate over all ROI, load modalities and
    for img_index in dset.names:
        print(f"Testing for ROI {img_index}")
        # inserted multi-modality here
        #S2_1_full, S2_2_full, cm_full = dset.get_img(img_index)
        fullImgs = dset.get_img(img_index)
        S2_1_full, S2_2_full, cm_full = fullImgs['time_1']['S2'], fullImgs['time_2']['S2'], fullImgs['label']
        s = cm_full.shape
        
        predictedFull = torch.zeros(s)

        if TYPE in [4, 5]:
            S1_1_full, S1_2_full = fullImgs['time_1']['S1'], fullImgs['time_2']['S1']

        stepsRow = np.arange(0, s[0], ceil(s[0] / dimWiseSlice))
        stepsCol = np.arange(0, s[1], ceil(s[1] / dimWiseSlice))
        for ii in range(dimWiseSlice):
            for jj in range(dimWiseSlice):
                rowMin = stepsRow[ii]
                if ii == dimWiseSlice - 1:
                    rowMax = s[0]
                else:
                    rowMax = stepsRow[ii + 1]
                colMin = stepsCol[jj]
                if jj == dimWiseSlice - 1:
                    colMax = s[1]
                else:
                    colMax = stepsCol[jj + 1]
                
                # inserted multi-modality here
                S2_1 = S2_1_full[:, rowMin:rowMax, colMin:colMax]
                S2_2 = S2_2_full[:, rowMin:rowMax, colMin:colMax]
                cm = cm_full[rowMin:rowMax, colMin:colMax]

                S2_1 = Variable(torch.unsqueeze(S2_1, 0).float()).cuda()
                S2_2 = Variable(torch.unsqueeze(S2_2, 0).float()).cuda()
                cm = Variable(torch.unsqueeze(torch.from_numpy(1.0 * cm), 0).float()).cuda()

                if TYPE in [4, 5]:
                    S1_1 = S1_1_full[:, rowMin:rowMax, colMin:colMax]
                    S1_2 = S1_2_full[:, rowMin:rowMax, colMin:colMax]
                    S1_1 = Variable(torch.unsqueeze(S1_1, 0).float()).cuda()
                    S1_2 = Variable(torch.unsqueeze(S1_2, 0).float()).cuda()

                # predict output via network and compute losses
                
                if TYPE in [4, 5]:
                    output = net(S2_1, S2_2, S1_1, S1_2)
                else:
                    output = net(S2_1, S2_2)
                totCount   += np.prod(cm.size())

                

                offsetValue = 1*(EVIDENTIAL_STATISTICS-torch.mean(output[0,0,:,:]))/EVIDENTIAL_STATISTICS
                output[0,0,:,:]=output[0,0,:,:]+offsetValue
                
                
                _, predicted = torch.max(output.data, 1)
                
                predictedFull[rowMin:rowMax,colMin:colMax] = predicted 

                # compare predictions with change maps and count correct predictions
                c = (predicted.int() == cm.data.int())
                for i in range(c.size(1)):
                    for j in range(c.size(2)):
                        l = int(cm.data[0, i, j])
                        classCorrect[l] += c[0, i, j]
                        classTotal[l] += 1

                pr = (predicted.int() > 0).cpu().numpy()
                #pr = morphology.remove_small_objects(pr, 32)
                gt = (cm.data.int() > 0).cpu().numpy()

                # evaluate TP, TN, FP & FN
                tp += np.logical_and(pr, gt).sum()
                tn += np.logical_and(np.logical_not(pr), np.logical_not(gt)).sum()
                fp += np.logical_and(pr, np.logical_not(gt)).sum()
                fn += np.logical_and(np.logical_not(pr), gt).sum()
                

        predictedFull = 1-predictedFull
        cm_fullModified = 1- cm_full
        I = np.stack((255*np.squeeze(predictedFull.cpu().numpy()),255*np.squeeze(predictedFull.cpu().numpy()),255*np.squeeze(predictedFull.cpu().numpy())),2)
        fccImage = np.stack((255*(cm_fullModified),255*np.squeeze(predictedFull.cpu().numpy()),255*(cm_fullModified)),2)
        #io.imsave(os.path.join(resultFiguresSavePath, f'{img_index}.png'), I)
        #io.imsave(os.path.join(resultFiguresSavePath, f'{img_index}_FCC.png'), fccImage)



    # compute goodness of predictions
    netAccuracy = 100 * (tp + tn) / totCount

    for i in range(2):  # compute classwise accuracies, 2 is number of classes here: changed and unchanged
        classAccuracy[i] = 100 * classCorrect[i] / max(classTotal[i], 0.00001)

    # get precision, recall etc
    prec    = tp / (tp + fp)
    rec     = tp / (tp + fn)
    f_meas  = 2 * prec * rec / (prec + rec)
    prec_nc = tn / (tn + fn)
    rec_nc  = tn / (tn + fp)
    pr_rec  = [prec, rec, f_meas, prec_nc, rec_nc]

    return netAccuracy, classAccuracy, pr_rec




if __name__ == '__main__':


    ## test data
    testDataset = multiCD(PATH_TO_DATASET, split="test", run_on_onera_patches=ONERA_PATCHES, normalize=NORMALISE_IMGS)
    


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
    print('We now proceed for test')
    net.load_state_dict(torch.load(os.path.join(trainedModelLoadPath, 'checkpoints', 'netBest.pth.tar')))
    print('Best model loaded for testing')
    netAccuracyTest, classAccuracyTest, prRecTest = test(testDataset)
    
    print('Net accuracy on test data is: '+str(netAccuracyTest))    

    print('Precision computed on test data is'+str(prRecTest[0]))
    print('Recall computed on test data is'+str(prRecTest[1]))
    print('F1 score computed on test data is'+str(prRecTest[2]))


