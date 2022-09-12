## Code was developed in different phases by Sudipan Saha and Patrick Ebel
## Additionally, some functions were taken from the repository of OSCD dataset

## Code author: Sudipan Saha, Patrick Ebel

# PyTorch
import torch
import torch.nn as nn
from torch.autograd import Variable

# Data
from dataLoader import multiCD

# Models
from models.siamunetConc import SiamUnet_conc, SiamUnet_conc_multi



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
import argparse


### Command line arguments
### Only manual seed is set as command line argument. Rest are set in the code itself.
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--manualSeed', type=int, default=40, help='manual seed')
opt = parser.parse_args()
manualSeed=opt.manualSeed
print('Manual seed is '+str(manualSeed))
torch.manual_seed(manualSeed)
torch.cuda.manual_seed_all(manualSeed)
np.random.seed(manualSeed)



# Global Variables' Definitions
PATH_TO_DATASET = "../../data/ONERA_s1_s2/"
BATCH_SIZE = 32  # number of elements in a batch
NUM_THREADS = 2  # number of parallel threads in data loader
NET = 'SiamUnet_conc'  # 'Unet', 'SiamUnet_conc-simple', 'SiamUnet_conc', 'SiamUnet_diff', 'FresUNet'
N_EPOCHS = 50    # number of epochs to train the network
TYPE = 4  #  type of input to the network: 0-RGB | 1-RGBIr | 2-All bands s.t. resulution <= 20m | 3-All bands | 4-All bands Sentinel-2 & Sentinel-1 | 5-RGB bands Sentinel-2 & Sentinel-1
DATA_AUG = True  # whether to apply data augmentation (mirroring, rotating) or not
ONERA_PATCHES = True  # whether to train on patches sliced on the original Onera images or not
NORMALISE_IMGS = False  # z-standardizing on full-image basis, note: only implemented for online slicing!



dimWiseSlice = 2  # along each dimension, number of slices during test time (helps to conserve memory)

augm_str  = 'Augm' if DATA_AUG else 'noAugm'
save_path = os.path.join('../../results/siameseComplex/', f'type{TYPE}-seed{manualSeed}-epochs{N_EPOCHS}')

if not os.path.exists(save_path):
    os.makedirs(save_path)
    os.makedirs(os.path.join(save_path, 'checkpoints'))



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(net, trainLoader, trainDataset, validationDataset, n_epochs=N_EPOCHS, save=True):
    t = np.linspace(1, n_epochs, n_epochs)


    epochValidationLoss = 0 * t
    epochValidationAccuracy = 0 * t
    epochValidationChangeAccuracy = 0 * t
    epochValidationNochangeAccuracy = 0 * t
    epochValidationPrecision = 0 * t
    epochValidationRecall = 0 * t
    epochValidationFmeasure = 0 * t


    fm = 0
    bestFm = 0


    ## Defining the optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
    
#    optimizer = torch.optim.Adam(net.parameters(), weight_decay=1e-4)
#    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95)
    
    # train the network for a given number of epochs
    print('Length of train loader is:'+str(len(trainLoader)))
    for epoch_index in range(n_epochs):
        net.train()
        print('Epoch: ' + str(epoch_index + 1) + ' of ' + str(N_EPOCHS))

        totCount = 0
        totLoss = 0
        totAccurate = 0
        classCorrect = list(0. for i in range(2))
        classTotal = list(0. for i in range(2))

        # iterate over batch
        for batchStep,batch in enumerate(trainLoader):
            # print(batchStep)
            # inserted multi-modality here
            S2_1 = Variable(batch['time_1']['S2'].float().cuda())
            S2_2 = Variable(batch['time_2']['S2'].float().cuda())
            if TYPE in [4, 5]:
                S1_1 = Variable(batch['time_1']['S1'].float().cuda())
                S1_2 = Variable(batch['time_2']['S1'].float().cuda())
            label = torch.squeeze(Variable(batch['label'].cuda()))

            # get predictions, compute losses and optimize network
            optimizer.zero_grad()
            # outputs of the network are [dimWiseSlice x 2 x H x W]
            # label is of shape [32, 96, 96]
            if TYPE in [4, 5]:
                output = net(S2_1, S2_2, S1_1, S1_2)
            else:
                output = net(S2_1, S2_2)
            loss = criterion(output, label.long())
            loss.backward() # 
            optimizer.step()
#        scheduler.step()
            





        # evaluate network statistics on validation split and keep track
        epochValidationLoss[epoch_index], epochValidationAccuracy[epoch_index], cl_acc, pr_rec = test(validationDataset)
        epochValidationNochangeAccuracy[epoch_index] = cl_acc[0]
        epochValidationChangeAccuracy[epoch_index] = cl_acc[1]
        epochValidationPrecision[epoch_index] = pr_rec[0]
        epochValidationRecall[epoch_index] = pr_rec[1]
        epochValidationFmeasure[epoch_index] = pr_rec[2]




        fm = epochValidationFmeasure[epoch_index]
        print('F1 score on the validation data is: '+str(fm))
        if fm > bestFm:
            bestFm = fm
            saveStr = os.path.join(save_path, 'checkpoints', 'netBest.pth.tar')
            torch.save(net.state_dict(), saveStr)


           
    out = {'validationLoss': epochValidationLoss[-1],
       'validationAccuracy': epochValidationAccuracy[-1],
       'validationNochangeAccuracy': epochValidationNochangeAccuracy[-1],
       'validationChangeAccuracy': epochValidationChangeAccuracy[-1]}

    print('pr_c, rec_c, f_meas, pr_nc, rec_nc')
    print(pr_rec)

    return out

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
                ymin = jj
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
                loss         = criterion(output, cm.long())
                totLoss    += loss.data * np.prod(cm.size())
                totCount   += np.prod(cm.size())

                _, predicted = torch.max(output.data, 1)

                # compare predictions with change maps and count correct predictions
                c = (predicted.int() == cm.data.int())
                for i in range(c.size(1)):
                    for j in range(c.size(2)):
                        l = int(cm.data[0, i, j])
                        classCorrect[l] += c[0, i, j]
                        classTotal[l] += 1

                pr = (predicted.int() > 0).cpu().numpy()
                gt = (cm.data.int() > 0).cpu().numpy()

                # evaluate TP, TN, FP & FN
                tp += np.logical_and(pr, gt).sum()
                tn += np.logical_and(np.logical_not(pr), np.logical_not(gt)).sum()
                fp += np.logical_and(pr, np.logical_not(gt)).sum()
                fn += np.logical_and(np.logical_not(pr), gt).sum()


    # compute goodness of predictions
    netLoss = totLoss / totCount
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

    return netLoss, netAccuracy, classAccuracy, pr_rec



if __name__ == '__main__':


    ## training data and data loader
    trainingDataset = multiCD(PATH_TO_DATASET, split="train", transform=DATA_AUG, run_on_onera_patches=ONERA_PATCHES, normalize=NORMALISE_IMGS)
    trainLoader = torch.utils.data.DataLoader(trainingDataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_THREADS)
    
    
    ## validation data 
    validationDataset = multiCD(PATH_TO_DATASET, split="validation", run_on_onera_patches=ONERA_PATCHES, normalize=NORMALISE_IMGS)

    ## test data
    testDataset = multiCD(PATH_TO_DATASET, split="test", run_on_onera_patches=ONERA_PATCHES, normalize=NORMALISE_IMGS)
    


    # get train split weighting of pixel labels
    weights = torch.FloatTensor(trainingDataset.weights).cuda()
    print(f"Train data set weighting is: {weights}")
    print(f"Total pixel numbers: {trainingDataset.n_pix}")
    print(f"Changed pixel numbers: {trainingDataset.true_pix}")
    print(f"Change-to-total ratio: {trainingDataset.true_pix / trainingDataset.n_pix}")

    # 0-RGB | 1-RGBIr | 2-All bands s.t. resolution <= 20m | 3-All bands | 4-All bands Sentinel-2 & Sentinel-1 | 5-RGB bands Sentinel-2 & Sentinel-1

    if TYPE == 0:
        if NET == 'Unet': net, net_name = Unet(2*3, 2), 'FC-EF'
        if NET == 'SiamUnet_conc': net, net_name = SiamUnet_conc(3, 2), 'FC-Siam-conc'
        if NET == 'SiamUnet_diff': net, net_name = SiamUnet_diff(3, 2), 'FC-Siam-diff'
        if NET == 'FresUNet': net, net_name = FresUNet(2*3, 2), 'FresUNet'
    elif TYPE == 1:
        if NET == 'Unet': net, net_name = Unet(2*4, 2), 'FC-EF'
        if NET == 'SiamUnet_conc': net, net_name = SiamUnet_conc(4, 2), 'FC-Siam-conc'
        if NET == 'SiamUnet_diff': net, net_name = SiamUnet_diff(4, 2), 'FC-Siam-diff'
        if NET == 'FresUNet': net, net_name = FresUNet(2*4, 2), 'FresUNet'
    elif TYPE == 2:
        if NET == 'Unet': net, net_name = Unet(2*10, 2), 'FC-EF'
        if NET == 'SiamUnet_conc': net, net_name = SiamUnet_conc(10, 2), 'FC-Siam-conc'
        if NET == 'SiamUnet_diff': net, net_name = SiamUnet_diff(10, 2), 'FC-Siam-diff'
        if NET == 'FresUNet': net, net_name = FresUNet(2*10, 2), 'FresUNet'
    elif TYPE == 3:
        if NET == 'Unet': net, net_name = Unet(2*13, 2), 'FC-EF'
        if NET == 'SiamUnet_conc': net, net_name = SiamUnet_conc(13, 2), 'FC-Siam-conc'
        if NET == 'SiamUnet_diff': net, net_name = SiamUnet_diff(13, 2), 'FC-Siam-diff'
        if NET == 'FresUNet': net, net_name = FresUNet(2*13, 2), 'FresUNet'
    elif TYPE == 4:
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

    # define loss
    # criterion = nn.NLLLoss(weight=weights)
    criterion = nn.CrossEntropyLoss(weight=weights)
    print('Number of trainable parameters:', count_parameters(net))

    tStart = time.time()
    ## Train the network
    outDic = train(net, trainLoader, trainingDataset, validationDataset)
    tEnd = time.time()
    print('Training process completed')
    print('Elapsed time:')
    print(tEnd - tStart)
    
    
    ## After training, load the best model and test on it
    # train the network
    print('We now proceed for test')
    net.load_state_dict(torch.load(os.path.join(save_path, 'checkpoints', 'netBest.pth.tar')))
    print('Best model loaded for testing')
    netLossTest, netAccuracyTest, classAccuracyTest, prRecTest = test(testDataset)
    
    print('Net accuracy on test data is: '+str(netAccuracyTest))    

    print('Precision computed on test data is'+str(prRecTest[0]))
    print('Recall computed on test data is'+str(prRecTest[1]))
    print('F1 score computed on test data is'+str(prRecTest[2]))



