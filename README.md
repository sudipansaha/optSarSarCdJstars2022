# optSarSarCdJstars2022

## Supervised Change Detection using Pre-Change Optical-SAR and Post-Change SAR Data

This work proposes a novel change detection data setting which uses both optical and SAR images pre-change, yet only SAR imagery post-change. For this challenging scenario, we propose a Siamese network that processes the pre-change and post-change SAR inputs using a shared set of weights, while the pre-change optical input is processed using a network that do not share the weights with the SAR inputs.

**Before running the code**, download the OSCD multi-sensor dataset as instructed in the paper https://www.int-arch-photogramm-remote-sens-spatial-inf-sci.net/XLIII-B3-2021/243/2021/isprs-archives-XLIII-B3-2021-243-2021.pdf  (dataset link is provided in Section 3.1)

**For training model**: $ python -u mainAllChannelsSiameseComplex.py --manualSeed 19   (manusal seed can be changed) <br/>
Note that path to the OSCD multi-sensor dataset is set in line 48 and the path where the trained model will be stored is set in line 63. Please change them as appropriate.

**After training**, to compute statistics relevant for test-time adaptation,  <br/>
$ python computeStatisticsFromTrainingImages.py <br/>
Set the path to the dataset in line 31 and path to where the folder where model is stored in 46

**To evaluate scores**, $ python evaluateScores.py <br/>
Set the path to the dataset in line 32  <br/>
Set the statistics for test-time adaptation computed in previous step in line 42 <br/>
Set where the model is stored in line 48
