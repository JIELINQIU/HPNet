# HPNet

This is an implementation of HPNet, a computational hierarchical network model for spatiotemporal sequence learning effects observed in the primate visual cortex. 

## Setup

Python libraries: pytorch==0.4.1, numpy>=1.15. 

Tested environment: cuda (>=8.0) and cudnn (>=6.0).

## Datasets and Data Processing

The KTH data can be downloaded here: [KTH](http://www.nada.kth.se/cvap/actions/). The script 'download.sh' can be used to download all the raw data. 

Video frames were extracted from the raw videos, where the script 'prepare_data.py' can be used to extract video frames from the raw video sequences. 

The extracted video frames need to be processed and encoded, where the script 'process_data.py' was used, and split the data for the training process,  the outputs of the script were used as inputs in the training process, in the .hkl data format.

The script 'hkl2pkl.py' can be used to provide the .pkl format data. 

After the above process, the whole training data was created, including X_train, X_test, X_val, along with their sources, twelve data files in total, the data is under the '/KTH_data' folder in the provided link. 


## Training and Testing

The model was trained using the script ' DATA_train.py' by simply using command 'python DATA_train.py'. The version of pytorch tested is 0.4.1, the experimental environment is the autonlab cluster with Nvidia GeForce RTX. 

The training weights are saved in the '/models' folder, in the .pt format. After training, the script 'DATA_test.py' was used to generate the prediction results, by the command ' python DATA_test.py', and the outputs are two images named 'Origin' and 'Predicted', which are the original frames and the predictions. 

## Analysis

There are two quantitative index: Mean-Squared Error (MSE) and the Structural Similarity Index Measure (SSIM). The values of SSIM range from -1 to 1, with a larger value indicating greater similarity between the predicted frames and the original frames. The script 'compare.py' was used to compute the SSIM and MSE index. 
