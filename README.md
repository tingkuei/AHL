# AHL
How to use the code

1. Extracted the features from the last layer of C3D+ConvLSTM. (The SPP layer)
   Please follow the setting in https://github.com/GuangmingZhu/Conv3D_CLSTM. 
   Run 'python extract_train_features.py & python extract_valid_features.py' to extract features. 

2. We provide trained models for RGB and Depth stream. Please put it in the appropriate path accordingly. 

3. Run 'python train_and_valid.py ' for the final tinueing of the network. 
   The test accurancy will be converged at 54.50%. 

4.We will release the final trained model soon.

