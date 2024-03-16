# SCUNet++ with MHA Skip Connections

## 1. Environment

- Please prepare an environment with python=3.7, and then use the command "pip install -r requirements.txt" for the dependencies.

## 2. Train/Test

- Run lists/lists_Synapse/tool.py to Generate txt file
- Run train.py to Train (Put the dataset in npz format into datasets/Synapse/train_npz)
-  Run test.py to Test (Put the dataset in npz format into datasets/test)
- The batch size we are going to use is 24. If you do not have enough GPU memory, the batch size can be reduced to 12 or 6 to save memory.
