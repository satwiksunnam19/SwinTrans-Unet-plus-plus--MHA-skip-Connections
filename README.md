# SCUNet++ With MHA Skip Connections 

## 1. Download pre-trained swin transformer model (Swin-T)

* [Get pre-trained model in this link] (https://drive.google.com/drive/folders/1UC3XOoezeum0uck4KBVGa8osahs6rKUY?usp=sharing): Put pretrained Swin-T into folder "pretrained_ckpt/"

## 2. Environment

- Please prepare an environment with python=3.7, and then use the command "pip install -r requirements.txt" for the dependencies.

## 3. Train/Test

- Run lists/lists_Synapse/tool.py to Generate txt file
- Run train.py to Train (Put the dataset in npz format into datasets/Synapse/train_npz)
-  Run test.py to Test (Put the dataset in npz format into datasets/test)
- The batch size we used is 24. If you do not have enough GPU memory, the bacth size can be reduced to 12 or 6 to save memory.
