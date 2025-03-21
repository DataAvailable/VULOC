## About VULOC
VULOC is a vulnerability scanning tool for C/C++ source code. It uses a customized deep learning architecture, combined with high-level abstract features of vulnerability source code and low-level fine-grained features of assembly code, to detect vulnerable functions and accurately locate vulnerable lines.

| Method     | Precision  | F1         | Re         | VL         |
| ---------- | ---------- | ---------- | ---------- | ---------- |
| RATS       | 68.9%      | 69.9%      | 71.0%      | 7.5        |
| Flawfinder | 75.1%      | 76.7%      | 78.3%      | 5.2        |
| VUDDY      | 70.1%      | 72.8%      | 75.7%      | 10.4       |
| BovdGFE    | 93.1%      | 88.8%      | 90.0%      | 21.5       |
| SySeVR     | 90.8%      | 92.6%      | 94.5%      | 17.1       |
| HAN-BSVD   | 97.9%      | 97.1%      | 96.4%      | 8.5        |
| VULOC      | 97.4%      | 97.7%      | 98.0%      | 1.0        |

## How to replicate
### Requirements
```
python = 3.8
torch = 2.6.0
nltk = 3.9.1
matplotlib = 3.5.1
scikit-learn = 1.0.2
gcc = 9.4.0
gdb = 8.1.1 # no python packages
```
If you simply want to replicate the experimental results, you can skip the preprocessing steps (including compilation, disassembly, slicing, and tokenization) and directly execute the following commands for model training and vulnerability detection.
```shell
sudo apt install unzip
cd dataset
unzip dataset.zip
cd ..
python train.py
```
If you want to gain a deeper understanding of VULOC or build your own dataset, you can follow the detailed feature engineering and preprocessing steps from scratch by executing the commands step by step as shown below.
### Step 1: Compile And Disassemble
1.Download the initial dataset.
```shell
cd testcases
wget https://samate.nist.gov/SARD/downloads/test-suites/2017-10-01-juliet-test-suite-for-c-cplusplus-v1-3.zip
unzip 2017-10-01-juliet-test-suite-for-c-cplusplus-v1-3.zip
mv C/testcases/* ./
rm -rf ./C
cd ../testcasesupport
unzip testcasesupport.zip
cd ..
```

2.Compile the code using GCC.
```shell
python compile.py
```
After compilation is complete, the binary files corresponding to each sample will be available in the BinaryFile directory.

3.Diassemble the object files.
```shell
gdb -q -x disassemble_addr2line.py
```
After disassembly is completed, the disassembly files bad_function_assembly.txt and good_function_assembly.txt will be saved to the assemble directory.
## Step 2: Slice and Label
1.Slice and label the generated assembly code, then split the samples into train.txt and test.txt files in an 8:2 ratio. 
```shell
python label_dataset.py
```
2.Remove samples with assembly instruction lines fewer than 10 or more than 120, and regenerate the train.txt and test.txt files, saving them to the dataset directory.
```shell
python del_long_short.py
```
## Step 3: Train and Detect
After completing the steps above, you can proceed to train the model. Here, we use the following default hyperparameters: dropout=0.2, batch_size=8, epochs=100, learning_rate=0.001, hidden_dim=512, and n_layers=2.
```shell
python train.py
```
