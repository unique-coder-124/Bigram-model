# My version of GPT Bigram

## Credit
#### Original Code:
> Andrej Karpathy

#### Github project: 
> https://github.com/karpathy/ng-video-lecture/tree/master

## Overview
This is my own recreation of the gpt model by andrej karpathy made to be more user friendly. This is currently only lightly modefied but in the future I intend to replace the tokenizer with my own custom word level tokenizer and a fully functional gui. I also intend to translate this to a faster programming language such as __c__ or __c#__. 

## Installations
You should first install an interpreter or python environment.

(The following instructions will assume you have already installed anaconda)

#### __Anaconda__
Create the virtual environment. 
```bash
conda create --name <your_environment_name> python=3.10
```

Activate the virtual environment. 
```bash
conda activate <your_environment_name>
```
<br>
<br>

Once the environment is created you can install all the required packages.
> This cannot be done through a requirements.txt as __cuda__ and __mps__ support is unavailable in the base install of pytorch.

#### __'Mac' torch download__
```bash
conda install pytorch-nightly::pytorch torchvision torchaudio -c pytorch-nightly
```

#### __'Windows' and 'Linux' torch download__
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch-nightly -c nvidia
```
> refer to https://pytorch.org/get-started/locally/ for your specific device
#### __Installing all other modules__
> works on 'Mac', 'Windows' and 'Linux'
```bash
conda install tqdm
```
#### __For use of the gui script__
```bash
conda install pysimplegui
```

## How to use
### Full CLI
#### Base code
```bash
python gpt-main.py
```
#### Loading Training data in blocks to save on memory
```bash
python gpt-file-loader-experimental.py
```

### Partial GUI
```bash
python gpt_pysimplegui
```
