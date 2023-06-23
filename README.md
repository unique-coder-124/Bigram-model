# My version of Bigram model

## Credit
#### Original Code:
> Andrej Karpathy

#### Github project: 
> https://github.com/karpathy/ng-video-lecture/tree/master

## Table of contents
  * [Overview](#overview)
  * [Installations](#installations)
      - [Anaconda](#anaconda)
      - [Setup](#setup)
  * [How to use](#how-to-use)
    + [Full CLI](#full-cli)
    + [GUI](#gui)
    + [Parameters](#parameters)

## Overview
This is my own recreation of the gpt model by andrej karpathy made to be more user friendly. This is currently only lightly modefied but in the future I intend to replace the tokenizer with my own custom word level tokenizer and a fully functional gui. I also intend to translate this to a faster programming language such as __c__ or __c#__. 

## Installations
You should first install an interpreter or python environment.

(The following instructions will assume you have already installed anaconda)

#### Anaconda
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

#### Setup
```bash
python setup.py
```

## How to use
> When loading a pretrained model ensure you set the training data to be the same as the one initially used to train the model. 
> 
> 
> This is temorary as the characters must be loaded the same as when trained. I will be including code that includes the characters as part of the model in the future but for now this is the solution.


> When selecting a training file, the path is relative to the training_data directory. __YOU CANNOT USE ABSOLUTE PATHS__
> 
> 
> When selecting a model, the path is relative to the models directory. __YOU CANNOT USE ABSOLUTE PATHS__

### Full CLI
#### Base code
> Good base code used as the template for everything else. 
```bash
python gpt-main.py
```
#### Character Model
> The dictionary of the model is stored in the model file with this code. 
```bash
python gpt-char.py
```
#### Chunk loading training data (experimental branch)
> This loads the training file in chunks to save on memory usage and avoid crashing. the chunk size can be specified. 
>
> You can use Python expressions in this code.
>
> e.g. Chunk Size: 2**20
> e.g. Block Size: 2+7-1
```bash
python gpt-file-loader-experimental.py
```

#### Gradient checkpoint (experimental branch)
> This loads the training using gradient checkpoints to save memory. 
```bash
python gpt-char-exp.py
```

### GUI
#### Partial GUI
> some inputs replaced with text boxes using pysimplegui
```bash
python gpt_pysimplegui
```

### Parameters
> Batch size: number of samples processed before the model is updated
>
> Block size: the context that the model understands in tokens (1 token = 1 letter/character)
>
> max training iters: the number of training cycles
>
> max tokens: max response length (1 token = 1 letter/character)
>
> prompt: leave empty to exit script (prompt does not provide differentiation between the user and ai due to the simplicity of the ai this would reduce the quality of the results. there is a new line printed after the response so the ai will always responds in a separate paragraph.)
