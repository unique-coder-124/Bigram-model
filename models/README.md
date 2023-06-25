# MODELS
__The only setting that has to be the same when loading a model is 'training data'.__ 

~~This will be changed in future iterations.~~
This is not the case for newer models as they can be loaded with gpt-char.py. the char models are not compatible with the main script. 

models that can be used without specifying training data will be labelled 'char model'. 

### model.pt
> Block size: 64
> 
> Batch size: 128
>
> Training iters: 10000
>
> Training data: input.txt __(the entire works of shakespear)__
>
> main model

### model_char.pt
> Block size: 128
> 
> Batch size: 256
>
> Training iters: 100000
>
> Training data: input.txt __(the entire works of shakespear)__
>
> char model

### 05losschess.pt
> Block size: 128
> 
> Batch size: 256
>
> Training iters: 5000000
>
> Training data: DATABASE4U.pgn __(9 million chess games. Download url in training_data directory)__
>
> char model
