import os
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import time
from tqdm import tqdm
import threading

# Printing the os name
print("os name is: ", os.name)

if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    if input('mps or cpu: ') == 'mps':
        device = 'mps'
        print('device mps')
    else:
        device = 'cpu'
        print('device cpu')
elif torch.cuda.is_available():
    if input('cuda or cpu: ') == 'cuda':
        device = 'cuda'
        print('device cuda')
    else:
        device = 'cpu'
        print('device cpu')
else:
    device = 'cpu'
    print('device cpu')
    
# ANSI escape sequences for text colors
RESET = '\033[0m'
BLACK = '\033[30m'
RED = '\033[31m'
GREEN = '\033[32m'
YELLOW = '\033[33m'
BLUE = '\033[34m'
MAGENTA = '\033[35m'
CYAN = '\033[36m'
WHITE = '\033[37m'

# ANSI escape sequences for background colors
BG_BLACK = '\033[40m'
BG_RED = '\033[41m'
BG_GREEN = '\033[42m'
BG_YELLOW = '\033[43m'
BG_BLUE = '\033[44m'
BG_MAGENTA = '\033[45m'
BG_CYAN = '\033[46m'
BG_WHITE = '\033[47m'

# ANSI escape sequences for text styles
BOLD = '\033[1m'
UNDERLINE = '\033[4m'

training_data = './training_data/' + input(MAGENTA + 'training data path (include file extention): ' + RESET)
batch_size = eval(input(BLUE + 'batch size \'powers of 2\': ' + RESET)) # how many independent sequences will we process in parallel?
block_size = eval(input(BLUE + 'block size \'powers of 2\' (atleast 2 times batch size): ' + RESET)) # what is the maximum context length for predictions?
chunk_size = eval(input(BLUE + 'file chunk size \'powers of 2\' (atleast 2 times block size): ' + RESET))
file_stat = os.stat(training_data)
chars = []

# hyperparameters
max_iters = input(BLUE + 'max iterations \'multiple of 100\': ' + RESET)
chunks = math.ceil(file_stat.st_size/(chunk_size))
max_iters_chunk = int(max_iters)//chunks
eval_interval = 1
learning_rate = 1e-3
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.0
# ------------

# path for saving and loading models
model_path = './models/model.pt'

load_model = input(CYAN + "Would you like to load A pre-existing model? (yes(Y)/no(N)): " + RESET)
print('file size: ' + str(file_stat.st_size))
# print('reading ' + str(min(10000000, file_stat.st_size)) + ' bytes')

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
for i in range(0, file_stat.st_size, (chunk_size)):
    with open(training_data, 'r', encoding='utf-8', errors='ignore') as f:
        f.seek(i)
        text = f.read(min((chunk_size), file_stat.st_size - i))
        f.close()

    # here are all the unique characters that occur in this text

    for char in text:
        if char not in chars:
            chars.append(char)

    print('curently initialising chunk: ' + str(math.ceil(i/chunk_size)) + '/' + str(chunks) + ', bytes: ' + str(len(text)))

chars.sort()
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

class ThreadWithReturnValue(threading.Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        threading.Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None
    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)
    def join(self, *args):
        threading.Thread.join(self, *args)
        return self._return

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self, n_embd, n_head, n_layer):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

if load_model.lower() == 'y':
    pre_trained_path = 'models/' + str(input(MAGENTA + 'default(model.pt) specify model path: ' + RESET))
    model_path = pre_trained_path if os.path.isfile(pre_trained_path) else model_path
    checkpoint = torch.load(model_path)
    n_embd = checkpoint['n_embd']
    block_size = checkpoint['block_size']
    batch_size = checkpoint['batch_size']
    n_head = checkpoint['n_head']
    n_layer = checkpoint['n_layer']
    model = BigramLanguageModel(n_embd, n_head, n_layer)  # Pass parameters here
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    print("Model loaded.")
    for i in range(0, file_stat.st_size, (chunk_size)):
        with open(training_data, 'r', encoding='utf-8', errors='ignore') as f:
            f.seek(i)
            text = f.read(min((chunk_size), file_stat.st_size - i))
            f.close()

        # here are all the unique characters that occur in this text

        for char in text:
            if char not in chars:
                chars.append(char)

        print('curently loading chunk: ' + str(math.ceil(i/chunk_size)) + '/' + str(chunks) + ', bytes: ' + str(len(text)))

    chars.sort()
    vocab_size = len(chars)
    # create a mapping from characters to integers
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
    decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

    # Train and test splits
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9*len(data)) # first 90% will be train, rest val
    train_data = data[:n]
    val_data = data[n:]

else:
    print("A new model will be trained.")
    model = BigramLanguageModel(n_embd, n_head, n_layer)
    model = model.to(device)
    loss_thread = ThreadWithReturnValue(target=estimate_loss)
    loss_thread.start()
    for i in range(0, file_stat.st_size, (chunk_size)):
        with open(training_data, 'r', encoding='utf-8', errors='ignore') as f:
            f.seek(i)
            text = f.read(min((chunk_size), file_stat.st_size - i))
            f.close()

        if not len(text) < chunk_size:
            # here are all the unique characters that occur in this text
            print('curently loaded chunk: ' + str(math.ceil(i/chunk_size)) + '/' + str(chunks) + ', bytes: ' + str(len(text)))
            # print('chunks ' + str(math.ceil(i/chunk_size)) + '/' + str(chunks))

            for char in text:
                if char not in chars:
                    chars.append(char)

            chars.sort()
            vocab_size = len(chars)
            # create a mapping from characters to integers
            stoi = { ch:i for i,ch in enumerate(chars) }
            itos = { i:ch for i,ch in enumerate(chars) }
            encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
            decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

            # Train and test splits
            data = torch.tensor(encode(text), dtype=torch.long)
            n = int(0.9*len(data)) # first 90% will be train, rest val
            train_data = data[:n]
            val_data = data[n:]
            # model = BigramLanguageModel(n_embd, n_head, n_layer)
            # model = model.to(device)

            # create a PyTorch optimizer
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

            # Initialize the progress bar
            pbar = tqdm(total=max_iters_chunk, desc="Training progress")

            if not loss_thread.is_alive():
                losses = loss_thread.join()  # get the losses if the thread has finished
                # start a new thread for the next calculation
                loss_thread = ThreadWithReturnValue(target=estimate_loss)
                loss_thread.start()
    
            # start training
            for iter in range(max_iters_chunk):
                # get a batch of data
                xb, yb = get_batch('train')

                # compute the loss
                logits, loss = model(xb, yb)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                losses = {'train': 0.0, 'val': 0.0}

                # evaluate the loss on train and val sets
                if iter % eval_interval == 0 or iter == max_iters_chunk - 1:
                    if not loss_thread.is_alive():
                        losses = loss_thread.join()  # get the losses if the thread has finished
                        # start a new thread for the next calculation
                        loss_thread = ThreadWithReturnValue(target=estimate_loss)
                        loss_thread.start()
                        pbar.set_postfix({'train loss': f"{losses['train']:.4f}", 'val loss': f"{losses['val']:.4f}"})
            
                # Updating the progress bar
                pbar.update(1)

                # get a batch of data
                xb, yb = get_batch('train')

                # compute the loss
                logits, loss = model(xb, yb)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            # Closing the progress bar
            pbar.close()

        else:
            pass

    model_path = './models/' + str(input(MAGENTA + 'save model as: ' + RESET)) + '.pt'
    # save the model
    torch.save({
        'model_state_dict': model.state_dict(),
        'n_embd': n_embd,
        'n_head': n_head,
        'n_layer': n_layer,
        'block_size': block_size,
        'batch_size': batch_size
    }, model_path)

    # torch.save(model.state_dict(), model_path)
    print("Model saved.")


max_response = int(input(BLUE + 'response length(letters): ' + RESET))
user_prompt = '\n\nuser: \n' + str(input(MAGENTA + "Please enter a starting prompt: " + RESET))
gpt_prompt = ''
# generate from the model
while user_prompt != '\n\nuser: \n':
    gpt_prompt = gpt_prompt + '\n\n' + user_prompt + '\n'
    context = torch.tensor(encode((gpt_prompt.replace('\n\nuser: \n', ' ')).replace('\n\nAI: \n', ' ')), dtype=torch.long, device=device).unsqueeze(0) # add batch dimension
    response = (str(decode(model.generate(context, max_new_tokens=max_response)[0].tolist()))).replace(str(gpt_prompt.replace('\n\nuser: \n', ' ')).replace('\n\nAI: \n', ' '), '')
    gpt_prompt = gpt_prompt + '\n\nAI: \n' + str(response)
    if(os.name == 'posix'):
    # screen will clear for mac and linux
        os.system('clear')
    # else screen will be cleared for windows
    else:
        os.system('cls')
    print(str(gpt_prompt.replace('\n\nuser: \n', BOLD + RED + '\n\nuser: \n' + RESET)).replace('\n\nAI: \n', BOLD + GREEN + '\n\nAI: \n' + RESET))
    # print('\n' + gpt_prompt + '\n')
    user_prompt = '\n\nuser: \n' + str(input(MAGENTA + "Please enter a prompt: " + RESET))
exit()
