import logging
logging.basicConfig(level=logging.INFO)

import string
import unicodedata
import torch
import sys
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import os
from tqdm import tqdm
from datetime import datetime

from utils import RNN, DEVICE


## Liste des symboles autorisés
LETTRES = string.ascii_letters + string.punctuation+string.digits+' '
## Dictionnaire index -> lettre
id2lettre = dict(zip(range(1,len(LETTRES)+1),LETTRES))
id2lettre[0]='' ##NULL CHARACTER
## Dictionnaire lettre -> index
lettre2id = dict(zip(id2lettre.values(),id2lettre.keys()))

def normalize(s):
    """ Nettoyage d'une chaîne de caractères. """
    return ''.join(c for c in unicodedata.normalize('NFD', s) if  c in LETTRES)

def string2code(s):
    """ Transformation d'une chaîne de caractère en tenseur d'indexes """
    return torch.tensor([lettre2id[c] for c in normalize(s)])

def code2string(t):
    """ Transformation d'une liste d'indexes en chaîne de caractères """
    if type(t) !=list:
        t = t.tolist()
    return ''.join(id2lettre[i] for i in t)

class TrumpDataset(Dataset):
    def __init__(self,text,maxsent=None,maxlen=None):
        """  Dataset pour les tweets de Trump
            * text : texte brut
            * maxsent : nombre maximum de phrases.
            * maxlen : longueur maximale des phrases.
        """
        maxlen = maxlen or sys.maxsize
        full_text = normalize(text)
        self.phrases = [p[:maxlen].strip()+"." for p in full_text.split(".") if len(p)>0]
        if maxsent is not None:
            self.phrases=self.phrases[:maxsent]
        self.MAX_LEN = max([len(p) for p in self.phrases])

    def __len__(self):
        return len(self.phrases)
    def __getitem__(self,i):
        t = string2code(self.phrases[i])
        t = torch.cat([torch.zeros(self.MAX_LEN-t.size(0),dtype=torch.long),t])
        return t[:-1],t[1:]

BATCH_SIZE = 32
PATH = "./data/"
data_trump = DataLoader(TrumpDataset(open(PATH+"trump_full_speech.txt","rb").read().decode(),maxlen=1000), batch_size=BATCH_SIZE, shuffle=True)


#  TODO
"""
The Trump dataset in "trump_full_speech.txt" file contains a set of Trump's pre-election speeches.

The following is my code for building and training a generation model (according to question 4 of tp sujet).
It, from an input sequence of symbols (letters, punctuations and digits), aims to produce the next symbols of the sequence.
I would like to notify that the following generation model is restricted to the generation of fixed-size sequences that will be determined in advance.
"""

class embedding(nn.Module):
    """
    This is an embedding that I code by hand.
    """
    def __init__(self, vocab_size, embed_dim):
        super(embedding, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_matrix = nn.Linear(vocab_size, embed_dim)
    
    def forward(self, x):
        """
        Parameters:
            x: a tensor of shape batch x sequence_length
        Return:
            Embeddings of x of shape batch x sequence_length x embed_dim
        """
        one_hot = nn.functional.one_hot(x, num_classes=self.vocab_size).float()
        embeddings = self.embedding_matrix(one_hot)
        
        return self.embedding_matrix(one_hot)


class RNN_generation(RNN):
    def __init__(self, vocab_size, embed_dim, dim_latent, dim_output):
        super(RNN_generation, self).__init__(embed_dim, dim_latent, dim_output)
        self.embedding = embedding(vocab_size, embed_dim)


def generation(rnn, start="", length=200):
    """
    This function generates, either from some symbols or from an empty initial state, a sequence of fixed length. 
    Paramters:
        start: start of the sequence to be generated
        length: length of the sequence to be generated
    Return:
        The sequence to be generated
    """
    
    if len(start) > 0:  
        input_seq = string2code(start).unsqueeze(-1).to(DEVICE) # dim = length x batch(1)
    else:
        input_seq = torch.zeros((1, 1), device=DEVICE, dtype=torch.long) # dim = length(1) x batch(1)

    output_seq = ''

    for _ in range(length):
        
        embedded_input = rnn.embedding(input_seq) # dim = length x batch x embedding_dim
        hidden = rnn(embedded_input)[-1] # dim = batch x latent_dim
        output = decoder(hidden) # dim = batch x output_dim

        next_token = torch.argmax(output, dim=-1).item()
        output_seq += id2lettre[next_token]

        input_seq = torch.tensor([next_token], device=DEVICE).reshape(1, 1)
    
    return output_seq

    
class State:
    """
    This class serves for checkpointing purpose.
    """
    def __init__(self, path: Path, model, optim):
        self.path = path
        self.model = model
        self.optim = optim
        self.epoch = 0
        self.iteration = 0

    @staticmethod
    def load(path: Path):
        if path.is_file():
            with path.open("rb") as fp:
                state = torch.load(fp, map_location=DEVICE)
                logging.info("Starting back from epoch %d", state.epoch)
                return state
        return State(path, None, None)

    def save(self):
        savepath_tmp = self.path.parent / ("%s.tmp" % self.path.name)
        with savepath_tmp.open("wb") as fp:
            torch.save(self, fp)
        os.rename(savepath_tmp, self.path) 


def train_loop(data: DataLoader, state: State, loss_fn):
    """
    This function is the training phase in an epoch.
    Parameters:
        data: training dataloader with a batch of shape batch x sequence_length 
        state: checkpointing of the training process
    Return:
        average train loss in that epoch
    """
    
    total_loss = 0.
    model = state.model
    optimizer = state.optim
    
    for x, y in tqdm(data, desc=f"Epoch {state.epoch+1}"):
        optimizer.zero_grad()

        x, y = x.to(DEVICE), y.to(DEVICE)
        x = model.embedding(x)
        x = x.transpose(0, 1)
        h = model.forward(x)
        y_decode = model.decode(h)
        loss = loss_fn(y_decode.reshape(-1, model.dim_output), y.reshape(-1))

        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

        state.iteration += 1
        
    return total_loss/len(data)


def run(train_loader, model, nb_epochs, lr):
    """
    This function is the training process.
    """
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    writer = SummaryWriter(f'runs/Trump_generation_{timestamp}')
    
    savepath = Path("Trump_generation.pth")
    state = State.load(savepath)
    if state.model is None:
        state.model =  model.to(DEVICE)
        state.optim = torch.optim.Adam(state.model.parameters(), lr=lr)

    loss_fn = nn.CrossEntropyLoss()
    
    for epoch in range(state.epoch, nb_epochs):
        loss_train = train_loop(train_loader, state, loss_fn)

        writer.add_scalar('Loss_train', loss_train, state.epoch)

        print (f'Train loss: {loss_train} ')

        state.epoch = epoch + 1
        state.save()

NB_EPOCHS = 10
EMBED_DIM = 50
DIM_LATENT = 70
LEARNING_RATE = 1e-4
model = RNN_generation(len(LETTRES)+1, EMBED_DIM, DIM_LATENT, len(LETTRES)+1)
run(data_trump, model, NB_EPOCHS, LEARNING_RATE)
# END TODO
