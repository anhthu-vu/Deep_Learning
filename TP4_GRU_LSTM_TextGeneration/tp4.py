import logging
logging.basicConfig(level=logging.INFO)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from textloader import *
from generate import *
from pathlib import Path
from datetime import datetime
import os
from tqdm import tqdm


# TODO
"""
The Trump dataset in "trump_full_speech.txt" file contains a set of Trump's pre-election speeches.

The following is my code for building and training text generation models using different types of RNN: a traditional RNN, a GRU and a LSTM.
It, from an input sequence of symbols (letters, punctuations, digits and space), aims to produce the next symbols of the sequence.
"""

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = "./data/trump_full_speech.txt"

def maskedCrossEntropy(output: torch.Tensor, target: torch.LongTensor, padcar: int):
    """
    In this tp, we haven't used the class PackedSequence of torch.nn.utils.rnn, so this functions is used to calculate the loss without taking null characters into account. 
    Parameters:
        output: tensor of shape length x batch x output_dim
        target: tensor of shape length x batch
        padcar: index of the padding character
    """
    
    mask = target != padcar
    output_dim = output.shape[-1]
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    masked_loss = loss_fn(output.reshape(-1, output_dim), target.reshape(-1)) * mask.reshape(-1).float()
    
    return masked_loss.sum()/mask.sum()


class RNN(nn.Module):
    """
    This is a simple RNN that I code from scratch.
    It takes as input a batch of shape sequence_length x batch x dim_input.
    """
    
    def __init__(self, vocab_size, embed_dim, dim_latent, dim_output):
        super(RNN, self).__init__()

        self.dim_input = embed_dim 
        self.dim_latent = dim_latent 
        self.dim_output = dim_output 
        self.embed = nn.Embedding(vocab_size, self.dim_input)
        self.encoder = nn.Linear(self.dim_input+self.dim_latent, self.dim_latent)
        self.activation = nn.Tanh()
        self.decoder = nn.Linear(self.dim_latent, self.dim_output)

    def embedding(self, x):
        """
        Embedding function
        Parameters:
            x: tensor of shape sequence_length x batch 
        """
        
        return self.embed(x)
        
    def one_step(self, x, h):
        """
        This function processes a timestep.
        Parameters: 
            x: input of that timestep t of shape batch x dim_input
            h: latent state of the previous timestep (t-1) of shape batch x dim_latent
        Return:
            Latent state of that timestep t of shape batch x dim_latent
        """
        
        return self.activation(self.encoder(torch.cat([x, h], dim=-1)))

    def forward(self, x, h_0=None):
        """
        Parameters: 
            x: input of RNN of shape sequence_length x batch x dim_input
            h_0: initial hidden state
        Return:
            hidden_states: sequence of hidden states of shape sequence_length x batch x dim_latent
        """
        
        length, batch = x.shape[0], x.shape[1] 
        
        if h_0 is None:
            h_0 = torch.zeros((batch, self.dim_latent), device=x.device) 
            
        h_t_minus_1 = h_0
        hidden_states = []

        for t in range(length):
            h_t = self.one_step(x[t], h_t_minus_1)
            h_t_minus_1 = h_t
            hidden_states.append(h_t)
            
        hidden_states = torch.stack(hidden_states)
        return hidden_states

    def decode(self, h):
        """
        This function decodes latent states to produce an output for downstream tasks.
        Parameters:
            h: latent states of shape sequence_length x batch x dim_latent
        Return:
            Output of the decoder of shape sequence_length x batch x dim_output
        """
        
        return self.decoder(h)


# Learning with RNN
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


def train_loop(data: DataLoader, state: State, grad_track=False):
    """
    This function is the training phase in an epoch.
    Parameters:
        data: training dataloader with a batch of shape batch x sequence_length 
        state: checkpointing of the training process
        grad_track: whether or not track the norms of the gradients w.r.t the network's parameters
    Return:
        average train loss in that epoch
    """
    
    total_loss = 0.
        
    model = state.model
    optimizer = state.optim
    
    if grad_track:
        grad_norm = {name: [] for name, _ in model.named_parameters()}
    
    for x in tqdm(data, desc=f"Epoch {state.epoch+1}"):
        optimizer.zero_grad()

        input_x = x[:-1, :]  # dim = length x batch
        target = x[1:, :] # dim = length x batch
        input_x, target = input_x.to(DEVICE), target.to(DEVICE)

        embed_x = model.embedding(input_x) # dim = length x batch x embedding_dim
        h = model.forward(embed_x) # dim = length x batch x latent_dim
        output = model.decode(h) # dim = length x batch x output_dim
        loss = maskedCrossEntropy(output, target, PAD_IX)

        loss.backward()
        
        if grad_track:
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm[name].append(torch.norm(param.grad))
                    
        optimizer.step()
        
        total_loss += loss.item()

        state.iteration += 1
        
    if grad_track:   
        return total_loss/len(data), grad_norm
    else:
        return total_loss/len(data)


def run(train_loader, model, nb_epochs, lr, name_model='', grad_track_interval=None):
    """
    This function is the training process.
    Parameters:
        grad_track_interval: track the norms of the gradients w.r.t the network's parameters every «grad_track_interval» epochs
    """
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    writer = SummaryWriter(f'runs/Trump_generation_{name_model}_{timestamp}')
    
    savepath = Path(f"Trump_generation_{name_model}.pth")
    state = State.load(savepath)
    if state.model is None:
        state.model =  model.to(DEVICE)
        state.optim = torch.optim.Adam(state.model.parameters(), lr=lr)
    
    for epoch in range(state.epoch, nb_epochs):
        if grad_track_interval is not None and epoch % grad_track_interval == 0: 
                loss_train, grad_norms = train_loop(train_loader, state, grad_track=True)
                for name, grad_norm in grad_norms.items():
                    writer.add_histogram(name, torch.stack(grad_norm), epoch)
        else:
            loss_train = train_loop(train_loader, state, grad_track=False)
        
        writer.add_scalar('Loss_train', loss_train, state.epoch)
        
        print (f'Train loss: {loss_train} ')

        state.epoch = epoch + 1
        state.save()

BATCH_SIZE = 32
data_trump = DataLoader(TextDataset(open(DATA_PATH ,"rb").read().decode(), maxlen=1000), batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_collate_fn)

NB_EPOCHS = 150
EMBED_DIM = 60
DIM_LATENT = 80
LEARNING_RATE = 1e-4
model = RNN(len(lettre2id), EMBED_DIM, DIM_LATENT, len(lettre2id))
run(data_trump, model, NB_EPOCHS, LEARNING_RATE, 'RNN')

state = State.load(Path(f"Trump_generation_RNN.pth"))
output_seq_determinist = generate(state.model, EOS_IX, start="", maxlen=50, random=False)
print ('Output with deterministic generation:\n' + output_seq_determinist + '\n')

output_seq_random = generate(state.model, EOS_IX, start="", maxlen=50, random=True)
print ('Output with random generation:\n' + output_seq_random + '\n')

output_seq_beam = generate_beam(state.model, EOS_IX, k=3, start="")
print ('Output with beam search generation:\n' + output_seq_beam[0] + '\n')

output_seq_beam_nucleus = generate_beam_with_p_nucleus(state.model, EOS_IX, alpha=0.95, k=3, start="", maxlen=50)
print ('Output with nucleus - beam search generation:\n' + output_seq_beam_nucleus[0] + '\n')


class LSTM(RNN):
    """
    This is a simple LSTM that I code from scratch.
    It takes as input a batch of shape sequence_length x batch x dim_input.
    """
    
    def __init__(self, vocab_size, embed_dim, dim_latent, dim_output):
        super(LSTM, self).__init__(vocab_size, embed_dim, dim_latent, dim_output)
        
        self.forget = nn.Linear(self.dim_input+self.dim_latent, self.dim_latent) # gate f_t
        self.enter = nn.Linear(self.dim_input+self.dim_latent, self.dim_latent) # gate i_t
        self.out = nn.Linear(self.dim_input+self.dim_latent, self.dim_latent) # gate o_t 
        self.gate_activation = nn.Sigmoid()
        
    def one_step(self, x, h, c):
        """
        This function processes a timestep.
        Parameters: 
            x: input of that timestep t of shape batch x dim_input
            h: latent state of the previous timestep (t-1) of shape batch x dim_latent
            c: internal state of the previous timestep (t-1) of shape batch x dim_latent
        Return:
            Latent state of that timestep t of shape batch x dim_latent
        """
        
        concat_h_x = torch.cat([h, x], dim=-1)
        f_t = self.gate_activation(self.forget(concat_h_x))
        i_t = self.gate_activation(self.enter(concat_h_x))
        o_t = self.gate_activation(self.out(concat_h_x))
        c_t = f_t*c + i_t*self.activation(self.encoder(concat_h_x))
        
        return c_t, o_t*self.activation(c_t)

    def forward(self, x, h_0=None):
        length, batch = x.shape[0], x.shape[1] 
        c_t_minus_1 = torch.zeros((batch, self.dim_latent), device=x.device) 
        
        if h_0 is None:
            h_0 = torch.zeros((batch, self.dim_latent), device=x.device) 
            
        h_t_minus_1 = h_0
        hidden_states = []

        for t in range(length):
            c_t, h_t = self.one_step(x[t], h_t_minus_1, c_t_minus_1)
            h_t_minus_1 = h_t
            c_t_minus_1 = c_t
            hidden_states.append(h_t)
            
        hidden_states = torch.stack(hidden_states)
        return hidden_states

# Learning with LSTM
NB_EPOCHS = 50
model = LSTM(len(lettre2id), EMBED_DIM, DIM_LATENT, len(lettre2id))
run(data_trump, model, NB_EPOCHS, LEARNING_RATE, 'LSTM', grad_track_interval=5)


class GRU(RNN):
    """
    This is a simple GRU that I code from scratch.
    It takes as input a batch of shape sequence_length x batch x dim_input.
    """
    
    def __init__(self, vocab_size, embed_dim, dim_latent, dim_output):
        super(GRU, self).__init__(vocab_size, embed_dim, dim_latent, dim_output)
        
        self.reset = nn.Linear(self.dim_input+self.dim_latent, self.dim_latent) # gate r_t
        self.update = nn.Linear(self.dim_input+self.dim_latent, self.dim_latent) # gate (1-z_t)
        self.gate_activation = nn.Sigmoid()
        
    def one_step(self, x, h):
        """
        This function processes a timestep.
        Parameters: 
            x: input of that timestep t of shape batch x dim_input
            h: latent state of the previous timestep (t-1) of shape batch x dim_latent
        Return:
            Latent state of that timestep t of shape batch x dim_latent
        """
        
        concat_h_x = torch.cat([h, x], dim=-1)
        r_t = self.gate_activation(self.reset(concat_h_x))
        z_t = self.gate_activation(self.update(concat_h_x))
        
        return (1-z_t)*h + z_t*self.activation(self.encoder(torch.cat([h*r_t, x], dim=-1)))

# Learning with GRU
NB_EPOCHS = 50
model = GRU(len(lettre2id), EMBED_DIM, DIM_LATENT, len(lettre2id))
run(data_trump, model, NB_EPOCHS, LEARNING_RATE, 'GRU', grad_track_interval=5)

# END TODO
