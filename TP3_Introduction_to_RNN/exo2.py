import logging
logging.basicConfig(level=logging.INFO)

from utils import RNN, DEVICE, SampleMetroDataset
import torch
from torch import nn
from torch.utils.data import DataLoader
from pathlib import Path
import os
from tqdm import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


# Nombre de stations utilisé
CLASSES = 10
#Longueur des séquences
LENGTH = 20
# Dimension de l'entrée (1 (in) ou 2 (in/out))
DIM_INPUT = 2
#Taille du batch
BATCH_SIZE = 32

PATH = "./data/"


matrix_train, matrix_test = torch.load(open(PATH+"hzdataset.pch","rb"))
ds_train = SampleMetroDataset(matrix_train[:, :, :CLASSES, :DIM_INPUT], length=LENGTH)
ds_test = SampleMetroDataset(matrix_test[:, :, :CLASSES, :DIM_INPUT], length = LENGTH, stations_max = ds_train.stations_max)
data_train = DataLoader(ds_train,batch_size=BATCH_SIZE,shuffle=True)
data_test = DataLoader(ds_test, batch_size=BATCH_SIZE,shuffle=False)


# TODO: 
"""
A quick overview of the Hangzhou dataset contained in "hzdataset.pch" file:
    - It describes the incoming and outgoing flows of 80 stations in Hangzhou aggregated by quarter-hour between 5:30 AM and 11:30 PM each day.
    - It contains two tensors: one for training and one for testing. They are of size D×T×S×2 with D the number of days, T = 73 the successive quarter-hour slices between 5:30 AM and 11:30 PM, S = 80 the number of stations and the incoming and outgoing flows for the last dimension.
    
The following is my code for building and training a classification model (according to question 2 of tp sujet).
It, from the incoming and outgoing flows of 20 successive quarter-hour slices (this can be modified by variable LENGTH above), aims to decide which station the sequence belongs to among 10 stations (this can be modified by variable CLASSES above). 
"""

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
        data: training dataloader with a batch of shape batch x sequence_length x dim_input 
        state: checkpointing of the training process
    Return:
        average train loss and train accuracy in that epoch
    """
    
    total_loss, total_acc  = 0., 0.
    model = state.model
    optimizer = state.optim
    
    for x, y in tqdm(data, desc=f"Epoch {state.epoch+1}"):
        optimizer.zero_grad()
        
        x = x.transpose(0, 1)
        x, y = x.to(DEVICE), y.to(DEVICE)
        final_h = model.forward(x)[-1]
        y_decode = model.decode(final_h)
        loss = loss_fn(y_decode, y)

        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_acc += (torch.argmax(y_decode, dim=-1) == y).float().mean().item() 

        state.iteration += 1
        
    return total_loss/len(data), total_acc/len(data)


def test_loop(data: DataLoader, state: State, loss_fn):
    """
    This function is the testing phase in an epoch.
    Parameters:
        data: test dataloader with a batch of shape batch x sequence_length x dim_input 
        state: checkpointing of the training process
    Return:
        average test loss and test accuracy in that epoch
    """
    
    total_loss, total_acc  = 0., 0.
    model = state.model
    
    with torch.no_grad():
        for x, y in data:
            x = x.transpose(0, 1)
            x, y = x.to(DEVICE), y.to(DEVICE)
            final_h = model.forward(x)[-1]
            y_decode = model.decode(final_h)
    
            total_loss += loss_fn(y_decode, y).item()
            total_acc += (torch.argmax(y_decode, dim=-1) == y).float().mean().item() 
        
    return total_loss/len(data), total_acc/len(data)


def run(train_loader, test_loader, model, nb_epochs, lr):
    """
    This function is the training process.
    """
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    writer = SummaryWriter(f'runs/Hangzhouclassif_{timestamp}')
    
    savepath = Path("Hangzhouclassif.pth")
    state = State.load(savepath)
    if state.model is None:
        state.model =  model.to(DEVICE)
        state.optim = torch.optim.Adam(state.model.parameters(), lr=lr)

    loss_fn = nn.CrossEntropyLoss()
    
    for epoch in range(state.epoch, nb_epochs):
        loss_train, acc_train = train_loop(train_loader, state, loss_fn)
        loss_test, acc_test = test_loop(test_loader, state, loss_fn)

        writer.add_scalar('Loss_train', loss_train, state.epoch)
        writer.add_scalar('Acc_train', acc_train, state.epoch)
        writer.add_scalar('Loss_test', loss_test, state.epoch)
        writer.add_scalar('Acc_test', acc_test, state.epoch)

        print (f'Train loss: {loss_train} \t Train accuracy: {acc_train}')
        print (f'Test loss: {loss_test} \t Test accuracy: {acc_test}')

        state.epoch = epoch + 1
        state.save()

NB_EPOCHS = 100
DIM_LATENT = 8
LEARNING_RATE = 1e-4
model = RNN(DIM_INPUT, DIM_LATENT, CLASSES)
run(data_train, data_test, model, NB_EPOCHS, LEARNING_RATE)
# END TODO
