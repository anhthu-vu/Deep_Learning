import logging
logging.basicConfig(level=logging.INFO)

from utils import RNN, DEVICE,  ForecastMetroDataset
from torch.utils.data import  DataLoader
import torch
from torch import nn
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


matrix_train, matrix_test = torch.load(open(PATH+"hzdataset.pch", "rb"))
ds_train = ForecastMetroDataset(
    matrix_train[:, :, :CLASSES, :DIM_INPUT], length=LENGTH)
ds_test = ForecastMetroDataset(
    matrix_test[:, :, :CLASSES, :DIM_INPUT], length=LENGTH, stations_max=ds_train.stations_max)
data_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
data_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False)

#  TODO
"""
A quick overview of the Hangzhou dataset contained in "hzdataset.pch" file:
    - It describes the incoming and outgoing flows of 80 stations in Hangzhou aggregated by quarter-hour between 5:30 AM and 11:30 PM each day.
    - It contains two tensors: one for training and one for testing. They are of size D×T×S×2 with D the number of days, T = 73 the successive quarter-hour slices between 5:30 AM and 11:30 PM, S = 80 the number of stations and the incoming and outgoing flows for the last dimension.
    
The following is my code for building and training a forecasting model (according to question 3 of tp sujet).
It, from the incoming and outgoing flows of t successive quarter-hour slices, aims to predict the incoming and outgoing flows at time t+1.
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
        data: training dataloader with a batch of shape batch x sequence_length x dim_input (dim_input = nb_stations x 2)
        state: checkpointing of the training process
    Return:
        average train loss in that epoch
    """
    
    total_loss = 0.
    model = state.model
    optimizer = state.optim
    
    for x, y in tqdm(data, desc=f"Epoch {state.epoch+1}"):
        optimizer.zero_grad()

        batch, nb_stations = x.shape[0], x.shape[2]
        x = x.transpose(0, 1) 
        x, y = x.to(DEVICE), y.to(DEVICE)
        h_0 = torch.zeros((batch, nb_stations, model.dim_latent), device=x.device)
        h = model.forward(x, h_0)
        y_decode = model.decode(h).transpose(0, 1)
        # dim_y = batch x sequence_length x nb_stations x 2
        loss = loss_fn(y_decode.reshape(batch, -1), y.reshape(batch, -1)) 

        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

        state.iteration += 1
        
    return total_loss/len(data)


def test_loop(data: DataLoader, state: State, loss_fn):
    """
    This function is the testing phase in an epoch.
    Parameters:
        data: test dataloader with a batch of shape batch x sequence_length x dim_input (dim_input = nb_stations x 2)
        state: checkpointing of the training process
    Return:
        average test loss in that epoch
    """
    
    total_loss = 0.
    model = state.model
    
    with torch.no_grad():
        for x, y in data:
            batch, nb_stations = x.shape[0], x.shape[2]
            x = x.transpose(0, 1)
            x, y = x.to(DEVICE), y.to(DEVICE)
            h_0 = torch.zeros((batch, nb_stations, model.dim_latent), device=x.device)
            h = model.forward(x, h_0)
            y_decode = model.decode(h).transpose(0, 1)
    
            total_loss += loss_fn(y_decode.reshape(batch, -1), y.reshape(batch, -1)).item()
        
    return total_loss/len(data)

def run(train_loader, test_loader, model, nb_epochs, lr):
    """
    This function is the training process.
    """
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    writer = SummaryWriter(f'runs/Hangzhouforecasting_{timestamp}')
    
    savepath = Path("Hangzhouforecasting.pth")
    state = State.load(savepath)
    if state.model is None:
        state.model =  model.to(DEVICE)
        state.optim = torch.optim.Adam(state.model.parameters(), lr=lr)

    loss_fn = nn.MSELoss()
    
    for epoch in range(state.epoch, nb_epochs):
        loss_train = train_loop(train_loader, state, loss_fn)
        loss_test = test_loop(test_loader, state, loss_fn)

        writer.add_scalar('Loss_train', loss_train, state.epoch)
        writer.add_scalar('Loss_test', loss_test, state.epoch)

        print (f'Train loss: {loss_train} \t Test loss: {loss_test}')

        state.epoch = epoch + 1
        state.save()

NB_EPOCHS = 100
DIM_LATENT = 8
LEARNING_RATE = 1e-4
model = RNN(DIM_INPUT, DIM_LATENT, DIM_INPUT)
run(data_train, data_test, model, NB_EPOCHS, LEARNING_RATE)
# END TODO
