import torch
import torch.nn as nn
from torch.utils.data import Dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO
class RNN(nn.Module):
    """
    This is a simple RNN that I code by hand (according to question 1 of the tp sujet).
    It takes as input a batch of shape sequence_length x batch x dim_input.
    It aims to treat fixed size sequences per batch (different sizes for different batches are possible).
    """
    
    def __init__(self, dim_input, dim_latent, dim_output):
        super(RNN, self).__init__()

        self.dim_input = dim_input 
        self.dim_latent = dim_latent 
        self.dim_output = dim_output 
        self.encoder = nn.Linear(self.dim_input+self.dim_latent, self.dim_latent)
        self.activation = nn.Tanh()
        self.decoder = nn.Linear(self.dim_latent, self.dim_output)
        
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
# END TODO
        
        

class SampleMetroDataset(Dataset):
    def __init__(self, data,length=20,stations_max=None):
        """
            * data : tenseur des données au format  Nb_days x Nb_slots x Nb_Stations x {In,Out}
            * length : longueur des séquences d'exemple
            * stations_max : normalisation à appliquer
        """
        self.data, self.length= data, length
        ## Si pas de normalisation passée en entrée, calcul du max du flux entrant/sortant
        self.stations_max = stations_max if stations_max is not None else torch.max(self.data.view(-1,self.data.size(2),self.data.size(3)),0)[0]
        ## Normalisation des données
        self.data = self.data / self.stations_max
        self.nb_days, self.nb_timeslots, self.classes = self.data.size(0), self.data.size(1), self.data.size(2)

    def __len__(self):
        ## longueur en fonction de la longueur considérée des séquences
        return self.classes*self.nb_days*(self.nb_timeslots - self.length)

    def __getitem__(self,i):
        ## transformation de l'index 1d vers une indexation 3d
        ## renvoie une séquence de longueur length et l'id de la station.
        station = i // ((self.nb_timeslots-self.length) * self.nb_days)
        i = i % ((self.nb_timeslots-self.length) * self.nb_days)
        timeslot = i // self.nb_days
        day = i % self.nb_days
        return self.data[day,timeslot:(timeslot+self.length),station],station

        
class ForecastMetroDataset(Dataset):
    def __init__(self, data,length=20,stations_max=None):
        """
            * data : tenseur des données au format  Nb_days x Nb_slots x Nb_Stations x {In,Out}
            * length : longueur des séquences d'exemple
            * stations_max : normalisation à appliquer
        """
        self.data, self.length= data,length
        ## Si pas de normalisation passée en entrée, calcul du max du flux entrant/sortant
        self.stations_max = stations_max if stations_max is not None else torch.max(self.data.view(-1,self.data.size(2),self.data.size(3)),0)[0]
        ## Normalisation des données
        self.data = self.data / self.stations_max
        self.nb_days, self.nb_timeslots, self.classes = self.data.size(0), self.data.size(1), self.data.size(2)

    def __len__(self):
        ## longueur en fonction de la longueur considérée des séquences
        return self.nb_days*(self.nb_timeslots - self.length)

    def __getitem__(self,i):
        ## Transformation de l'indexation 1d vers indexation 2d
        ## renvoie x[d,t:t+length-1,:,:], x[d,t+1:t+length,:,:]
        timeslot = i // self.nb_days
        day = i % self.nb_days
        return self.data[day,timeslot:(timeslot+self.length-1)],self.data[day,(timeslot+1):(timeslot+self.length)]
