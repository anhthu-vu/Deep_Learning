import logging
logging.basicConfig(level=logging.INFO)

from torch.nn.modules.pooling import MaxPool1d
logging.basicConfig(level=logging.INFO)

import heapq
from pathlib import Path
import gzip

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import sentencepiece as spm
import os
from datetime import datetime

from tp8_preprocess import TextDataset

# Utiliser tp8_preprocess pour générer le vocabulaire BPE et
# le jeu de donnée dans un format compact

# --- Configuration

# Taille du vocabulaire
vocab_size = 5000
MAINDIR = Path(__file__).parent

# Chargement du tokenizer

tokenizer = spm.SentencePieceProcessor()
tokenizer.Load(f"wp{vocab_size}.model")
ntokens = len(tokenizer)
logging.info("Number of tokens: %d", ntokens)

def loaddata(mode):
    with gzip.open(f"{mode}-{vocab_size}.pth", "rb") as fp:
        return torch.load(fp)


train = loaddata("train")
TRAIN_BATCHSIZE=500
TEST_BATCHSIZE=500


# --- Chargements des jeux de données train, validation et test

val_size = 1000
test_size = 10000
train_size = len(train) - val_size -test_size
train, val, test = torch.utils.data.random_split(train, [train_size, val_size, test_size])

logging.info("Datasets: train=%d, val=%d, test=%d", train_size, val_size, len(test))
logging.info("Vocabulary size: %d", vocab_size)
train_iter = torch.utils.data.DataLoader(train, batch_size=TRAIN_BATCHSIZE, collate_fn=TextDataset.collate, shuffle=True)
val_iter = torch.utils.data.DataLoader(val, batch_size=TEST_BATCHSIZE, collate_fn=TextDataset.collate)
test_iter = torch.utils.data.DataLoader(test, batch_size=TEST_BATCHSIZE, collate_fn=TextDataset.collate)


#  TODO

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Baseline performance
train_label = torch.tensor([train[i][1] for i in range(train_size)])
test_label = torch.tensor([test[i][1] for i in range(test_size)])
values_train, counts_train = torch.unique(train_label, return_counts=True)
values_test, counts_test = torch.unique(test_label, return_counts=True)
logging.info("Baseline performance: train_accuracy=%f, test_accuracy=%f", torch.max(counts_train)/train_size, torch.max(counts_test)/test_size)


class CNN_sentiment(nn.Module): # batch_first = True
    """
    This is a 1D CNN model. This model will be trained on the Sentiment140 dataset (visit https://www.kaggle.com/datasets/kazanova/sentiment140 for more details). Dataset segmentation will be done in the provided tp8_prepocess.py file using `sentencepiece` (see https://github.com/google/sentencepiece for documentations). 
    
    The input of this model is expected to be of shape batch x sequence_length. 
    """
    def __init__(self, vocab_size=5000, embed_dim=500, out_channels=None, dim_out=3, kernel_size=3, stride=1):
        """
        Parameters:
            out_channels (List[int]): List specifying the number of channels in the output of each convolutional layer
        """
        super(CNN_sentiment, self).__init__()
        self.embed_dim = embed_dim
        self.dim_out = dim_out
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        convolution = [nn.Conv1d(embed_dim, self.out_channels[0], self.kernel_size, self.stride)]
        for i in range(len(self.out_channels)-1):
            convolution.append(MaxPool1d(self.kernel_size))
            convolution.append(nn.Conv1d(self.out_channels[i], self.out_channels[i+1], self.kernel_size, self.stride))
        
        self.embedding = nn.Embedding(vocab_size, embed_dim) 
        self.convolution = nn.Sequential(*convolution)
        self.classifier = nn.Linear(self.out_channels[-1], self.dim_out)

    def forward(self, x): # dim_x = batch x sequence_length
        x = self.embedding(x) # dim_x = batch x sequence_length x embedding_dim
        x = x.transpose(1, 2) # dim_x = batch x embedding_dim x sequence_length
        x = self.convolution(x) # dim_x = batch x self.out_channels[-1] x sequence_length 
        x = torch.max(x, dim=-1)[0] # dim_x = batch x self.out_channels[-1]
        x = self.classifier(x) # dim_x = batch x self.dim_out
        return x


# Training
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
        data: training dataloader with a padded batch of shape sequence_length x batch
        state: checkpointing of the training process
    Return:
        average train loss and train accuracy in that epoch
    """
    
    total_loss, total_acc = 0., 0.
        
    model = state.model
    optimizer = state.optim
    
    for x, y in tqdm(data, desc=f"Epoch {state.epoch+1}"):
        optimizer.zero_grad()

        x, y = x.to(DEVICE), y.to(DEVICE)
        output = model(x)
        loss = loss_fn(output, y)

        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_acc += (torch.argmax(output, dim=-1) == y).float().mean().item()

        state.iteration += 1

    return total_loss/len(data), total_acc/len(data)
        
                    
def test_loop(data: DataLoader, state: State, loss_fn):
    """
    This function is the testing phase in an epoch.
    Parameters:
        data: test dataloader with a padded batch of shape batch x sequence_length 
        state: checkpointing of the training process
    Return:
        average test loss and test accuracy in that epoch
    """
    
    total_loss, total_acc = 0., 0.
    model = state.model
    
    with torch.no_grad():
        for x, y in data:
            x, y = x.to(DEVICE), y.to(DEVICE)
            output = model(x)
            total_loss += loss_fn(output, y).item()
            total_acc += (torch.argmax(output, dim=-1) == y).float().mean().item()
        
    return total_loss/len(data), total_acc/len(data)


def run(train_loader, test_loader, model, nb_epochs, lr):
    """
    This function is the training process.
    """
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    writer = SummaryWriter(f'runs/CNN_Sentiment_{timestamp}')
    
    savepath = Path("CNN_Sentiment.pth")
    state = State.load(savepath)
    if state.model is None:
        state.model =  model.to(DEVICE)
        state.optim = torch.optim.Adam(state.model.parameters(), lr=lr)

    loss_fn = nn.CrossEntropyLoss()
    
    for epoch in range(state.epoch, nb_epochs):
        loss_train, acc_train = train_loop(train_loader, state, loss_fn)
        loss_test, acc_test = test_loop(test_loader, state, loss_fn)
        
        writer.add_scalar('Loss_train', loss_train, state.epoch)
        writer.add_scalar('Loss_test', loss_test, state.epoch)
        writer.add_scalar('Acc_train', acc_train, state.epoch)
        writer.add_scalar('Acc_test', acc_test, state.epoch)
        
        print (f'Train loss: {loss_train} \t Test loss: {loss_test}')
        print (f'Train accuracy: {acc_train} \t Test accuracy: {acc_test}')

        state.epoch = epoch + 1
        state.save()

NB_EPOCHS = 20
LEARNING_RATE = 1e-4
EMBED_DIM = 200
model = CNN_sentiment(vocab_size=5000, embed_dim=EMBED_DIM, out_channels=[150, 100, 50], dim_out=2, kernel_size=3, stride=1)
run(train_iter, test_iter, model, NB_EPOCHS, LEARNING_RATE)
# END TODO
