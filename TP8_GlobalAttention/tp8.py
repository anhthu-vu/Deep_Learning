import logging
import re
from pathlib import Path
from tqdm import tqdm
import numpy as np


import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence, pack_padded_sequence
from torch.utils.tensorboard import SummaryWriter
import time
import os
from torch.distributions.categorical import Categorical

GLOVE_PATH = Path("data/glove")
DATASET_PATH = Path("data/aclImdb")
IMDB_CLASSES  = ['neg','pos']

class FolderText(Dataset):
    """Dataset basé sur des dossiers (un par classe) et fichiers"""

    def __init__(self, classes, folder: Path, tokenizer, load=False):
        self.tokenizer = tokenizer
        self.files = []
        self.filelabels = []
        self.labels = {}
        for ix, key in enumerate(classes):
            self.labels[key] = ix

        for label in classes:
            for file in (folder / label).glob("*.txt"):
                self.files.append(file.read_text() if load else file)
                self.filelabels.append(self.labels[label])

    def __len__(self):
        return len(self.filelabels)

    def __getitem__(self, ix):
        s = self.files[ix]
        return self.tokenizer(s if isinstance(s, str) else s.read_text()), self.filelabels[ix]

def get_imdb_data(embedding_size=50):
    """Renvoie l'ensemble des donnéees nécessaires pour l'apprentissage

    - dictionnaire word vers ID
    - embeddings (Glove)
    - DataSet (FolderText)

    """
    WORDS = re.compile(r"\S+")
    glove_fn = open(GLOVE_PATH / ("glove.6B.%dd.txt" % embedding_size))
    words, embeddings = [], []
    for line in glove_fn:
        values = line.split()
        words.append(values[0]) # words: List(str)
        embeddings.append([float(x) for x in values[1:]]) 

    OOVID = len(words)
    words.append("__OOV__")

    word2id = {word: ix for ix, word in enumerate(words)}
    embeddings = np.vstack((embeddings, np.zeros(embedding_size))) 

    def tokenizer(t):
        return [word2id.get(x, OOVID) for x in re.findall(WORDS, t.lower())]

    logging.info("Loading embeddings")

    logging.info("Get the IMDB dataset")


    return word2id, embeddings, FolderText(IMDB_CLASSES, DATASET_PATH /"train", tokenizer, load=False), FolderText(IMDB_CLASSES, DATASET_PATH / "test", tokenizer, load=False)


#  TODO: 
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32
word2id, embeddings, train_dataset, test_dataset = get_imdb_data()
embeddings = np.vstack((embeddings, np.ones(50)))


def collate_fn(batch):
    sentences, lengths, labels = [], [], []
    for s in batch:
        sentences.append(torch.LongTensor(s[0])) 
        lengths.append(len(s[0]))
        labels.append(s[1])

    sentences = pad_sequence(sentences, padding_value=len(word2id), batch_first=True)
    lengths = torch.tensor(lengths)
    labels = torch.tensor(labels)
    
    return sentences, lengths, labels

train_loader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=False)


class basic_model(nn.Module):
    """
    This is the baseline model where a text is represented by the mean of the embeddings of the words it contains.
    """
    def __init__(self, embed_dim):
        super(basic_model, self).__init__()
        self.embed_dim = embed_dim
        self.classifier = nn.Linear(self.embed_dim, 2)
        
    def forward(self, embed_batch, lengths): # dim_batch = batch_size x length x embed_dim, dim_lengths = batch_size
        mask = [torch.ones(i) for i in lengths]
        mask = pad_sequence(mask, batch_first=True).unsqueeze(-1).to(DEVICE) # dim = batch_size x length x 1
        t = torch.sum(embed_batch*mask, dim=1)/lengths.unsqueeze(-1)
        y = self.classifier(t)
        return y

    
class State:
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


def train_loop(state, dataloader, loss_fn, emb):
    total_loss, total_acc = 0., 0.
    model = state.model
    optim = state.optim

    for sentences, lengths, labels in tqdm(dataloader, desc=f'Epoch {state.epoch+1} Train'):
        optim.zero_grad()
        
        sentences, lengths, labels = sentences.to(DEVICE), lengths.to(DEVICE), labels.to(DEVICE)
        sentences_emb = emb(sentences).float()
        output = model(sentences_emb, lengths)
        loss = loss_fn(output, labels)
        
        total_loss += loss.item()
        loss.backward()
        optim.step()

        total_acc += (output.argmax(dim=1) == labels).float().mean().item()
        
        state.iteration += 1

    return total_loss/len(dataloader), total_acc/len(dataloader)


def test_loop(state, dataloader, loss_fn, emb):
    total_loss, total_acc = 0., 0.
    model = state.model

    for sentences, lengths, labels in tqdm(dataloader, desc='Test'):
        with torch.no_grad():
            sentences, lengths, labels = sentences.to(DEVICE), lengths.to(DEVICE), labels.to(DEVICE)
            sentences_emb = emb(sentences).float()
            output = model(sentences_emb, lengths)
            total_loss += loss_fn(output, labels).item()
            total_acc += (output.argmax(dim=1) == labels).float().mean().item()
    
    return total_loss/len(dataloader), total_acc/len(dataloader)


def run(model, train_loader, test_loader, NB_EPOCHS, LEARNING_RATE, loss_fn, emb, name_model=''):
    writer = SummaryWriter(f'runs/{name_model}-' + time.asctime())
    savepath = Path(f'{name_model}.pth')
    state = State.load(savepath)

    if state.model is None:
        state.model = model.to(DEVICE)
        state.optim = torch.optim.Adam(state.model.parameters(), lr=LEARNING_RATE)
        
    for _ in range(state.epoch, NB_EPOCHS):
        loss_train, acc_train = train_loop(state, train_loader, loss_fn, emb)
        loss_test, acc_test = test_loop(state, test_loader, loss_fn, emb)

        writer.add_scalar('Loss_train', loss_train, state.epoch)
        writer.add_scalar('Acc_train', acc_train, state.epoch)
        writer.add_scalar('Loss_test', loss_test, state.epoch)
        writer.add_scalar('Acc_test', acc_test, state.epoch)

        print (f'Loss_train: {loss_train}, Loss_test: {loss_test}')
        print (f'Acc_train: {acc_train}, Acc_test: {acc_test}')
        state.epoch += 1
        state.save()


emb = nn.Embedding.from_pretrained(torch.tensor(embeddings), freeze=True).to(DEVICE)
NB_EPOCHS = 150
LEARNING_RATE = 1e-4
loss_fn = nn.CrossEntropyLoss()
model = basic_model(50)
run(model, train_loader, test_loader, NB_EPOCHS, LEARNING_RATE, loss_fn, emb, name_model='BasicModel')


class SimpleAttention(nn.Module):
    """
    This is a simple global attention model, where the query q is a learnable vector, the key k and value v are the embeddings of the words.
    """
    def __init__(self, embed_dim):
        super(SimpleAttention, self).__init__()
        self.embed_dim = embed_dim
        self.classifier = nn.Linear(self.embed_dim, 2)
        self.question = nn.Parameter(torch.randn(self.embed_dim)) # dim = embed_dim

    def forward(self, embed_batch, lengths): # dim_batch = batch_size x length x embed_dim
        mask = [torch.zeros(i, dtype=torch.bool) for i in lengths]
        mask = pad_sequence(mask, batch_first=True, padding_value=1).to(DEVICE) # dim = batch_size x length 
        attention = torch.matmul(embed_batch, self.question.unsqueeze(-1).unsqueeze(0)).squeeze(-1) # dim = batch_size x length
        attention = attention.masked_fill(mask, -float('inf'))
        attention = F.softmax(attention, dim=1) # dim = batch_size x length
        t = torch.matmul(embed_batch.transpose(1, 2), attention.unsqueeze(-1)).squeeze(-1) # dim = batch_size x embed_dim
        y = self.classifier(t)
        
        return y, attention


def train_loop(state, dataloader, loss_fn, emb, writer, log_entropy=False):
    total_loss, total_acc = 0., 0.
    model = state.model
    optim = state.optim

    for sentences, lengths, labels in tqdm(dataloader, desc=f'Epoch {state.epoch+1} Train'):
        optim.zero_grad()
        
        sentences, lengths, labels = sentences.to(DEVICE), lengths.to(DEVICE), labels.to(DEVICE)
        sentences_emb = emb(sentences).float()
        if log_entropy:
            output, attention = model(sentences_emb, lengths)
            entropy = Categorical(attention).entropy().detach()
            writer.add_histogram('Entropy', entropy, state.epoch)
        else:
            output, _ = model(sentences_emb, lengths)
        loss = loss_fn(output, labels)
        
        total_loss += loss.item()
        loss.backward()
        optim.step()

        total_acc += (output.argmax(dim=1) == labels).float().mean().item()
        
        state.iteration += 1

    return total_loss/len(dataloader), total_acc/len(dataloader)


def test_loop(state, dataloader, loss_fn, emb):
    total_loss, total_acc = 0., 0.
    model = state.model

    for sentences, lengths, labels in tqdm(dataloader, desc='Test'):
        with torch.no_grad():
            sentences, lengths, labels = sentences.to(DEVICE), lengths.to(DEVICE), labels.to(DEVICE)
            sentences_emb = emb(sentences).float()
            output, _ = model(sentences_emb, lengths)
            total_loss += loss_fn(output, labels).item()
            total_acc += (output.argmax(dim=1) == labels).float().mean().item()
    
    return total_loss/len(dataloader), total_acc/len(dataloader)


def run(model, train_loader, test_loader, NB_EPOCHS, LEARNING_RATE, loss_fn, emb, name_model=''):
    writer = SummaryWriter(f'runs/{name_model}-' + time.asctime())
    savepath = Path(f'{name_model}.pth')
    state = State.load(savepath)

    if state.model is None:
        state.model = model.to(DEVICE)
        state.optim = torch.optim.Adam(state.model.parameters(), lr=LEARNING_RATE)
        
    for _ in range(state.epoch, NB_EPOCHS):
        if state.epoch % 10 == 9:
            loss_train, acc_train = train_loop(state, train_loader, loss_fn, emb, writer, log_entropy=True)
        else: 
            loss_train, acc_train = train_loop(state, train_loader, loss_fn, emb, writer)
        loss_test, acc_test = test_loop(state, test_loader, loss_fn, emb)

        writer.add_scalar('Loss_train', loss_train, state.epoch)
        writer.add_scalar('Acc_train', acc_train, state.epoch)
        writer.add_scalar('Loss_test', loss_test, state.epoch)
        writer.add_scalar('Acc_test', acc_test, state.epoch)

        print (f'Loss_train: {loss_train}, Loss_test: {loss_test}')
        print (f'Acc_train: {acc_train}, Acc_test: {acc_test}')
        
        state.epoch += 1
        state.save()


emb = nn.Embedding.from_pretrained(torch.tensor(embeddings), freeze=True).to(DEVICE)
NB_EPOCHS = 150
LEARNING_RATE = 1e-4
loss_fn = nn.CrossEntropyLoss()
model = SimpleAttention(50)
run(model, train_loader, test_loader, NB_EPOCHS, LEARNING_RATE, loss_fn, emb, name_model='SimpleAttention')


# Adding entropy regularization
def train_loop(state, dataloader, loss_fn, emb, writer, log_entropy=False, reg_entropy=None):
    total_loss, total_acc = 0., 0.
    model = state.model
    optim = state.optim

    for sentences, lengths, labels in tqdm(dataloader, desc=f'Epoch {state.epoch+1} Train'):
        optim.zero_grad()
        
        sentences, lengths, labels = sentences.to(DEVICE), lengths.to(DEVICE), labels.to(DEVICE)
        sentences_emb = emb(sentences).float()
        output, attention = model(sentences_emb, lengths)
        entropy = Categorical(attention).entropy()
        if log_entropy:
            writer.add_histogram('Entropy', entropy.detach(), state.epoch)
        
        loss = loss_fn(output, labels) 
        if reg_entropy is not None:
            loss_with_entropy = loss + reg_entropy*entropy.mean() 
        else:
            loss_with_entropy = loss
        total_loss += loss.item()
        loss_with_entropy.backward()
        optim.step()
        
        total_acc += (output.argmax(dim=1) == labels).float().mean().item()
        
        state.iteration += 1

    return total_loss/len(dataloader), total_acc/len(dataloader)


def run(model, train_loader, test_loader, NB_EPOCHS, LEARNING_RATE, loss_fn, emb, name_model=''):
    writer = SummaryWriter(f'runs/{name_model}-' + time.asctime())
    savepath = Path(f'{name_model}.pth')
    state = State.load(savepath)

    if state.model is None:
        state.model = model.to(DEVICE)
        state.optim = torch.optim.Adam(state.model.parameters(), lr=LEARNING_RATE)
        
    for _ in range(state.epoch, NB_EPOCHS):
        if state.epoch % 10 == 9:
            loss_train, acc_train = train_loop(state, train_loader, loss_fn, emb, writer, log_entropy=True, reg_entropy=0.01)
        else: 
            loss_train, acc_train = train_loop(state, train_loader, loss_fn, emb, writer, reg_entropy=0.01)
        loss_test, acc_test = test_loop(state, test_loader, loss_fn, emb)

        writer.add_scalar('Loss_train', loss_train, state.epoch)
        writer.add_scalar('Acc_train', acc_train, state.epoch)
        writer.add_scalar('Loss_test', loss_test, state.epoch)
        writer.add_scalar('Acc_test', acc_test, state.epoch)

        print (f'Loss_train: {loss_train}, Loss_test: {loss_test}')
        print (f'Acc_train: {acc_train}, Acc_test: {acc_test}')
        
        state.epoch += 1
        state.save()

        
class ComplexAttention(nn.Module):
    """
    This is a more complex global attention model, where the query q is a linear transformation of the mean of the embeddings, the key k are the embeddings, and the value v is a linear transformation of the embeddings.
    """
    def __init__(self, embed_dim):
        super(ComplexAttention, self).__init__()
        self.embed_dim = embed_dim
        self.classifier = nn.Linear(self.embed_dim, 2)
        self.question = nn.Linear(self.embed_dim, self.embed_dim) 
        self.value = nn.Linear(self.embed_dim, self.embed_dim) 

    def forward(self, embed_batch, lengths): # dim_batch = batch_size x length x embed_dim
        mask = [torch.ones(i, dtype=torch.bool) for i in lengths]
        mask = pad_sequence(mask, batch_first=True, padding_value=0).to(DEVICE) # dim = batch_size x length 
    
        t = torch.sum(embed_batch*mask.unsqueeze(-1), dim=1)/lengths.unsqueeze(-1) # dim = batch_size x embed_dim
        question = self.question(t) 
        
        attention = torch.matmul(embed_batch, question.unsqueeze(-1)).squeeze(-1) # dim = batch_size x length
        attention = attention.masked_fill(~mask, -float('inf'))
        attention = F.softmax(attention, dim=1) # dim = batch_size x length
        
        values = self.value(embed_batch)
        t_hat = torch.matmul(values.transpose(1, 2), attention.unsqueeze(-1)).squeeze(-1) # dim = batch_size x embed_dim
        y = self.classifier(t_hat)
        
        return y, attention


emb = nn.Embedding.from_pretrained(torch.tensor(embeddings), freeze=True).to(DEVICE)
NB_EPOCHS = 40
LEARNING_RATE = 1e-4
loss_fn = nn.CrossEntropyLoss()
model = ComplexAttention(50)
run(model, train_loader, test_loader, NB_EPOCHS, LEARNING_RATE, loss_fn, emb, name_model='ComplexAttention')


class Attention_with_GRU(nn.Module):
    """
    This global attention model contains a GRU. The query q is a linear transformation of the last hidden state, the key k  and value v are the hidden states.
    """
    def __init__(self, embed_dim, hidden_dim):
        super(Attention_with_GRU, self).__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.classifier = nn.Linear(self.hidden_dim, 2)
        self.question = nn.Linear(self.hidden_dim, self.hidden_dim) 
        self.GRU = nn.GRU(self.embed_dim, self.hidden_dim, batch_first=True) 

    def forward(self, packed_embed_batch, lengths): # dim_batch = batch_size x length x embed_dim
        mask = [torch.ones(i, dtype=torch.bool) for i in lengths]
        mask = pad_sequence(mask, batch_first=True, padding_value=0).to(DEVICE) # dim = batch_size x length 

        hidden_states, h_n = self.GRU(packed_embed_batch) 
        hidden_states, _ = pad_packed_sequence(hidden_states, batch_first=True) # dim = batch_size x length x hidden_dim
        h_n = h_n.squeeze(0) # dim = batch_size x hidden_dim
        question = self.question(h_n) # dim = batch_size x hidden_dim
        
        attention = torch.matmul(hidden_states, question.unsqueeze(-1)).squeeze(-1) # dim = batch_size x length
        attention = attention.masked_fill(~mask, -float('inf'))
        attention = F.softmax(attention, dim=1) # dim = batch_size x length
        
        t_hat = torch.matmul(hidden_states.transpose(1, 2), attention.unsqueeze(-1)).squeeze(-1) # dim = batch_size x hidden_dim
        y = self.classifier(t_hat)
        
        return y, attention


def train_loop(state, dataloader, loss_fn, emb, writer, log_entropy=False):
    total_loss, total_acc = 0., 0.
    model = state.model
    optim = state.optim

    for sentences, lengths, labels in tqdm(dataloader, desc=f'Epoch {state.epoch+1} Train'):
        optim.zero_grad()
        
        sentences, lengths, labels = sentences.to(DEVICE), lengths.to(DEVICE), labels.to(DEVICE)
        sentences_emb = emb(sentences).float()
        sentences_emb = pack_padded_sequence(sentences_emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        if log_entropy:
            output, attention = model(sentences_emb, lengths)
            entropy = Categorical(attention).entropy()
            writer.add_histogram('Entropy', entropy, state.epoch)
        else:
            output, _ = model(sentences_emb, lengths)
        loss = loss_fn(output, labels)
        
        total_loss += loss.item()
        loss.backward()
        optim.step()

        total_acc += (output.argmax(dim=1) == labels).float().mean().item()
        
        state.iteration += 1

    return total_loss/len(dataloader), total_acc/len(dataloader)


def test_loop(state, dataloader, loss_fn, emb):
    total_loss, total_acc = 0., 0.
    model = state.model

    for sentences, lengths, labels in tqdm(dataloader, desc='Test'):
        with torch.no_grad():
            sentences, lengths, labels = sentences.to(DEVICE), lengths.to(DEVICE), labels.to(DEVICE)
            sentences_emb = emb(sentences).float()
            sentences_emb = pack_padded_sequence(sentences_emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
            output, _ = model(sentences_emb, lengths)
            total_loss += loss_fn(output, labels).item()
            total_acc += (output.argmax(dim=1) == labels).float().mean().item()
    
    return total_loss/len(dataloader), total_acc/len(dataloader)


emb = nn.Embedding.from_pretrained(torch.tensor(embeddings), freeze=True).to(DEVICE)
NB_EPOCHS = 30
LEARNING_RATE = 1e-4
loss_fn = nn.CrossEntropyLoss()
model = Attention_with_GRU(50, 50)
run(model, train_loader, test_loader, NB_EPOCHS, LEARNING_RATE, loss_fn, emb, name_model='Attention_with_GRU')
# END TODO