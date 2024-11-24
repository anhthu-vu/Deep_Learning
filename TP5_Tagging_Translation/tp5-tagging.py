import itertools
import logging
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch
from typing import List
import time
from conllu import parse_incr
from pathlib import Path
from datetime import datetime
import os
logging.basicConfig(level=logging.INFO) 

DATA_PATH = "./data/"


# Format de sortie décrit dans
# https://pypi.org/project/conllu/

class Vocabulary:
    """Permet de gérer un vocabulaire.

    En test, il est possible qu'un mot ne soit pas dans le
    vocabulaire : dans ce cas le token "__OOV__" est utilisé.
    Attention : il faut tenir compte de cela lors de l'apprentissage !

    Utilisation:

    - en train, utiliser v.get("blah", adding=True) pour que le mot soit ajouté
      automatiquement s'il n'est pas connu
    - en test, utiliser v["blah"] pour récupérer l'ID du mot (ou l'ID de OOV)
    """
    OOVID = 1
    PAD = 0

    def __init__(self, oov: bool):
        """ oov : autorise ou non les mots OOV """
        self.oov =  oov
        self.id2word = [ "PAD"]
        self.word2id = { "PAD" : Vocabulary.PAD}
        if oov:
            self.word2id["__OOV__"] = Vocabulary.OOVID
            self.id2word.append("__OOV__")

    def __getitem__(self, word: str):
        if self.oov:
            return self.word2id.get(word, Vocabulary.OOVID)
        return self.word2id[word]

    def get(self, word: str, adding=True):
        try:
            return self.word2id[word]
        except KeyError:
            if adding:
                wordid = len(self.id2word)
                self.word2id[word] = wordid
                self.id2word.append(word)
                return wordid
            if self.oov:
                return Vocabulary.OOVID
            raise

    def __len__(self):
        return len(self.id2word)

    def getword(self,idx: int):
        if idx < len(self):
            return self.id2word[idx]
        return None

    def getwords(self,idx: List[int]):
        return [self.getword(i) for i in idx]



class TaggingDataset():
    def __init__(self, data, words: Vocabulary, tags: Vocabulary, adding=True):
        self.sentences = []

        for s in data:
                self.sentences.append(([words.get(token["form"], adding) for token in s], [tags.get(token["upostag"], adding) for token in s]))
    def __len__(self):
        return len(self.sentences)
    def __getitem__(self, ix):
        return self.sentences[ix]


def collate_fn(batch):
    """Collate using pad_sequence"""
    return tuple(pad_sequence([torch.LongTensor(b[j]) for b in batch]) for j in range(2))

logging.info("Loading datasets...")
words = Vocabulary(True)
tags = Vocabulary(False)

data_file = open(DATA_PATH+"fr_gsd-ud-train.conllu",encoding="utf-8")
train_data = TaggingDataset(parse_incr(data_file), words, tags, True)

data_file = open(DATA_PATH+"fr_gsd-ud-dev.conllu",encoding='utf-8')
dev_data = TaggingDataset(parse_incr(data_file), words, tags, True)

data_file = open(DATA_PATH+"fr_gsd-ud-test.conllu",encoding="utf-8")
test_data = TaggingDataset(parse_incr(data_file), words, tags, False)


logging.info("Vocabulary size: %d", len(words))
logging.info("Tag size: %d", len(tags))


BATCH_SIZE=100

train_loader = DataLoader(train_data, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=True)
dev_loader = DataLoader(dev_data, collate_fn=collate_fn, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_data, collate_fn=collate_fn, batch_size=BATCH_SIZE)


# TODO
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Seq2SeqTagger(nn.Module):
    """
    This is a tagging model focuses on the task of syntactic analysis (Part-Of-Speech). It will be trained on will be trained on the GSD dataset (visit https://github.com/UniversalDependencies/UD_French-GSD for more details). 

    The dataset has already been segmented into words using the provided classes, Vocabulary and TaggingDataset, above.
    """
    
    def __init__(self, vocab_size, tagset_size, embedding_dim, hidden_dim):
        super(Seq2SeqTagger, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=Vocabulary.PAD)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentences, lengths):
        """
        Parameters: 
            sentences: padded batch of sentences
            lengths: lengths of examples in «sentences»
        Return:
            hidden_states: sequence of hidden states of shape sequence_length x batch x dim_latent
        """
        
        embed = self.embedding(sentences)
        packed_embed = pack_padded_sequence(embed, lengths.cpu(), enforce_sorted=False)
        packed_hiddens, _ = self.lstm(packed_embed)
        hiddens, _ = pad_packed_sequence(packed_hiddens)
        
        return hiddens

    def decode(self, hiddens):
        """
        This function decodes latent states to produce an output for downstream tasks.
        Parameters:
            hiddens: latent states of shape sequence_length x batch x dim_latent
        Return:
            Output of the decoder of shape sequence_length x batch x dim_output
        """
        
        return self.decoder(hiddens)

# Introduce random OOV tokens
def add_random_oov(batch, oov_prob=0.1):
    """Randomly replaces tokens in the batch with OOV token with probability oov_prob"""
    for i in range(batch.shape[0]):
        if torch.rand(1).item() < oov_prob:
            batch[i] = Vocabulary.OOVID
            
    return batch


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


def train_loop(data: DataLoader, state: State, loss_fn, oov_prob=0.1):
    """
    This function is the training phase in an epoch.
    Parameters:
        data: training dataloader with a batch of shape sequence_length x batch
        state: checkpointing of the training process
        oov_prob: the probability to replace tokens in the batch with OOV token
    Return:
        average train loss in that epoch
    """
    
    total_loss = 0.
    model = state.model
    optimizer = state.optim
    
    for sentences, tags in tqdm(data, desc=f"Epoch {state.epoch+1}"):
        optimizer.zero_grad()

        sentences, tags = sentences.to(DEVICE), tags.to(DEVICE) # dim_sentences = length x batch
        lengths = (sentences != Vocabulary.PAD).sum(dim=0)
        sentences = add_random_oov(sentences.flatten(), oov_prob).view_as(sentences)  # Apply random OOV replacement

        hiddens = model(sentences, lengths)
        tag_scores = state.model.decode(hiddens) # dim = length x batch x size_tagset
        loss = loss_fn(tag_scores.view(-1, tag_scores.shape[-1]), tags.view(-1))

        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

        state.iteration += 1
        
    return total_loss/len(data)
        

def test_loop(data: DataLoader, state: State, loss_fn):
    """
    This function is the testing phase in an epoch.
    Parameters:
        data: test dataloader with a batch of shape sequence_length x batch 
        state: checkpointing of the training process
    Return:
        average test loss in that epoch
    """
    
    total_loss = 0.
    model = state.model
    
    with torch.no_grad():
        for sentences, tags in data:
            sentences, tags = sentences.to(DEVICE), tags.to(DEVICE) # dim_sentences = length x batch
            lengths = (sentences != Vocabulary.PAD).sum(dim=0)
    
            hiddens = model(sentences, lengths)
            tag_scores = state.model.decode(hiddens) # dim = length x batch x size_tagset
            total_loss += loss_fn(tag_scores.view(-1, tag_scores.shape[-1]), tags.view(-1)).item()

    return total_loss/len(data)

    
def run(train_loader, test_loader, model, nb_epochs, lr):
    """
    This function is the training process.
    """
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    writer = SummaryWriter(f'runs/Tagging_{timestamp}')
    
    savepath = Path("Tagging.pth")
    state = State.load(savepath)
    if state.model is None:
        state.model =  model.to(DEVICE)
        state.optim = torch.optim.Adam(state.model.parameters(), lr=lr)

    loss_fn = nn.CrossEntropyLoss(ignore_index=Vocabulary.PAD)
    
    for epoch in range(state.epoch, nb_epochs):
        loss_train = train_loop(train_loader, state, loss_fn)
        loss_test = test_loop(test_loader, state, loss_fn)

        writer.add_scalar('Loss_train', loss_train, state.epoch)
        writer.add_scalar('Loss_test', loss_test, state.epoch)

        print (f'Train loss: {loss_train} \t Test loss: {loss_test}')

        state.epoch = epoch + 1
        state.save()


# Training 
EMBED_DIM = 200
HIDDEN_DIM = 100
NB_EPOCHS = 100
LEARNING_RATE = 1e-4
model = Seq2SeqTagger(len(words), len(tags), EMBED_DIM, HIDDEN_DIM)
run(train_loader, test_loader, model, NB_EPOCHS, LEARNING_RATE)


# Visualize predictions 
logging.info('Visualize predictions')
sentence = ['Le', 'chat', 'est', 'noir', '.']
state = State.load(Path("Tagging.pth"))
with torch.no_grad():
    sentence2id = torch.LongTensor([words.get(w, adding=False) for w in sentence]).unsqueeze(-1).to(DEVICE)
    length = torch.LongTensor([len(sentence)])
    hiddens = state.model(sentence2id, length)
    tag_scores = state.model.decode(hiddens)
    predictions = tag_scores.argmax(dim=-1).squeeze(-1).cpu().tolist()
    tags_predicted = tags.getwords(predictions)
    
print (f"Sentence: {' '.join(sentence)}")
print (f"Predicted Tags: {' '.join(tags_predicted)}")
# END TODO
