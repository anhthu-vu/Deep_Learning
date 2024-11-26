import logging
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch
import unicodedata
import string
from tqdm import tqdm
from pathlib import Path
from typing import List
from datetime import datetime
import os
import sentencepiece as spm

import time
import re


logging.basicConfig(level=logging.INFO)

FILE = "./data/en-fra.txt"


def normalize(s):
    return re.sub(' +',' ', "".join(c if c in string.ascii_letters else " "
         for c in unicodedata.normalize('NFD', s.lower().strip())
         if  c in string.ascii_letters+" "+string.punctuation)).strip()


class Vocabulary:
    """Permet de gérer un vocabulaire.

    En test, il est possible qu'un mot ne soit pas dans le
    vocabulaire : dans ce cas le token "__OOV__" est utilisé.
    Attention : il faut tenir compte de cela lors de l'apprentissage !

    Utilisation:

    - en train, utiliser v.get("blah", adding=True) pour que le mot soit ajouté
      automatiquement
    - en test, utiliser v["blah"] pour récupérer l'ID du mot (ou l'ID de OOV)
    """
    PAD = 0
    EOS = 1
    SOS = 2
    OOVID = 3

    def __init__(self, oov: bool):
        self.oov = oov
        self.id2word = ["PAD", "EOS", "SOS"]
        self.word2id = {"PAD": Vocabulary.PAD, "EOS": Vocabulary.EOS, "SOS": Vocabulary.SOS}
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

    def getword(self, idx: int):
        if idx < len(self):
            return self.id2word[idx]
        return None

    def getwords(self, idx: List[int]):
        return [self.getword(i) for i in idx]



class TradDataset():
    def __init__(self,data,vocOrig,vocDest,adding=True,max_len=10):
        self.sentences =[]
        for s in tqdm(data.split("\n")):
            if len(s)<1:continue
            orig,dest=map(normalize,s.split("\t")[:2])
            if len(orig)>max_len: continue
            self.sentences.append((torch.tensor([vocOrig.get(o) for o in orig.split(" ")]+[Vocabulary.EOS]),torch.tensor([vocDest.get(o) for o in dest.split(" ")]+[Vocabulary.EOS])))
    def __len__(self):return len(self.sentences)
    def __getitem__(self,i): return self.sentences[i]



def collate_fn(batch):
    orig,dest = zip(*batch)
    o_len = torch.tensor([len(o) for o in orig])
    d_len = torch.tensor([len(d) for d in dest])
    return pad_sequence(orig),o_len,pad_sequence(dest),d_len


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


with open(FILE) as f:
    lines = f.readlines()

lines = [lines[x] for x in torch.randperm(len(lines))]
idxTrain = int(0.8*len(lines))

vocEng = Vocabulary(True)
vocFra = Vocabulary(True)
MAX_LEN=100
BATCH_SIZE=100

datatrain = TradDataset("".join(lines[:idxTrain]),vocEng,vocFra,max_len=MAX_LEN)
datatest = TradDataset("".join(lines[idxTrain:]),vocEng,vocFra,max_len=MAX_LEN)

train_loader = DataLoader(datatrain, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(datatest, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=False)

# TODO
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info("English vocabulary size: %d", len(vocEng))
logging.info("French vocabulary size: %d", len(vocFra))

class Encoder(nn.Module):
    """
    Encoder of the Seq2Seq translation model. It processes the entire input sequence and generates the final hidden state.
    """
    
    def __init__(self, vocab_orig, embedding_dim, hidden_dim):
        """
        Parameters:
            vocab_orig: vocabulary size of the original language
        """
        
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_orig, embedding_dim, padding_idx=Vocabulary.PAD)
        self.gru = nn.GRU(embedding_dim, hidden_dim)
    
    def forward(self, sentences, lengths): # dim_sentences = length x batch, lengths = o_len  
        """
        Parameters:
            sentences: padded batch of original sentences of shape length_sequence x batch
            lengths: lengths of examples in «sentences»
        Return:
            hn: last hidden state of shape 1 x batch x hidden_dim
        """
        
        embed = self.embedding(sentences) # dim = length x batch x embedding_dim
        packed_embedded = pack_padded_sequence(embed, lengths.cpu(), enforce_sorted=False)
        _, hn = self.gru(packed_embedded) # Last hidden state
        
        return hn # dim = 1 x batch x hidden_dim


class Decoder(nn.Module):
    """
    Decoder of the Seq2Seq translation model. It includes two important methods:
        - forward: corresponds to the `teacher forcing` mode, where the target sentence is passed to the decoder
        - generate: corresponds to the unconstrained mode, where the token corresponding to the maximum probability of decoding the latent state of the previous step is introduced at each time step
    """
    
    def __init__(self, vocab_des, embedding_dim, hidden_dim):
        """
        Parameters:
            vocab_des: vocabulary size of the destination language
        """
        
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_des, embedding_dim, padding_idx=Vocabulary.PAD)
        self.gru = nn.GRU(embedding_dim, hidden_dim)
        self.decode = nn.Linear(hidden_dim, vocab_des)

    def one_step(self, x_t, h_t_minus_one):
        """
        Parameters:
            x_t: input of the timestep t of shape batch 
            h_t_minus_one: latent state of the previous timestep (t-1) of shape 1 x batch x dim_latent
        Return:
            h_t: latent state of the timestep t of shape 1 x batch x dim_latent
        """
        embed = self.embedding(x_t).unsqueeze(0)  # dim = 1 x batch x embedding_dim
        _, h_t = self.gru(embed, h_t_minus_one)
        
        return h_t # dim = 1 x batch x hidden_dim

    def forward(self, sentences, lengths, hn): # Teacher forcing: sentences = destinations, lengths = d_len
        """
        Parameters:
            sentences: padded batch of destination sentences of shape length_sequence x batch
            lengths: lengths of examples in «sentences»
            hn: latent state of shape 1 x batch x dim_latent produced by the encoder
        Return:
            output: logits of predicted tokens 
        """
         
        batch_size = hn.shape[1]
        sos = torch.tensor([Vocabulary.SOS]*batch_size, device=hn.device, dtype=torch.long).unsqueeze(0)
        sentences_with_sos = torch.cat([sos, sentences], dim=0) # dim = (length+1) x batch x embedding_dim
        lengths_with_sos = lengths + 1
        
        embed = self.embedding(sentences_with_sos) # dim = (length+1) x batch x embedding_dim
        packed_embed = pack_padded_sequence(embed, lengths_with_sos.cpu(), enforce_sorted=False)
        packed_hiddens, _ = self.gru(packed_embed, hn) # dim = (length+1) x batch x hidden_dim
        hiddens, _ = pad_packed_sequence(packed_hiddens)
        output = self.decode(hiddens[:-1]) # dim = length x batch x vocab_des

        return output

    def generate(self, hn, lenseq): # Unconstrained mode 
        """
        Parameters:
            hn: latent state of shape 1 x batch x dim_latent produced by the encoder
            lenseq: maximal length of sentences in the batch
        Return:
            generated_sequences: logits of predicted tokens
        """
        batch_size = hn.shape[1]
        input_token = torch.tensor([Vocabulary.SOS] * batch_size, device=hn.device, dtype=torch.long)
        hidden = self.one_step(input_token, hn) # dim = 1 x batch x hidden_dim
        output_token = self.decode(hidden) # dim = 1 x batch x vocab_des
        
        generated_sequences = [] # Store generated words in all timesteps 
        active_mask = torch.ones(batch_size, device=hn.device) # Whether EOS is generated or not
        
        for _ in range(lenseq):
            generated_sequences.append(output_token.squeeze(0))
            top1 = output_token.argmax(-1).squeeze(0)  # dim = batch 
            active_mask = (top1 != Vocabulary.EOS).int()
        
            if not active_mask.any():
                break
        
            input_token = top1 * active_mask # If active_mask[i]=0 (EOS is generated), then input_token[i] is just the padding ID
            hidden = self.one_step(input_token, hidden) 
            output_token = self.decode(hidden) 

        generated_sequences = torch.stack(generated_sequences)
        length, batch, output_dim = generated_sequences.shape
        if length < lenseq: # Padding in case that the length of the output is less than that of the target
            generated_sequences = torch.cat([generated_sequences, torch.empty((lenseq-length, batch, output_dim), device=hn.device).fill_(Vocabulary.PAD)])

        return generated_sequences 


class Seq2SeqTrans(nn.Module):
    """
    I would like to note that unlike the original paper (visit https://arxiv.org/pdf/1506.03099 for more details), the model makes the choice between the two learning modes (teacher forcing and unconstrained mode) for the entire minibatch with a fixed probability.
    """
    def __init__(self, encoder, decoder):
        super(Seq2SeqTrans, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, orig, des, len_orig, len_des, teacher_forcing=False):
        hidden = self.encoder(orig, len_orig)
        if teacher_forcing:
            output = self.decoder(des, len_des, hidden)
        else:
            lenseq = torch.max(len_des).item()
            output = self.decoder.generate(hidden, lenseq)
            
        return output 


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
        data: training dataloader (each example in the padded batch is a tuple of orig, len_orig, des, len_des)
        state: checkpointing of the training process
    Return:
        average train loss in that epoch
    """
    
    total_loss = 0.
    model = state.model
    optimizer = state.optim
    teacher_forcing = True
    
    for orig, o_len, des, d_len in tqdm(data, desc=f"Epoch {state.epoch + 1}"):
        optimizer.zero_grad()

        if torch.rand(1) < 0.5: # whether teacher forcing or unconstrained mode
            teacher_forcing = True
        else:
            teacher_forcing = False
            
        orig, des = orig.to(DEVICE), des.to(DEVICE)
        output = model(orig, des, o_len, d_len, teacher_forcing=teacher_forcing) # dim = length x batch x output_dim
        loss = loss_fn(output.view(-1, output.shape[-1]), des.view(-1))
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

        state.iteration += 1
        
    return total_loss/len(data)


def test_loop(data: DataLoader, state: State, loss_fn):
    """
    This function is the testing phase in an epoch.
    Parameters:
        data: test dataloader (each example in the padded batch is a tuple of orig, len_orig, des, len_des) 
        state: checkpointing of the training process
    Return:
        average test loss in that epoch
    """
    
    total_loss = 0.
    model = state.model
    
    with torch.no_grad():
        for orig, o_len, des, d_len in data:
            orig, des = orig.to(DEVICE), des.to(DEVICE)
            output = model(orig, des, o_len, d_len) # dim = length x batch x output_dim
            total_loss += loss_fn(output.view(-1, output.shape[-1]), des.view(-1))

    return total_loss/len(data)
    
    
def run(train_loader, test_loader, model, nb_epochs, lr, prefix=''):
    """
    This function is the training process.
    """
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    writer = SummaryWriter(f'runs/{prefix}_{timestamp}')
    
    savepath = Path(f"{prefix}.pth")
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
EMBED_DIM = 200 # Here I choose the same embedding_dim for both encoder and decoder
HIDDEN_DIM = 300
NB_EPOCHS = 30
LEARNING_RATE = 1e-4
encoder = Encoder(len(vocEng), EMBED_DIM, HIDDEN_DIM)
decoder = Decoder(len(vocFra), EMBED_DIM, HIDDEN_DIM)
model = Seq2SeqTrans(encoder, decoder)
run(train_loader, test_loader, model, NB_EPOCHS, LEARNING_RATE, prefix='Translation')


# Training with segmentation
class SegmentDataset():
    def __init__(self, data, vocOrig, vocDest, max_len=10):
        """
        Parameters:
            data: a list 
            vocOrig: a `sentencepiece` segmentation model of the original vocabulary
            vocDest: a `sentencepiece` segmentation model of the destination vocabulary
        """
        self.sentences =[]
        for s in tqdm(data.split("\n")):
            if len(s) < 1: continue
            orig, dest = s.split("\t")[:2]
            if len(orig) > max_len: continue
            self.sentences.append((torch.tensor(vocOrig.encode(orig, out_type=int)+[Vocabulary.EOS]), torch.tensor(vocDest.encode(dest, out_type=int)+[Vocabulary.EOS])))
    def __len__(self): return len(self.sentences)
    def __getitem__(self,i): return self.sentences[i]

eng_process = spm.SentencePieceProcessor(model_file='eng_segment.model')
fra_process = spm.SentencePieceProcessor(model_file='fra_segment.model')

datatrain_segment = SegmentDataset("".join(lines[:idxTrain]), eng_process, fra_process, max_len=MAX_LEN)
datatest_segment = SegmentDataset("".join(lines[idxTrain:]), eng_process, fra_process, max_len=MAX_LEN)

train_loader_segment = DataLoader(datatrain_segment, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=True)
test_loader_segment = DataLoader(datatest_segment, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=False)


logging.info("Training with segmentation")
EMBED_DIM = 200 # Here I choose the same embedding_dim for both encoder and decoder
HIDDEN_DIM = 300
NB_EPOCHS = 50
LEARNING_RATE = 1e-4
encoder = Encoder(8000, EMBED_DIM, HIDDEN_DIM)
decoder = Decoder(15000, EMBED_DIM, HIDDEN_DIM)
model = Seq2SeqTrans(encoder, decoder)
run(train_loader_segment, test_loader_segment, model, NB_EPOCHS, LEARNING_RATE, prefix='Translation_segmentation')


# Visualize translation
def generate_beam(model, eos, k, orig='', maxlen=200):
    """
    This function processes an English phrase and generates its translation using beam search. 
    
    Parameters:
        orig: English phrase to be translated 
        eos : index of end-of-sequence token
        k : beam-search parameter
        maxlen: maximal length of the generated sequence
    """
    
    encoder = model.encoder
    decoder = model.decoder
    
    hidden = encoder(orig.unsqueeze(-1), torch.tensor([len(orig)])).squeeze(1) # dim = length(1) x hidden_dim
    input_seq = torch.tensor([Vocabulary.SOS], device=DEVICE, dtype=torch.long)

    topk_seqs = [input_seq] # List of tensors of one dimension
    topk_scores = torch.zeros(1) # Tensor of one dimension

    for _ in range(maxlen):
        candidates = [] # List of tuples (score (tensor of zero dimension), seq (tensor of one dimension)) 
        
        for i in range(len(topk_seqs)):
            seq = topk_seqs[i]
            score = topk_scores[i]
            
            if seq[-1] == eos:
                candidates.append((score, seq))
            else: 
                hidden = decoder.one_step(seq[-1], hidden) # dim = length(1) x hidden_dim
                output = decoder.decode(hidden).squeeze(0)
    
                log_probs = nn.functional.log_softmax(output, dim=-1)
                scores, next_tokens = log_probs.topk(k)
    
                for j, next_token in enumerate(next_tokens):
                    candidates.append((score + scores[j], torch.cat([seq, torch.tensor([next_token], device=DEVICE)])))
                
        candidates.sort(reverse=True, key=lambda x: x[0])
        topk_seqs = [c[1] for c in candidates[:k]]
        topk_scores = torch.tensor([c[0] for c in candidates[:k]])

        if all(seq[-1] == eos for seq in topk_seqs):
            break

    return topk_seqs

# Without segmentation
sentence = ['I', 'am', 'a', 'student']
eng_seq = torch.tensor([vocEng.get(o, adding=False) for o in sentence] + [Vocabulary.EOS], device=DEVICE)
state = State.load(Path("Translation.pth"))
fra_seq = generate_beam(state.model, Vocabulary.EOS, k=3, orig=eng_seq)[0]
print (f'Translation without segmentation:', vocFra.getwords(fra_seq.tolist()))

# With segmentation
eng_seq = torch.tensor(eng_process.encode('I am a student', out_type=int) + [Vocabulary.EOS], device=DEVICE)
state = State.load(Path("Translation_segmentation.pth"))
fra_seq = generate_beam(state.model, Vocabulary.EOS, k=3, orig=eng_seq)[0]
print (f'Translation with segmentation:', fra_process.decode(fra_seq.tolist()))
# END TODO
