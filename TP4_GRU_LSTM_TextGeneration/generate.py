from textloader import  string2code, id2lettre, code2string
import math
import torch
import torch.nn as nn

# TODO
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate(rnn, eos, start="", maxlen=200, random=False):
    """  
    This function generates a sequence, starting either from «start» or from 0 if «start» is empty. The sequence has a maximum length of maxlen, or it stops when eos is generated. At each time step, it selects the next character either by choosing the most probable one or by sampling according to the distribution.
    
    Parameters:
        eos : index of end of sequence token
        start: beginning of the sentence
        maxlen: maximal length
        random: whether a random or determinist generation
    """
    
    if len(start) > 0:  
        input_seq = string2code(start).unsqueeze(-1).to(DEVICE) # dim = length x batch(1)
    else:
        input_seq = torch.zeros((1, 1), device=DEVICE, dtype=torch.long) # dim = length(1) x batch(1)

    output_seq = ''

    while len(output_seq) < maxlen:
        embedded_input = rnn.embedding(input_seq) # dim = length x batch x embedding_dim
        hidden = rnn(embedded_input)[-1] # dim = batch x latent_dim
        output = rnn.decode(hidden) # dim = batch x output_dim

        if random:
            probs = nn.Softmax(dim=-1)(output)
            next_token = torch.distributions.Categorical(probs).sample().item()
        else:
            next_token = torch.argmax(output, dim=-1).item()
        output_seq += id2lettre[next_token]

        if next_token == eos:
            break

        input_seq = torch.tensor([next_token], device=DEVICE).reshape(1, 1)
    
    return output_seq


def generate_beam(rnn, eos, k, start="", maxlen=200):
    """
    This function generates a sequence using beam search, starting either from «start» or from 0 if «start» is empty. The sequence has a maximum length of maxlen, or it stops when eos is generated. At each time step, for each candidate sequence, k most probable symbols are selected. Then, only the k best candidates (in terms of likelihood) are retained for the next iteration.
    
    Parameters:
        eos : index of end of sequence token
        k : beam-search parameter
        start: beginning of the sentence
        maxlen: maximal length
    """

    if len(start) > 0:  
        input_seq = string2code(start).to(DEVICE) 
    else:
        input_seq = torch.zeros(1, device=DEVICE, dtype=torch.long)

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
                embed_seq = rnn.embedding(seq.unsqueeze(-1))
                hidden = rnn(embed_seq)[-1]
                output = rnn.decode(hidden).squeeze(0)
    
                log_probs = nn.functional.log_softmax(output, dim=-1)
                scores, next_tokens = log_probs.topk(k)
    
                for j, next_token in enumerate(next_tokens):
                    candidates.append((score + scores[j], torch.cat([seq, torch.tensor([next_token], device=DEVICE)])))
                
        candidates.sort(reverse=True, key=lambda x: x[0])
        topk_seqs = [c[1] for c in candidates[:k]]
        topk_scores = torch.tensor([c[0] for c in candidates[:k]])

        if all(seq[-1] == eos for seq in topk_seqs):
            break

    topk_output = [code2string(seq) for seq in topk_seqs]
    return topk_output


def p_nucleus(decoder, alpha: float):
    """
    Returns a function that computes the probability distribution over the smallest set of outputs whose cumulative probability covers at least the probability mass alpha.

    Parameters:
        decoder: RNN's decoder
        alpha: probability mass to cover
    """
    
    def compute(h):
        """
        Parameters:
            h (torch.Tensor): Hidden state to decode
        """
        
        output = decoder(h).squeeze(0)
        probs = nn.Softmax(dim=-1)(output)
        probs_descend, indices_descend = probs.sort(descending=True)
        
        cumsum_probs = torch.cumsum(probs_descend, dim=0)
        index_threshold = torch.argmax((cumsum_probs >= alpha).int()).item()
        
        probs = probs_descend[:(index_threshold+1)]
        return probs/probs.sum(), indices_descend[:(index_threshold+1)]
        
    return compute


def generate_beam_with_p_nucleus(rnn, eos, alpha, k, start="", maxlen=200):
    """
    This function generates a sequence using beam search, starting either from «start» or from 0 if «start» is empty. The sequence has a maximum length of maxlen, or it stops when eos is generated. At each time step, for each candidate sequence, probability distribution over the outputs is calculated using Nucleus sampling. From this distribution, the k most probable symbols are selected. Then, only the k best candidates (in terms of likelihood) are retained for the next iteration.
    
    Parameters:
        eos : index of end of sequence token
        alpha: probability mass to cover
        k : the beam-search parameter
        start: beginning of the sentence
        maxlen: maximal length
    """
    
    p_nucleus_sampling = p_nucleus(rnn.decode, alpha)
    
    if len(start) > 0:  
        input_seq = string2code(start).to(DEVICE) 
    else:
        input_seq = torch.zeros(1, device=DEVICE, dtype=torch.long)

    topk_seqs = [input_seq] 
    topk_scores = torch.zeros(1) 

    for _ in range(maxlen):
        candidates = []  
        
        for i in range(len(topk_seqs)):
            seq = topk_seqs[i]
            score = topk_scores[i]
            
            if seq[-1] == eos:
                candidates.append((score, seq))
            else:  
                embed_seq = rnn.embedding(seq.unsqueeze(-1))
                hidden = rnn(embed_seq)[-1]
                probs, indices = p_nucleus_sampling(hidden)
    
                log_probs = torch.log(probs)
                scores, next_tokens = log_probs[:k], indices[:k]
    
                for j, next_token in enumerate(next_tokens):
                    candidates.append((score + scores[j], torch.cat([seq, torch.tensor([next_token], device=DEVICE)])))
                
        candidates.sort(reverse=True, key=lambda x: x[0])
        topk_seqs = [c[1] for c in candidates[:k]]
        topk_scores = torch.tensor([c[0] for c in candidates[:k]])

        if all(seq[-1] == eos for seq in topk_seqs):
            break
            
    topk_output = [code2string(seq) for seq in topk_seqs]
    
    return topk_output
