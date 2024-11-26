# TODO
from pathlib import Path
import sentencepiece as spm

eng_path = Path("./english_data.txt")
fra_path = Path('./french_data.txt')

FILE = "./data/en-fra.txt"
with open(FILE) as f:
    lines = f.readlines()

with open("english_data.txt", "w") as eng_file, open("french_data.txt", "w") as fra_file:
    for line in lines:
        if len(line.split("\t")) >= 2:  
            eng, fra = line.split("\t")[:2]
            eng_file.write(eng + "\n")  
            fra_file.write(fra + "\n") 

spm.SentencePieceTrainer.Train(
    input="english_data.txt", 
    model_prefix='eng_segment',
    vocab_size=8000
)

spm.SentencePieceTrainer.Train(
    input="french_data.txt", 
    model_prefix='fra_segment',
    vocab_size=15000
)
# END TODO
