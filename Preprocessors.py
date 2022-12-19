import pip
pip.main(['install', '-q', 'transformers']) # Package for pretrained BERT.

import numpy as np
import torch
import torchtext
from transformers import BertTokenizerFast

class ImagePreprocessor():
    def process(self, x):
        return torch.swapaxes(torch.swapaxes(x / 255.0, 3, 2), 2, 1)

class BertPreprocessor():
    def __init__(self, max_len=50):
        self.max_len = max_len
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased", model_max_length=self.max_len)

    def process(self, x):
        padded_list = self.tokenizer.batch_encode_plus(x, padding=True)
        input_ids = torch.LongTensor(np.array(padded_list["input_ids"])) # It looks like bypassing numpy is faster.
        attention_mask = torch.LongTensor(np.array(padded_list["attention_mask"]))
        return input_ids, attention_mask

class GloVePreprocessor():
    def __init__(self, max_len=50):
        self.max_len = max_len
        self.tokenizer = torchtext.data.get_tokenizer("basic_english")
        self.GloVe = torchtext.vocab.GloVe(name="6B", dim=50)

    def process(self, x):
        tokens = [self.tokenizer(text) for text in x]
        tokens = [token + [""] * (self.max_len - len(token)) if len(token) < self.max_len else token[:self.max_len] for token in tokens]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        embeddings = torch.zeros(len(x), self.max_len, 50).to(device)
        for i, token in enumerate(tokens):
            embeddings[i] = self.GloVe.get_vecs_by_tokens(token, lower_case_backup=True)
        return embeddings, None

def Preprocessor(name, max_len=70):
    if name in ["InceptionV3", "Xception", "ResNet"]:
        return ImagePreprocessor()
    elif name in ["BERT"]:
        return BertPreprocessor(max_len)
    elif name in ["GloVe"]:
        return GloVePreprocessor(max_len)
    else:
        raise ValueError(name + " has not been implemented!")
