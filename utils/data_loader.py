from torchtext.datasets import Multi30k
from torchtext import data


class DataLoader:
    source = None
    target = None

    def __init__(self, ext, tokenize_en, tokenize_de, init_token, eos_token):
        self.ext = ext
        self.tokenize_en = tokenize_en
        self.tokenize_de = tokenize_de
        self.init_token = init_token
        self.eos_token = eos_token
        print("Dataset initialize start")

    def make_dataset(self):
        print("Dataset initialize start")
        if self.ext == ('.de', '.en'):
            self.source = data.Field(tokenize=self.tokenize_de,
                                     init_token=self.init_token,
                                     eos_token=self.eos_token,
                                     lower=True,
                                     batch_first=True)
            self.target = data.Field(tokenize=self.tokenize_en,
                                     init_token=self.init_token,
                                     eos_token=self.eos_token,
                                     lower=True,
                                     batch_first=True)
        elif self.ext == ('.en', '.de'):
            self.source = data.Field(tokenize=self.tokenize_en,
                                     init_token=self.init_token,
                                     eos_token=self.eos_token,
                                     lower=True,
                                     batch_first=True)
            self.target = data.Field(tokenize=self.tokenize_de,
                                     init_token=self.init_token,
                                     eos_token=self.eos_token,
                                     lower=True,
                                     batch_first=True)
        train_data, valid_data, test_data = Multi30k.splits(exts=self.ext, fields=(self.source, self.target))
        return train_data, valid_data, test_data

    def build_vocab(self, train_data, min_freq):
        self.source.build_vocab(train_data, min_freq=min_freq)
        self.target.build_vocab(train_data, min_freq=min_freq)

    def make_iter(self, train, validate, test, batch_size, device):
        train_iteration, valid_iteration, test_iteration = data.BucketIterator.splits((train, validate, test),
                                                                                      batch_size=batch_size,
                                                                                      device=device)
        print("dataset initializing done!!!")
        return train_iteration, valid_iteration, test_iteration
