import torch
import torchtext
from torchtext.data import Field, RawField


def tokenize(text):
    """
    Tokenizes all words into characters, this achieves character-level tokenization
    """
    return [char for char in text]


class Data:
    BATCH_SIZE = 32

    def __init__(self, device=None):
        self.src_field = Field(tokenize=tokenize,
                               init_token='<sos>',
                               eos_token='<eos>',
                               lower=True,
                               batch_first=True)

        self.target_field = Field(tokenize=tokenize,
                                  init_token='<sos>',
                                  eos_token='<eos>',
                                  lower=True,
                                  batch_first=True)

        self.tags_field = RawField()

        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        """Returns iterators for model to run on, data sets and fields"""

        fields = {'src': ('src', self.src_field), 'trg': ('trg', self.target_field), 'tags': ('tags', self.tags_field)}
        lang = "zulu"
        self.train_data, self.valid_data, self.test_data = torchtext.data.TabularDataset.splits(
            path=f"../../Data/{lang}/json",
            train=f'{lang}-train.json',
            test=f'{lang}-test.json',
            validation=f'{lang}-valid.json',
            format='json',
            fields=fields)
        self.src_field.build_vocab(self.train_data)
        self.target_field.build_vocab(self.train_data)

        self.train_iterator, self.valid_iterator, self.test_iterator = torchtext.data.BucketIterator.splits(
            (self.train_data, self.valid_data, self.test_data),
            sort_key=lambda x: x.src,
            batch_size=Data.BATCH_SIZE,
            device=self.device)

    def get_iterators(self):
        return (self.train_data, self.train_iterator, self.valid_data, self.valid_iterator, self.test_data,
                self.test_iterator, self.src_field, self.target_field)
