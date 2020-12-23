import sys
import click
import gin
import logging
import typing as T
import itertools as it

import torch

from torch import nn
from torch.utils import data

from pathlib import Path


logging.basicConfig(
  format='[%(levelname)s:%(asctime)s] %(message)s',
  datefmt='%m%d@%H%M',
  handlers=[logging.FileHandler('log.txt','a'), logging.StreamHandler(sys.stdout)],
  level=logging.INFO)
L = logging.getLogger(__name__)

data_dir = Path('data')


'''def collate_fn(batch):
  texts, labels = [], []
  for s1, s2, label in batch.split('\t'):
    texts.append(txt)
    labels.append(label)
  return texts, labels'''


class SentenceData(data.Dataset):
  def __init__(self, loc):
    with open(loc) as f:
      self.data = [(s1,s2,int(l)) for s1,s2,l in map(lambda x: x.split('\t'), f)]
  def __len__(self):
    return len(self.data)
  def __getitem__(self, i):
    return self.data[i]

dataset = SentenceData(data_dir/'train.tsv')
dataloader = data.DataLoader(
    dataset,
    batch_size=8,)
    #collate_fn=collate_fn)
for idx, (s1, s2, label) in enumerate(dataloader):
    print(idx, s1, s2, label)
    break


''' LEGACY:

TEXT = data.Field()
LABELS = data.Field()

fields = [
  ('sent1', TEXT),
  ('sent2', TEXT),
  ('labels', LABELS)
]

train, test = data.TabularDataset.splits(
    path=data_dir,
    train='train.tsv',
    #validation='_dev.tsv',
    test='test.tsv',
    format='tsv',
    fields=fields)

train_iter, test_iter = data.BucketIterator.splits(
    (train, test),
    batch_sizes=(16, 256),
    sort_key=lambda x: len(x.text),
    device=0)

TEXT.build_vocab(train)
LABELS.build_vocab(train)

#dataset = data.TabularDataset(
#  path=data_dir / 'train.tsv',
#  format='tsv',
'''


