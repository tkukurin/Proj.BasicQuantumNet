
import click
import gin

import sys
import logging
import itertools as it

import torch
import typing as T

from types import SimpleNamespace
from pathlib import Path
from collections import Counter


logconf = dict(
  format='[%(levelname)s:%(asctime)s] %(message)s',
  datefmt='%m%d@%H%M',
  handlers=[logging.FileHandler('log.txt','a'), logging.StreamHandler(sys.stdout)],
  level=logging.INFO)


type = SimpleNamespace(
  TFile=T.Union[str, Path]
)


def run_with_gin_config(f: T.Callable, required=True):
  @click.command(help='Run program main, loading provided GIN_CONFIG file.')
  @click.argument(
    'config', type=click.Path(exists=True, dir_okay=False), envvar='GIN_CONFIG',
    required=required)
  def cli(config):
    if config: gin.parse_config_file(config)
    f()
  return cli()


class Vocab:
  S = '<S>'
  PAD = '<pad>'
  RESERVED = set([S, PAD])

  @classmethod  # TODO from_words + save() on instance
  def save(cls, output_path: type.TFile, tokenized_sents: T.List[str], separator: str):
    words = Counter(it.chain(*[tokens for tokens in tokenized_sents]))
    if cls.RESERVED.intersection(words):
      raise Exception(f'Please remove reserved keywords from vocab ({cls.RESERVED})')
    with open(output_path, 'w') as f:
      for word, count in words.most_common():
        f.write(f'{word}{separator}{count}\n')

  @classmethod
  def load(cls, path: type.TFile, hidden_dim):
    vocab = cls(dim=(hidden_dim, hidden_dim))
    with open(path, 'r') as f:
      for line in f:
        word, _ = line.strip().split('\t')
        vocab.add(word)
    return vocab

  @classmethod
  def build(cls, dataset, hidden_dim):
    vocab = cls(dim=(hidden_dim, hidden_dim))
    for i in range(len(dataset)):
      sent1, sent2, _ = dataset[i]
      for word in sent1.split():
        vocab.add(word)
      for word in sent2.split():
        vocab.add(word)
    return vocab

  def __init__(self, dim):
    self.dim = dim
    self.words = {}
    for s in self.RESERVED:
      self.words[s] = torch.ones(*dim) #self._newvec((dim[0], 1))

  def _newvec(self, dim):
    return torch.randn(*dim)

  def add(self, word):
    if word not in self.words:
      self.words[word] = self._newvec(self.dim)
    return self.words[word]

  def get(self, word):
    return self.words[word]

  def __getitem__(self, word):
    return self.get(word)

  def encode(self, sequence: T.List[T.List[str]], max_length=None):
    sequence = [[self.get(w) for w in s.split()] for s in sequence]
    if max_length is None:
      max_length = max([len(s) for s in sequence]) + 1
    padding = lambda s: max_length - 1 - len(s)
    return torch.stack([
      torch.stack([self.get(self.S)] + s + [self.get(self.PAD)] * padding(s))
      for s in sequence
    ])

