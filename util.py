
import click
import gin

import sys
import json
import logging
import itertools as it

import torch
import typing as T

from types import SimpleNamespace
from pathlib import Path


INF = float('inf')

logconf = dict(
  format='[%(levelname)s:%(asctime)s] %(message)s',
  datefmt='%m%d@%H%M',
  handlers=[logging.FileHandler('log.txt','a'), logging.StreamHandler(sys.stdout)],
  level=logging.INFO)

type = SimpleNamespace(
  TFile=T.Union[str, Path],
)


click_helper = SimpleNamespace(
  with_path=lambda option_name: click.option(option_name,
    type=click.Path(exists=True), expose_value=True, required=True,
    callback=lambda _c, _p, loc: Path(loc)),
  with_gin_config=lambda required=False, expose=False: click.option('--config',
    type=click.Path(exists=True, dir_okay=False), envvar='GIN_CONFIG',
    expose_value=expose, required=required,
    callback=lambda _c, _p, cfg: gin.parse_config_file(cfg))
)


def device(prefer='cuda'):
  return torch.device(prefer) if torch.cuda.is_available() else torch.device('cpu')

def jsondump(obj, fname: type.TFile):
  with open(fname, 'w') as f:
    json.dump(obj, f)

def jsonload(fname: type.TFile):
  with open(fname, 'r') as f:
    return json.load(f)


class Vocab:
  PAD = '<pad>'
  SPECIAL = {PAD}
  SEPARATOR='\t'

  @classmethod
  def build(cls, tokenized_sents: T.List[str], special:set=SPECIAL):
    words = it.chain(*[tokens for tokens in tokenized_sents])
    words = {w:i for i, w in enumerate(set(words), start=len(special))}
    if special.intersection(words.keys()):
      raise Exception(f'remove reserved keywords from tokens ({special})')
    for i, w in enumerate(special):
      words[w] = i
    return cls(words=words)

  @classmethod
  def load(cls, path: type.TFile):
    special = set()
    words = {}
    with open(path, 'r') as f:
      for line in map(str.strip, f):  # load specials until first empty line
        if not line: break
        word, idx = line.split(cls.SEPARATOR)
        words[word] = int(idx)
        special.add(word)
      for line in map(str.strip, f):  # load the rest
        word, idx = line.strip().split(cls.SEPARATOR)
        words[word] = int(idx)
    return cls(words=words, special=special)

  def __init__(self, words:dict=None, special:set=SPECIAL):
    if words is not None and not all(k in words for k in special):
      raise Exception('All specials should be in the word dictionary.')
    self.special = special
    self.words = words or {k:i for i, k in enumerate(self.special)}

  def _new(self):
    return len(self.words)

  def add(self, word):
    if word not in self.words:
      self.words[word] = self._new()
    return self.words[word]

  def get(self, word):
    return self.words[word]

  def __getitem__(self, word):
    return self.get(word)

  def __len__(self):
    return len(self.words)

  def encode(self, sequence: T.List[T.List[str]], max_length=None):
    sequence = [[self.get(w) for w in s.split()] for s in sequence]
    if max_length is None:
      max_length = max([len(s) for s in sequence]) + 1
    padding = lambda s: max_length - 1 - len(s)
    return torch.stack([
      torch.tensor(s + [self.get(self.PAD)] * padding(s))
      for s in sequence
    ])

  def save(self, output_path: type.TFile):
    with open(output_path, 'w') as f:
      for word in self.special:
        f.write(f'{word}{self.SEPARATOR}{self.words[word]}\n')
      f.write('\n')
      for word, idx in self.words.items():
        if word not in self.special:
          f.write(f'{word}{self.SEPARATOR}{idx}\n')

