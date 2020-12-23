import sys
import click
import gin
import logging
import typing as T
import itertools as it

from nltk.parse.generate import generate
from nltk import CFG
from pathlib import Path


logging.basicConfig(
  format='[%(levelname)s:%(asctime)s] %(message)s',
  datefmt='%m%d@%H%M',
  handlers=[logging.FileHandler('log.txt','a'), logging.StreamHandler(sys.stdout)],
  level=logging.INFO)
L = logging.getLogger(__name__)


import torch
from torch import nn



