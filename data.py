from torch.utils import data


def data_splitter(line):
  s1, s2, label = line.split('\t')
  return s1, s2, int(label)


class SentPairData(data.Dataset):
  def __init__(self, loc):
    with open(loc) as f:
      self.data = list(map(data_splitter, f))

  def __len__(self):
    return len(self.data)

  def __getitem__(self, i):
    return self.data[i]


class SentPairStream(data.IterableDataset):
  def __init__(self, loc):
    self.loc = loc

  def __iter__(self):
    worker_info = data.get_worker_info()
    if worker_info is not None:  # in a worker process
      raise Exception('Not intended for multiprocess')
    return map(data_splitter, self.loc)

  def __enter__(self):
    self.loc = open(self.loc, 'r')
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    self.loc.close()
