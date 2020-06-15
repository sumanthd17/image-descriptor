import sys

sys.path.append("../")

from utils.transforms import transform_val
from utils.dataloader import dataloader

import torch.utils.data as data

MODE = "val"
TRANSFORM = transform_val
BATCH_SIZE = 10
VOCAB_THRESHOLD = 5
VOCAB_FILE = "../pickle/vocab.pkl"
FROM_VOCAB_FILE = True

data_loader = dataloader(
    mode=MODE,
    transform=TRANSFORM,
    batch_size=BATCH_SIZE,
    vocab_threshold=VOCAB_THRESHOLD,
    vocab_file=VOCAB_FILE,
    from_vocab_file=FROM_VOCAB_FILE,
)

indices = data_loader.dataset.get_indices()
print('{} sampled indices: {}'.format(len(indices), indices))
new_sampler = data.sampler.SubsetRandomSampler(indices=indices)
data_loader.batch_sampler.sampler = new_sampler

for batch in data_loader:
  images, captions = batch[0], batch[1]
  break

print('images.shape', images.shape)
print('captions.shape', captions.shape)