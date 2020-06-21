import torch
import torch.nn as nn
import torch.utils.data as data

from utils.transforms import transform_train, transform_val
from utils.dataloader import dataloader

from models.encoders import ResNet50
from models.decoders import TextualHead


visual = ResNet50(1024)
textual = TextualHead(10000, 1024, 1, 16, 4096, 0.1, 30, 0)

visual.train()
textual.train()

dl = dataloader("val", transform_val, 32, 5, "./pickle/vocab.pkl", True)

criterion = nn.CrossEntropyLoss()

params = list(visual.parameters()) + list(textual.parameters())
optimizer = torch.optim.Adam(params=params, lr=0.01)
for iteration in range(1):

    indices = dl.dataset.get_indices()
    new_sampler = data.sampler.SubsetRandomSampler(indices=indices)
    dl.batch_sampler.sampler = new_sampler

    for batch in dl:
        images, captions = batch[0], batch[1]
        print("images.shape", images.shape)
        print("captions.shape", captions.shape)
        # print("caption: ", captions.view(-1))

    v_features = visual(images)
    t_features = textual(v_features, captions, captions.shape[1] * torch.ones(32))
    # print(t_features.view(-1, 10000))

    loss = criterion(t_features.view(-1, 10000), captions.view(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
