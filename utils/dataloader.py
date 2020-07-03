from torch.utils.data import Dataset
from pycocotools.coco import COCO
from nltk.tokenize import word_tokenize
import numpy as np
from tqdm import tqdm
from PIL import Image
import os
from .vocabulary import Vocabulary
import torch.utils.data as data
import torch
import requests
from io import BytesIO


class COCODataset(Dataset):
    def __init__(
        self,
        transform,
        mode,
        batch_size,
        root_dir,
        annotation_file,
        vocab_file,
        from_vocab_file,
        threshold,
        data_url,
        max_cap_len,
        model="attention",
    ):
        self.transform = transform
        self.mode = mode
        self.batch_size = batch_size
        self.root_dir = root_dir
        self.data_url = data_url
        self.model = model
        self.max_cap_len = max_cap_len
        self.vocab = Vocabulary(threshold, vocab_file, from_vocab_file, annotation_file)

        if self.mode == "train" or self.mode == "val":
            self.coco = COCO(annotation_file)
            self.ids = list(self.coco.anns.keys())
            all_tokens = [
                word_tokenize(str(self.coco.anns[self.ids[index]]["caption"]).lower())
                for index in tqdm(np.arange(len(self.ids)))
            ]
            self.caption_lengths = [len(token) for token in all_tokens]

    def __getitem__(self, index):
        # get the corresponding image, captions
        ann_id = self.ids[index]
        caption = self.coco.anns[ann_id]["caption"]
        image_id = self.coco.anns[ann_id]["image_id"]
        path = self.coco.loadImgs(image_id)[0]["file_name"]

        # convert image to tensor
        if self.data_url:
            response = requests.get(self.data_url + path)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            image = Image.open(os.path.join(self.root_dir, path)).convert("RGB")

        image = self.transform(image)

        caption, caplen = self.encode_captions(caption)

        caplen = torch.Tensor([caplen]).long()
        caption = torch.Tensor(caption).long()

        all_captions = []
        if self.mode == "val":
            # all caption ids corresponding to the image
            all_captions = self.coco.getAnnIds(imgIds=[image_id])
            all_captions = [self.coco.anns[i]["caption"] for i in all_captions]
            all_captions = [self.encode_captions(i, True)[0] for i in all_captions]
            all_captions = np.array(all_captions)

        return image, caption, caplen, all_captions

    def __len__(self):
        if self.mode == "train" or self.mode == "val":
            return len(self.ids)

    def encode_captions(self, caption, flag=False):
        tokens = word_tokenize(str(caption).lower())
        caption = []
        caption.append(self.vocab(self.vocab.start_word))
        caption.extend([self.vocab(token) for token in tokens])
        caption.append(self.vocab(self.vocab.end_word))
        if flag or self.model == "attention":
            caption.extend(
                [self.vocab(self.vocab.pad_word)] * (self.max_cap_len - len(tokens))
            )
        return caption, len(tokens) + 2

    def get_indices(self):
        length = np.random.choice(self.caption_lengths)
        all_indices = np.where(
            [
                self.caption_lengths[i] == length
                for i in np.arange(len(self.caption_lengths))
            ]
        )[0]
        return list(np.random.choice(all_indices, size=self.batch_size))


def dataloader(
    mode,
    transform,
    batch_size,
    vocab_threshold,
    vocab_file,
    from_vocab_file,
    data_path,
    image_data_unavailable=True,
    model="attention",
    max_cap_len=30,
):
    if from_vocab_file:
        assert os.path.exists(
            vocab_file
        ), "Vocab file doesn't exist. Set from_vocab_file=False"

    img_dir = data_path + "/images/{}2014".format(mode)
    data_url = (
        "http://images.cocodataset.org/{}2014/".format(mode)
        if (image_data_unavailable)
        else None
    )
    captions = data_path + "/annotations/captions_{}2014.json".format(mode)

    dataset = COCODataset(
        transform=transform,
        mode=mode,
        batch_size=batch_size,
        root_dir=img_dir,
        annotation_file=captions,
        vocab_file=vocab_file,
        from_vocab_file=from_vocab_file,
        threshold=vocab_threshold,
        data_url=data_url,
        max_cap_len=max_cap_len,
        model=model,
    )

    if mode == "train":
        indices = dataset.get_indices()
        sampler = data.sampler.SubsetRandomSampler(indices=indices)
        data_loader = data.DataLoader(
            dataset=dataset,
            batch_sampler=data.sampler.BatchSampler(
                sampler=sampler, batch_size=batch_size, drop_last=False
            ),
        )
    elif mode == "val":
        data_loader = data.DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=True
        )

    return data_loader
