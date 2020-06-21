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
    ):
        self.transform = transform
        self.mode = mode
        self.batch_size = batch_size
        self.root_dir = root_dir
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
        # check mode
        if self.mode == "train" or self.mode == "val":
            # get the corresponding image, captions
            ann_id = self.ids[index]
            caption = self.coco.anns[ann_id]["caption"]
            image_id = self.coco.anns[ann_id]["image_id"]
            path = self.coco.loadImgs(image_id)[0]["file_name"]

            # convert image to tensor
            image = Image.open(os.path.join(self.root_dir, path)).convert("RGB")
            image = self.transform(image)

            # convert caption to tensor
            tokens = word_tokenize(str(caption).lower())
            caption = []
            caption.append(self.vocab(self.vocab.start_word))
            caption.extend([self.vocab(token) for token in tokens])
            caption.append(self.vocab(self.vocab.end_word))
            caption = torch.Tensor(caption).long()

            return image, caption

    def __len__(self):
        if self.mode == "train" or self.mode == "val":
            return len(self.ids)

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
    mode, transform, batch_size, vocab_threshold, vocab_file, from_vocab_file
):
    if mode == "train":
        assert os.path.exists(vocab_file), "Vocab file doesn't exist"
        img_dir = "../data/train2014"
        captions = "../data/annotations/captions_train2014.json"
        # annotation_file = '../data/annotations/instances_train2014.json'
    elif mode == "val":
        assert os.path.exists(vocab_file), "Vocab file doesn't exist"
        img_dir = "./data/val2014"
        captions = "./data/annotations/captions_val2014.json"
        # annotation_file = '../data/annotations/instances_val2014.json'

    dataset = COCODataset(
        transform=transform,
        mode=mode,
        batch_size=batch_size,
        root_dir=img_dir,
        annotation_file=captions,
        vocab_file=vocab_file,
        from_vocab_file=from_vocab_file,
        threshold=vocab_threshold,
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
