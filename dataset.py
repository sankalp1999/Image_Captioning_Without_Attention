import os
import pandas as pd
import spacy # for tokenizer
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from vocab import Vocab_Builder



class FlickrDataset(Dataset):
    def __init__(self, root_dir, captions_file, transform = None, freq_threshold = 5):
        self.root_dir = root_dir
        self.df = pd.read_csv(captions_file)
        self.transform = transform
        # Get images, caption column from pandas
        self.imgs = self.df["image"]
        self.captions = self.df["caption"]

        #Init and Build vocab
        self.vocab = Vocab_Builder(freq_threshold) # freq threshold is experimental
        self.vocab.build_vocabulary(self.captions.tolist())

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index: int):
        
        caption = self.captions[index]
        img_id = self.imgs[index]
        img = Image.open(os.path.join(self.root_dir, img_id)).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        
        numericalized_caption = [self.vocab.stoi["<SOS>"]] #stoi is string to index, start of sentence
        numericalized_caption += self.vocab.numericalize(caption) # Convert each word to a number in our vocab
        numericalized_caption.append(self.vocab.stoi["<EOS>"])

        #return tensor
        temp = torch.tensor(numericalized_caption)
        return img, torch.tensor(numericalized_caption)
    
    @staticmethod
    def evaluation(self, index : int):
        caption = self.captions[index]
        img_id = self.imgs[index]
        img = Image.open(os.path.join(self.root_dir, img_id)).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)
        return img, caption
# Caption lengths will be different, in our batch all have to be same length


'''
Goes to the dataloader
'''
class Collate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=False, padding_value=self.pad_idx)

        return imgs, targets

# caption file, Maybe change num_workers

def get_loader( root_folder,annotation_file,  transform, batch_size = 32,  num_workers = 8, shuffle = True, pin_memory = False):
    


    dataset =  FlickrDataset(root_folder,  annotation_file, transform = transform)
    pad_idx = dataset.vocab.stoi["<PAD>"]

    loader = DataLoader(
        dataset = dataset,
        batch_size = 1,
        num_workers = num_workers,
        shuffle = shuffle,
        pin_memory = pin_memory,
        collate_fn =  Collate(pad_idx = pad_idx)
    )

    return loader, dataset

# print(len(dataset.vocab))