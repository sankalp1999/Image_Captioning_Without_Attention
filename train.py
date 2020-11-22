import pickle
import PIL
from vocab import Vocab_Builder
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision.models as models
from model import DecoderRNN, CNNtoRNN
from dataset import get_loader
from utils import show_image


def train():

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = transforms.Compose(
        [transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)]
    )
    data_location = './flickr8k'
    train_loader, dataset = get_loader(
        root_folder = data_location+"/Images",
        annotation_file = data_location+"/captions.txt",
        transform = transform, 
        num_workers = 4,
    )
    torch.backends.cudnn.benchmark = True # Get some boost probaby
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = 'cpu'
    load_model = False
    save_model = False
    train_CNN = False
    #Hyperparameters
    embed_size = 256
    hidden_size = 256
    vocab_size = len(dataset.vocab)
    num_layers = 2
    learning_rate = 3e-4
    num_epochs = 20

    
    step = 0
    # init model, loss
    model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    
    if load_model:
        step = load_checkpoint(torch.load("../input/checkpoint2-epoch20/my_checkpoint2.pth.tar",map_location = 'cpu'), model, optimizer)

    model.train()
    wanna_print = 100

    for epoch in range(num_epochs):

        if save_model:
            checkpoint = {
                "state_dict" : model.state_dict(),
                "optimizer" : optimizer.state_dict(),
                "step" : step
            }
            save_checkpoint(checkpoint)

        for idx, (imgs, captions) in enumerate(train_loader):
            
         
            imgs = imgs.to(device)
            captions = captions.to(device)

            # Don't pass the <EOS>
            outputs = model(imgs, captions[:-1])

            # loss accepts only 2 dimension
            # seq_len, N, vocabulary_size --> (seq_len, N) Each time as its own example

            print("Outputs shape ", outputs.shape)
            
            loss = criterion(outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1))

            print("Step", idx, loss.item())

            step += 1

            optimizer.zero_grad()
            
            loss.backward(loss)
            
            optimizer.step()
            
            if (idx+1)%wanna_print == 0:
                print("Epoch: {} loss: {:.5f}".format(epoch,loss.item()))


                #generate the caption
                model.eval()
                with torch.no_grad():
                    dataiter = iter(train_loader)
                    img,_ = next(dataiter)
                    print(img[0].shape)
                    caps = model.caption_image(img[0:1].to(device),vocabulary=dataset.vocab)
                    caption = ' '.join(caps)
                    show_image(img[0],title=caption)
                model.train()
if __name__ == "__main__":
    train()
