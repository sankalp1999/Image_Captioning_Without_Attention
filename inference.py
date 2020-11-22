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


device = 'cpu'

mean = [0.485, 0.456, 0.406]

std = [0.229, 0.224, 0.225]

transform = transforms.Compose(
    [transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)]
)

class EncoderCNN(nn.Module):

    def __init__(self, embed_size=256, train_CNN = False):
        super(EncoderCNN, self).__init__()
        self.train_CNN = train_CNN
        self.resnet50 = models.resnet50(pretrained=False)
        self.resnet_path = './weights/resnet50.pt'
        self.resnet50 = self.resnet50.load_state_dict( torch.load(self.resnet_path, map_location = 'cpu') )
        
        for name, param in self.resnet50.named_parameters():  
            if "fc.weight" in name or "fc.bias" in name:
                param.requires_grad = True
            else:
                param.requires_grad = self.train_CNN
        
        in_features = self.resnet50.fc.in_features
        
        modules = list(self.resnet50.children())[:-1]
        self.resnet50 = nn.Sequential(*modules)
        
        self.fc = nn.Linear(in_features,embed_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, images):
        # Fine tuning, we don't want to train 
        features = self.resnet50(images)
        features = features.view(features.size(0), -1)
        features = self.fc(features)
        return features

def load_checkpoint(checkpoint, model, optimizer):
    
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    step = checkpoint["step"]
    return step


# vocab = Vocab_Builder(freq_threshold = )

# Load the pickle dump
vocab_path = 'vocab.pickle'

with open(vocab_path, 'rb') as f:
    vocab = pickle.load(f)

load_model = True
embed_size = 256
hidden_size = 256
vocab_size = len(vocab)
num_layers = 2
learning_rate = 3e-4
print(len(vocab))

model_path = './weights/my_checkpoint2.pth.tar'

model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)

criterion = nn.CrossEntropyLoss(ignore_index = vocab.stoi["<PAD>"])
optimizer = optim.Adam(model.parameters(), lr = learning_rate)
if load_model:
    step = load_checkpoint(torch.load(model_path ,map_location = 'cpu'), model, optimizer)

model.eval()

# image_path = 'flickr8k/Images/54501196_a9ac9d66f2.jpg'
image_path = './test_examples/child.jpg'

img = PIL.Image.open(image_path).convert("RGB")
img.show()

img_t = transform(img)

caps = model.caption_image(img_t.unsqueeze(0), vocab)
# print(caps)
caps = caps[1:-1]

caption = ' '.join(caps)

print(caption)










