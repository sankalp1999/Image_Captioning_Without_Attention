import torch
import torch.nn as nn
import torchvision.models as models
class EncoderCNN(nn.Module):
    def __init__(self, embed_size=256, train_CNN = False):
        super(EncoderCNN, self).__init__()
        self.train_CNN = train_CNN
        self.resnet50 = models.resnet50(pretrained=True)
        
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

class DecoderRNN(nn.Module):
    def __init__(self,embed_size, hidden_size, vocab_size, num_layers):
        super(DecoderRNN,self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size) # Embedding layer courtesy Pytorch
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size , vocab_size)
        self.dropout = nn.Dropout(0.5)

        # output from lstm is mapped to vocab size
    def forward(self,features, captions):
        embeddings = self.dropout(self.embed(captions))

        # Add an additional dimension so it's viewed as a time step, (N, M ) -- > (1, N, M) * t , t timesteps
        embeddings = torch.cat((features.unsqueeze(0), embeddings), dim = 0)
        hiddens, _ = self.lstm(embeddings)
        # Take the hidden state, _ unimportant

        outputs = self.linear(hiddens)

        return outputs

class CNNtoRNN(nn.Module):
    def __init__(self, embed_size, hidden_size,vocab_size, num_layers):
        super(CNNtoRNN, self).__init__()
        self.encoderCNN = EncoderCNN(embed_size)
        self.decoderRNN = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images, captions):
        features = self.encoderCNN(images)
        outputs = self.decoderRNN(features, captions)
        return outputs
   
    def caption_image(self, image, vocabulary, max_length = 50):
        # Getting the damn caption

        result_caption = []
        with torch.no_grad():
            x = self.encoderCNN(image).unsqueeze(0)
            states = None

            for _ in range(max_length):
                hiddens, states = self.decoderRNN.lstm(x, states)
                output = self.decoderRNN.linear(hiddens.squeeze(0))
                predicted = output.argmax(1)

                result_caption.append(predicted.item()) # item is used to get python scalar from cuda object
                x = self.decoderRNN.embed(predicted).unsqueeze(0)

                if vocabulary.itos[predicted.item()] == "<EOS>": # break when end of sequence
                    break
        return [vocabulary.itos[idx] for idx in result_caption] # returns the actual sentence



         
