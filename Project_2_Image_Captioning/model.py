import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, drop_prob=0.3):

        super(DecoderRNN, self).__init__()

        self.output_size = vocab_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, dropout=drop_prob, batch_first=True)
                
        self.fc = nn.Linear(hidden_size, self.output_size)
    
    def forward(self, features, captions):
        
        batch_size = features.size(0)

        hidden = self.init_hidden(batch_size)
        
        captions = captions[:,:-1].long()
                
        embeds = self.embedding(captions)
        
        
        embeds = torch.cat((features.unsqueeze(1), embeds), dim=1)
        
        out, hidden = self.lstm(embeds, hidden)

        out = out.contiguous().view(-1, self.hidden_size)

        out = self.fc(out)
        
        out = out.view(batch_size, -1, self.output_size)

        return out

    def init_hidden(self, batch_size):

        weight = next(self.parameters()).data

        hidden = (weight.new(self.num_layers, batch_size, self.hidden_size).zero_().cuda(),
              weight.new(self.num_layers, batch_size, self.hidden_size).zero_().cuda())


        return hidden

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        
        batch_size = inputs.size(0)
        
        if states==None: 
            states = self.init_hidden(batch_size)

        output_list = []
        last_int = 0
        for i in range(max_len):
            embeds = 0
            if i == 0:
                embeds = inputs
            else:
                embeds = self.embedding(torch.tensor(last_int).cuda())
                embeds = embeds.unsqueeze(0).unsqueeze(0)
            
            out, states = self.lstm(embeds, states)
                
            out = out.contiguous().view(-1, self.hidden_size)
            out = self.fc(out)
            out = out.view(batch_size, -1, self.output_size)
            _, pred = torch.max(out, 2)
            pred.view(batch_size)
            
            last_int = pred.item()
            output_list.append(last_int)
            
            if last_int == 1:
                break
        
        return output_list
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        