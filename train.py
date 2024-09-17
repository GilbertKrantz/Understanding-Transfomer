import torch
import torch.nn as nn
from tqdm import tqdm

from Models.Transformer import Transformer

class Trainer:
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout, device, validate=False):
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        
        self.device = device
        self.model = Transformer(self.src_vocab_size, self.tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        
    def train(self, src_data, tgt_data):
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(src_data, tgt_data[:, :-1])
        output =  output.contiguous().view(-1, self.tgt_vocab_size)
        tgt_data_shifted = tgt_data[:, 1:].contiguous().view(-1)
        loss = self.criterion(output, tgt_data_shifted) # Train Loss
        loss.backward()
        self.optimizer.step()         
        return loss.item()
    
    def validate(self, val_src_data, val_tgt_data):
        self.model.eval()
        with torch.no_grad():
            output = self.model(val_src_data, val_tgt_data[:, :-1])
            output = output.contiguous().view(-1, self.tgt_vocab_size)
            tgt_data_shifted = val_tgt_data[:, 1:].contiguous().view(-1)
            loss = self.criterion(output, tgt_data_shifted) # Validation Loss
        return loss.item()   
    
    def model_train(self, epochs, src_data, tgt_data, val_src_data=None, val_tgt_data=None):
        src_data = src_data.to(self.device)
        tgt_data = tgt_data.to(self.device)
        
        if val_src_data is not None:
            val_src_data = val_src_data.to(self.device)
            val_tgt_data = val_tgt_data.to(self.device)
            
        for epoch in tqdm(range(epochs)):
            train_loss = self.train(src_data=src_data, tgt_data=tgt_data)
            if val_src_data is not None:
                val_loss = self.validate(val_tgt_data=val_tgt_data, val_src_data=val_src_data)
            print(f"Epoch: {epoch+1}, Train Loss: {train_loss}",  end="")
            if val_src_data is not None:
                print(f", Validation Loss: {val_loss}")
            else:
                print()
        
    @staticmethod  
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
            
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        
    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
    