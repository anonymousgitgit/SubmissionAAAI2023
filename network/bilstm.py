import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer, decoders
import random
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import Sequence, Strip, Lowercase, NFD, StripAccents

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(input_dim, emb_dim)
        
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout = dropout, batch_first = True, bidirectional = True)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        
        #src = [src len, batch size]
        
        embedded = self.dropout(self.embedding(src.long()))
        
        #embedded = [src len, batch size, emb dim]
        
        outputs, (hidden, cell) = self.rnn(embedded)
        
        #outputs = [src len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #outputs are always from the top hidden layer
        
        return hidden, cell
class ConcatAttn(nn.Module):
    '''Attention(concat)
    Params:
        hidden_size: hidden size
    '''
    def __init__(self, hidden_size):
        super(ConcatAttn, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(2 * hidden_size, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1.0 / sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)
    
    def forward(self, hidden, encoder_output):
        '''
        Inputs:
            hidden: [1*B*H] 
            encoder_output: [T*B*H]
        Outputs:
            energy: normalised weights [B*1*T]
        '''
        # Expand hidden [1*B*H] -> [T*B*H] -> [B*T*H]
        hidden = hidden.repeat(encoder_output.size(0), 1, 1).transpose(0, 1)

        # Transfer encoder_output to [B*T*H]
        encoder_output = encoder_output.transpose(0, 1)

        # Calculate energy and normalise  [B*1*T]
        attn_energy = self.score(hidden, encoder_output)
        return F.softmax(attn_energy, dim=2)

    def score(self, hidden, encoder_output):
        '''
        Inputs:
            hidden: [B*T*H]
            encoder_output: [B*T*H]
        Outputs:
            attn_energy: weights [B*T]
        '''
        # Project vectors [B*T*2H] -> [B*T*H] -> [B*H*T]
        energy = self.attn(torch.cat([hidden, encoder_output], 2))
        energy = energy.transpose(1, 2)
        
        # Expend v  [H] -> [B*H] -> [B*1*H]
        v = self.v.repeat(encoder_output.size(0), 1).unsqueeze(1)
        
        # [B*1*H] * [B*H*T] -> [B*1*T]
        attn_energy = torch.bmm(v, energy)
        return attn_energy

        
class BilinearAttn(nn.Module):
    '''Attention(bilinear)
    Params:
        hidden_size: hidden size
    '''
    def __init__(self, hidden_size):
        super(BilinearAttn, self).__init__()
        self.hidden_size = hidden_size
        self.bilinear = nn.Linear(hidden_size, hidden_size)

    
    def forward(self, hidden, encoder_output):
        '''
        Inputs:
            hidden: [1*B*H] 
            encoder_output: [T*B*H]
        Outputs:
            energy: normalised weights [B*1*T]
        '''
        # [T*B*H] -> [T*B*H] -> [B*H*T]
        wh = self.bilinear(encoder_output).permute(1, 2, 0)
        
        # [1*B*H] -> [B*1*H] x [B*H*T] => [B*1*T]
        score = hidden.transpose(0, 1).bmm(wh)
        
        return F.softmax(score, dim=2)
    

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout = dropout, batch_first = True)
        self.attention = BilinearAttn(hid_dim)
        
        self.linear = nn.Linear(hid_dim * 2, hid_dim * 2)
        self.fc_out = nn.Linear(hid_dim * 2, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, cell, encoder_output):
        
        embedded = self.dropout(self.embedding(input.long()))
        
        #embedded = [1, batch size, emb dim]
                
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        # print(hidden.shape)
        ht = hidden[-1]
        attn_weights = self.attention(ht.unsqueeze(0), encoder_output)
        c = attn_weights.bmm(encoder_output.transpose(0, 1)).squeeze(1)
        # concat c and h => [B*2H] => [B*H] 
        attn_vector = torch.tanh(self.linear(
            torch.cat([c, ht], dim=1)
        )).unsqueeze(1)
        
        # [B*H] -> [B*O]
        prediction = self.fc_out(attn_vector)
        
        #prediction = [batch size, output dim]
        
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        
        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"
        
    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        
        #src = [src len, batch size]
        #trg = [trg len, batch size]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        # config['vocab_size'], config['emb_dim'], config['hid_dim'], config['n_layers'], config['dropout']
        batch_size = trg.shape[0]
        trg_len = trg.shape[1] - 1
        trg_vocab_size = self.decoder.output_dim
        
        #tensor to store decoder outputs
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(src.device)
        
        #last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(src)
        hidden = hidden[:4] + hidden[4:]
        cell = cell[:4] + cell[4:]
        encoder_output = hidden
        
        #first input to the decoder is the <sos> tokens
        input = trg[:, 0].unsqueeze(1)
        if self.training:
            for t in range(trg_len):
                
                #insert input token embedding, previous hidden and previous cell states
                #receive output tensor (predictions) and new hidden and cell states
                output, hidden, cell = self.decoder(input, hidden, cell, encoder_output)
                
                #place predictions in a tensor holding predictions for each token
                outputs[:, t] = output.squeeze(1)
                
                #decide if we are going to use teacher forcing or not
                teacher_force = random.random() < teacher_forcing_ratio
                
                #get the highest predicted token from our predictions
                top1 = output.argmax(2) 
                #if teacher forcing, use actual next token as next input
                #if not, use predicted token
                input = trg[:, t + 1].unsqueeze(1) if teacher_force else top1
        else:
            for t in range(trg_len):
                
                #insert input token embedding, previous hidden and previous cell states
                #receive output tensor (predictions) and new hidden and cell states
                output, hidden, cell = self.decoder(input, hidden, cell, encoder_output)
                
                #place predictions in a tensor holding predictions for each token
                outputs[:, t] = output.squeeze(1)
                
                top1 = output.argmax(2) 
                #if teacher forcing, use actual next token as next input
                #if not, use predicted token
                input = top1
        
        return outputs.transpose(1, 2)

if __name__ == "__main__":
    INPUT_DIM = len(SRC.vocab)
    OUTPUT_DIM = len(TRG.vocab)
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    HID_DIM = 512
    N_LAYERS = 2
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5

    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

    model = Seq2Seq(enc, dec, device).to(device)

    tokenizer = Tokenizer.from_file("vocab/SLM-Java-tokenizer-bpe.json")
    tokenizer.decoder = decoders.BPEDecoder()
