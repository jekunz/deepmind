import time, sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import CorpusLoader
from helpers import device, load_model, save_model

class LMEntity(nn.Module):
    def __init__(self, hidden_size, vocab_size, embedding_size, num_layers=1, init_range=(-0.1, 0.1), dropout=0.1, train_states=False):
        super().__init__()

        self.num_layers = num_layers
        self.batch_size = 1
        self.drop_layer = nn.Dropout(p=dropout)
        #****************#
        # # # Layers # # # 
        # init word embedding layer
        self.embedding_matrix = nn.Embedding(vocab_size, embedding_size)

        # init lstm 
        self.lstm = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, batch_first=True)
        self.train_states = train_states
        if train_states:
            self.h0 = nn.Parameter(torch.zeros(num_layers, self.batch_size, hidden_size).to(device), requires_grad=True)
            self.c0 = nn.Parameter(torch.zeros(num_layers, self.batch_size, hidden_size).to(device), requires_grad=True)

        # init W1 and W2
        self.W1 = nn.Linear(hidden_size, vocab_size)
        self.W2 = nn.Linear(hidden_size*2, hidden_size)


        # scoring matrix for attention part after stanford method

        # scoring layer for attention part
        self.W_score = nn.Linear(hidden_size, hidden_size)
        #self.v_a = Var(torch.FloatTensor(num_layers, lstm_batch_size, hidden_size), requires_grad=True)

        # weights on matrix for calculating z_i
        self.z_weights = nn.Linear(hidden_size*2, 1)
        # # # Layers END # # #

        #********************#
        # # # h_e states # # #
        # init h_e_0 state that indicates a new entity and is learnable parameter
        self.h_e_m = torch.Tensor().to(device)
        self.h_e_0 = nn.Parameter(torch.FloatTensor(num_layers, self.batch_size, hidden_size).to(device), requires_grad=True)
        # init set h_e which stores the entity states
        # # # h_e states END # # #

        # init weights
        self.weight_init(init_range)
        
        # initialize all necessary states
        self.reset_state()

    def reset_state(self):
        # reset hidden state
        # and clear all entities
        self.init_hidden()
        self.init_h_e()

    def init_hidden(self):
        # initialize hidden state either with zeros or pre-trained
        if self.train_states:
            self.hidden = (self.h0, self.c0)
        else:
            h0 = torch.zeros(self.num_layers, self.batch_size, self.lstm.hidden_size).to(device) 
            c0 = torch.zeros(self.num_layers, self.batch_size, self.lstm.hidden_size).to(device)
            self.hidden = (h0, c0)
        
    def init_h_e(self):
        # clear all entities from set of entities
        self.h_e_m = torch.Tensor().to(device)
        # add new entity parameter
        self.h_e_m = torch.cat((self.h_e_m, self.h_e_0), dim=0)


    def weight_init(self, init_range):
        # initialize layer weights
        self.embedding_matrix.weight.data.uniform_(*init_range)
        self.h_e_0.data.uniform_(*init_range)
        
        if self.train_states:
            self.c0.data.uniform_(*init_range)
            self.h0.data.uniform_(*init_range)
        
        self.W_score.bias.data.fill_(0)
        self.W_score.weight.data.uniform_(*init_range)

        self.z_weights.bias.data.fill_(0)
        self.z_weights.weight.data.uniform_(*init_range)

        self.W1.bias.data.fill_(0)
        self.W1.weight.data.uniform_(*init_range)
        self.W2.bias.data.fill_(0)
        self.W2.weight.data.uniform_(*init_range)


    def get_attn_scores(self, h):
        # generate (stanford) attention scores
        # query:  h resp hidden state
        # values: entity representations in set of entities
        p_v_ = torch.matmul(h , torch.t(self.drop_layer(self.W_score((self.h_e_m.view(-1, self.lstm.hidden_size))))))
        p_v = nn.functional.softmax(p_v_.view(-1), dim=0) # softmax for weighted sum
        return p_v, p_v_ #softmax normalized, raw values / logits
      
    def train_forward(self, x, entity_target):
        assert entity_target.item() <= self.h_e_m.size(0), f'entity index out of range: \nh_e_m.size(0): {self.h_e_m.size(0)}, entity_target: {entity_target.item()}'
            
        # apply embeding
        x = self.embedding_matrix(x.view(-1, 1))
        
        # forward RNN
        out, states = self.lstm(x, self.hidden)
        self.hidden = states
        hidden = self.drop_layer(states[0])
        
        # create attention scores over entity states
        p_v, p_v_ = self.get_attn_scores(hidden)

        # create weighted sum of entity states
        d_i = torch.sum(torch.matmul(p_v.diag(),self.h_e_m.view(-1, self.lstm.hidden_size)), dim=0).unsqueeze(0)

        # calculate z_i
        linear_z_i = self.drop_layer( self.z_weights(torch.cat((hidden[0], d_i), dim=1)) )
        z_i = torch.sigmoid(linear_z_i)

        # CASE 1: Next token is entity
        if entity_target:
            if entity_target.item() < self.h_e_m.size(0):
                # Update existing entity in set of entities with current hidden state
                self.h_e_m = self.h_e_m.index_copy(0, entity_target, hidden)
            else:
                # Add new entity to set of entities
                self.h_e_m = torch.cat((self.h_e_m , hidden), dim=0)
            
            # W1(tanh(W2([h_i−1 , h_e,vi ]))
            out = self.W1(torch.tanh(self.drop_layer( self.W2(torch.cat((hidden.squeeze(), self.h_e_m[entity_target].squeeze()), dim=0))) )).unsqueeze(0)
            

        # CASE 2: Next token is no entity
        else:
            out = self.W1(hidden[0])
        out = out.contiguous()
        return out, z_i, p_v

    def forward(self, x):
        # create embedded token
        x = self.embedding_matrix(x.view(-1, 1))
        # forward rnn
        out, states = self.lstm(x, self.hidden)
        self.hidden = states
        hidden = states[0]

        # create attention scores for for entities with hidden state
        p_v, p_v_ = self.get_attn_scores(hidden)
        
        # weighted sum of entity states
        d_i = torch.sum(torch.matmul(p_v.diag(),self.h_e_m.view(-1, self.lstm.hidden_size*self.num_layers)), dim=0).unsqueeze(0)
        
        # calculate z_i
        linear_z_i = self.z_weights(torch.cat((hidden[0], d_i), dim=1))
        z_i = torch.sigmoid(linear_z_i)

        if z_i > 0.5:
            # get most probable entity
            v_i = p_v.argmax()
            if v_i != 0:
                # Update exsisting entity in set of entities with current hidden state
                self.h_e_m = self.h_e_m.index_copy(0, v_i, hidden)

            else:
                # Add new entity to set of entities
                self.h_e_m = torch.cat((self.h_e_m , hidden), dim=0)
                
            # W1(tanh(W2([h_i−1 , h_e,vi ]))
            out = self.W1(torch.tanh(self.drop_layer( self.W2(torch.cat((hidden.squeeze(), self.h_e_m[entity_target].squeeze()), dim=0))) )).unsqueeze(0)
            
        else:
            out = self.W1(hidden[0])
        out = out.contiguous()
        return out, z_i, p_v
    
    
    
def run_lme(model, corpus, optimizer=None, epochs=1, eval_corpus=None, status_interval=25, str_pattern='{}_{}_epoch_{}.pkl', rz_amplifier=1):
    for epoch in range(1, epochs+1):
        # initialize variables for epoch stats
        X_epoch_loss, E_epoch_loss, Z_epoch_loss = 0, 0, 0
        epoch_tokens, epoch_e_div = 0, 0
        epoch_start = time.time()
        count_E = 0
        count_E_correct = 0
        count_Z = 0
        z_true_positive  = 0
        z_false_positive = 0
        
        # zero target for new entities
        zero_target = torch.zeros(1, dtype=torch.long, device=device)
        for i_doc, doc in enumerate(corpus.gen()):
            # reset all model states
            model.reset_state()
            doc_num_tokens    = doc.X.size(0)-1
            # initialize document loss
            X_loss = torch.tensor(0, dtype=torch.float, device=device)
            E_loss = torch.tensor(0, dtype=torch.float, device=device)
            Z_loss = torch.tensor(0, dtype=torch.float, device=device)
            
            # div counter for entity loss
            e_div = torch.sum(doc.Z).item()
            
            for t in range(doc_num_tokens):
                # targets
                x_target = doc.X[t+1].unsqueeze(0)
                e_target = doc.E[t+1].unsqueeze(0)
                z_target = doc.Z[t+1]
                # forward model
                x_out, z_i, p_v = model.train_forward(doc.X[t], e_target)
                # adding prediction stats for z
                if z_i > 0.5:
                    if z_target:
                        z_true_positive  += 1
                    else:
                        z_false_positive += 1
            
                # token loss
                X_loss += F.cross_entropy(x_out, x_target)
                # z_i loss
                Z_loss += F.binary_cross_entropy(z_i.squeeze(), z_target)*rz_amplifier
                
                # entity loss
                if e_target:
                    # adding new entity not occurred before
                    if p_v.size(0) == e_target:
                        E_loss += F.cross_entropy(p_v.unsqueeze(0), zero_target)
                        count_E_correct += int(p_v.argmax() == 0)
                    # already known entity
                    else:
                        E_loss += F.cross_entropy(p_v.unsqueeze(0), e_target)
                        count_E_correct += int(p_v.argmax() == e_target)
                    count_E += 1
            
            # add epoch loss and deviding loss values
            X_epoch_loss += X_loss.item()
            E_epoch_loss += E_loss.item()
            Z_epoch_loss += Z_loss.item()
            X_loss /= doc_num_tokens
            E_loss /= max(e_div, 1)
            Z_loss /= doc_num_tokens       
            
            
            epoch_tokens += doc_num_tokens
            epoch_e_div += e_div
            count_Z += torch.sum(doc.Z).item()

            if optimizer:
                # optimization step
                optimizer.zero_grad()
                loss = X_loss + E_loss + Z_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
            if status_interval and i_doc % status_interval == 0:
                # status output
                z_prec   = z_true_positive / max((z_true_positive+z_false_positive), 1)
                z_recall = z_true_positive / max(count_Z, 1)
                print(f'Doc {i_doc}/{len(corpus)-1}: X_loss {X_epoch_loss / epoch_tokens:0.3}, Z_loss {Z_epoch_loss / epoch_tokens:0.3}, E_loss {E_epoch_loss / epoch_e_div:0.3}, E_acc {count_E_correct/count_E:0.3}, Z_prec {z_prec:0.3}, Z_recall {z_recall:0.3}')
        
        # calulate readable time format
        seconds = round(time.time() - epoch_start)
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        x_hour_and_ = f'{h} hours and '*bool(h)
        if optimizer:
            print(f'Epoch {epoch} finished after {x_hour_and_}{m} minutes.')
        else:
            print(f'Evaluation on "{corpus.partition}" partition finished after {x_hour_and_}{m} minutes.')
            
        # calculate epoch stats: precision, recall and F-Score
        z_prec   = z_true_positive / max((z_true_positive+z_false_positive), 1)
        z_recall = z_true_positive / max(count_Z, 1)
        zf_score  = 2*((z_prec*z_recall)/(z_prec+z_recall))

        print(f'Loss: X_loss {X_epoch_loss / epoch_tokens:0.3}, Z_loss {Z_epoch_loss / epoch_tokens:0.3}, E_loss {E_epoch_loss / epoch_e_div:0.3}, E_acc {count_E_correct/count_E:0.3}, Z_prec {z_prec:0.3}, Z_recall {z_recall:0.3}, Z_Fscore {zf_score:0.3}')
        print()
        
        # if in train mode
        if optimizer:
            # saving model
            file_name = str_pattern.format(model.__class__.__name__, model.lstm.hidden_size, epoch)
            save_model(model, file_name)
            # evaluate on evaluation corpus
            if eval_corpus:
                with torch.no_grad():
                    model.eval()
                    run_model(model, eval_corpus, status_interval=None)
                    model.train()


corpus = CorpusLoader(partition='train', lengths=False)
eval_corpus = CorpusLoader(partition='dev', lengths=False)
d = 64
model = LMEntity(vocab_size=corpus.vocab_size, 
                        embedding_size=d, 
                        hidden_size=d,
                        dropout=0.2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
run_lme(model, corpus, optimizer, epochs=25, eval_corpus=eval_corpus, status_interval=250)


