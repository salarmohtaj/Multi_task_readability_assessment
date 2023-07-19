import torch
#from torchtext.legacy.data import Field, LabelField, TabularDataset, BucketIterator
from torchtext.legacy.data import Field, LabelField, TabularDataset, BucketIterator
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import os
import re
import pickle
import time
import argparse
import spacy
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertModel, BertForSequenceClassification, AdamW, BertConfig, BertTokenizer, get_linear_schedule_with_warmup


try:
    spacy_en = spacy.load('de_core_news_sm')
except:
    spacy_en = spacy.load('en')










device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# data_dir = "../Data/hate_speech/final_data/1/"
# data_dir = "../Data/fake_news/final_data/1/"


bert_model_name = "bert-base-german-cased"
MAX_LEN = 256
BATCH_SIZE = 32
N_EPOCHS = 50
learning_rate = 0.001
hidden_size = 128
dropout = 0.4


destination_folder = "Model/BERT"
report_address = "Reports/BERT"
########################################################################

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

try:
    os.mkdir("Model")
except:
    pass
try:
    os.mkdir("Reports")
except:
    pass
try:
    os.mkdir(destination_folder)
except:
    pass
with open(report_address,"a+") as f:
    f.write("\n\n")






def text_preprocess(text):
    text = re.sub("@([A-Za-z0-9_]+)", "username", text)
    text = re.sub(r"http\S+", "weblink", text)
    #text = demoji.replace_with_desc(text, sep=" ")
    text = re.sub("[ ]+", " ", text)
    return text

tokenizer = BertTokenizer.from_pretrained(bert_model_name)
def bert_tokenize(text):
    text = text_preprocess(text)
    return tokenizer.tokenize(text)[:MAX_LEN - 2]

def my_tokenizer(text):
    text = text_preprocess(text)
    return [tok.text for tok in spacy_en.tokenizer(text)][:MAX_LEN - 2]


def tokenize_and_cut(text):
    text = text_preprocess(text)
    tokens = tokenizer.tokenize(text)
    tokens = tokens[:MAX_LEN-2]
    return tokens


text_field = Field(batch_first = True,
                  use_vocab = False,
                  tokenize = tokenize_and_cut,
                  preprocessing = tokenizer.convert_tokens_to_ids,
                  init_token = tokenizer.cls_token_id,
                  eos_token = tokenizer.sep_token_id,
                  pad_token = tokenizer.pad_token_id,
                  unk_token = tokenizer.unk_token_id)


# text_field = Field(tokenize = my_tokenizer, sequential = True, lower = True, batch_first = True)
# text_field = Field(sequential=True, tokenize=bert_tokenize, pad_token=tokenizer.pad_token,init_token=tokenizer.cls_token, eos_token=tokenizer.sep_token,batch_first=True)
label_field = LabelField(is_target=True, dtype=torch.float)

# fields = [('text', text_field), ('label', label_field)]
LABEL = LabelField(is_target=True, sequential=False, use_vocab=False, dtype=torch.float)

fields = [(None, None), ('text', text_field), (None, None), (None, None), ('label', LABEL)]

config = BertConfig.from_pretrained(bert_model_name, output_hidden_states=True)
bert = BertModel.from_pretrained(bert_model_name, config=config)


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss

class BiGRU_Model(nn.Module):
    def __init__(self,
                 bert,
                 hidden_dim,
                 dropout
                 ):
        super().__init__()
        self.bert = bert
        # self.embedding = nn.Embedding(input_size, embedding_dim)
        self.embedding_dim = bert.config.to_dict()['hidden_size']
        self.rnn = nn.GRU(self.embedding_dim,
                          hidden_dim,
                          num_layers=2,
                          bidirectional=True,
                          batch_first=True,
                          dropout=dropout)
        self.linear = nn.Linear(hidden_dim * 2, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        # text = [batch size, sent len]

        with torch.no_grad():
            embedded = self.bert(text)[0]
            # embedded = self.bert(text)
            # cat_vec = torch.stack(embedded[2])
            # embedded = torch.mean(cat_vec[-2:], dim=0)
        # embedded = self.embedding(text)
        # embedded = [batch size, sent len, emb dim]
        _, hidden = self.rnn(embedded)
        # hidden = [n layers * n directions, batch size, emb dim]
        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        # hidden = [batch size, hid dim]
        output = self.linear(hidden)
        #output = self.dropout(output)
        output = self.out(output)
        # output = [batch size, out dim]
        # return output
        return output


# Save and Load Functions
def save_checkpoint(save_path, model, optimizer, valid_loss):
    if save_path == None:
        return
    state_dict = {'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'valid_loss': valid_loss}
    torch.save(state_dict, save_path)
    #print(f'Model saved to ==> {save_path}')

def load_checkpoint(load_path, model, optimizer):
    if load_path == None:
        return
    state_dict = torch.load(load_path, map_location=device)
    #print(f'Model loaded from <== {load_path}')
    model.load_state_dict(state_dict['model_state_dict'])
    optimizer.load_state_dict(state_dict['optimizer_state_dict'])
    return state_dict['valid_loss']

def save_metrics(save_path, train_loss_list, valid_loss_list, epoch_counter_list):
    if save_path == None:
        return
    state_dict = {'train_loss_list': train_loss_list,
                  'valid_loss_list': valid_loss_list,
                  'epoch_counter_list': epoch_counter_list}
    torch.save(state_dict, save_path)
    #print(f'Model saved to ==> {save_path}')

def load_metrics(load_path):
    if load_path == None:
        return
    state_dict = torch.load(load_path, map_location=device)
    #print(f'Model loaded from <== {load_path}')
    return state_dict['train_loss_list'], state_dict['valid_loss_list'], state_dict['epoch_counter_list']


def binary_accuracy(preds, y):
    #rounded_preds = torch.round(torch.sigmoid(preds))
    #rounded_preds = torch.round(preds)
    rounded_preds = (preds > 0.5).int()
    correct = (rounded_preds == y).float() #convert into float for division
    acc = correct.sum() / len(correct)
    return acc

def train_def(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    for batch in iterator:
        optimizer.zero_grad()
        labels = batch.label
        #text = batch.text.t()
        text = batch.text
        # output = model(text).squeeze(1)
        prediction = model(text)
        loss = criterion(torch.squeeze(prediction), labels)
        # loss = criterion(output, labels)
        # acc = binary_accuracy(output, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        # epoch_acc += acc.item()
    # return epoch_loss / len(iterator), epoch_acc / len(iterator)
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for batch in iterator:
            labels = batch.label
            # text = batch.text.t()
            text = batch.text
            # output = model(text).squeeze(1)
            prediction = model(text)
            y = labels.cpu().numpy()
            y_hat = np.concatenate(prediction.cpu().numpy(), axis=0)
            loss = criterion(torch.squeeze(prediction), labels)
            # loss = criterion(output, labels)
            rmse = np.sqrt(((y_hat - y) ** 2).mean())
            # acc = binary_accuracy(output, labels)
            epoch_loss += loss.item()
            # epoch_acc += acc.item()
    # return epoch_loss / len(iterator), epoch_acc / len(iterator)
    return epoch_loss / len(iterator), rmse


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs




def test_evaluation(bestmodel, test_loader):
    y_pred = []
    y_true = []
    bestmodel.eval()
    with torch.no_grad():
        for batch in test_loader:
            labels = batch.label.to(device)
            text = batch.text.to(device)
            prediction = model(text)
            y = labels.cpu().numpy()
            y_hat = np.concatenate(prediction.cpu().numpy(), axis=0)
            rmse = np.sqrt(((y_hat - y) ** 2).mean())
    with open(report_address, "a") as f:
        f.write("Classification Report:\n")
        # f.write(str(classificationreport) + "\n")
    return(rmse)


y_pred = []

y_true = []




train = TabularDataset(path="../Data/train.csv",format='CSV', fields=fields, skip_header=True)
valid = TabularDataset(path="../Data/valid.csv",format='CSV', fields=fields, skip_header=True)
test = TabularDataset(path="../Data/test.csv",format='CSV', fields=fields, skip_header=True)
#train,_ = train.split(0.3)
print(f'the train, validation and test sets includes {len(train)},{len(valid)} and {len(test)} instances, respectively')
print(type(train))
# text_field.build_vocab(train, min_freq=2, vectors="glove.6B.300d")
# text_field.build_vocab(train, min_freq=2)
label_field.build_vocab(train)

# text_field.vocab.stoi = tokenizer.vocab
# text_field.vocab.itos = list(tokenizer.vocab)

# print(label_field.vocab.stoi)


train_iter = BucketIterator(train, batch_size=BATCH_SIZE, sort_key=lambda x: len(x.text),device=device, sort=True, sort_within_batch=True)
valid_iter = BucketIterator(valid, batch_size=BATCH_SIZE, sort_key=lambda x: len(x.text),device=device, sort=True, sort_within_batch=True)
test_iter = BucketIterator(test, batch_size=BATCH_SIZE, sort_key=lambda x: len(x.text),device=device, sort=True, sort_within_batch=True)
bert = BertModel.from_pretrained(bert_model_name, config=config)
model = BiGRU_Model(bert, hidden_size, dropout)

batch = next(iter(train_iter))
phrase = batch.text
sent = batch.label
print(phrase.shape)
print(phrase)
print(sent.shape)
print(sent)

print("###################################")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f'The model has {count_parameters(model):,} parameters')

for name, param in model.named_parameters():
    if name.startswith('bert'):
        param.requires_grad = False

# model.embedding.weight.requires_grad = False

print(f'The model has {count_parameters(model):,} trainable parameters')
print("###################################")

# optimizer = optim.Adam(model.parameters(), lr=learning_rate)
optimizer = optim.Adam([param for param in model.parameters() if param.requires_grad == True], lr=learning_rate)
# criterion = nn.BCELoss()
# criterion = nn.MSELoss()
criterion = RMSELoss()
model = model.to(device)
criterion = criterion.to(device)

best_valid_loss = float('inf')
training_stats = []
train_loss_list = []
valid_loss_list = []
epoch_counter_list = []
last_best_loss = 0

for epoch in range(N_EPOCHS):
    start_time = time.time()
    train_loss = train_def(model, train_iter, optimizer, criterion)
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    valid_loss,rmse  = evaluate(model, valid_iter, criterion)
    train_loss_list.append(train_loss)
    valid_loss_list.append(valid_loss)
    epoch_counter_list.append(epoch)
    print(f'Epoch: {epoch + 1:02}/{N_EPOCHS} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'RMSE: {rmse}')
    # print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
    print(f'\tTrain Loss: {train_loss:.3f} | Val. Loss: {valid_loss:.3f}')
    # print(f'\tVal. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')
    with open(report_address, "a") as f:
        f.write("Epoch " + str(epoch + 1) + "\n")
        f.write("Train Loss: " + str(train_loss) + "\tVal. Loss: " + str(valid_loss) + "\n")
        # f.write("Val. Loss: " + str(valid_loss) + "\tVal. Acc" + str(valid_acc) + "\n")
    if valid_loss < best_valid_loss:
        last_best_loss = epoch
        print("\t---> Saving the model <---")
        best_valid_loss = valid_loss
        ##torch.save(model.state_dict(), 'tut6-model.pt')
        save_checkpoint(destination_folder + '/model.pt', model, optimizer, best_valid_loss)
        save_metrics(destination_folder + '/metrics.pt', train_loss_list, valid_loss_list, epoch_counter_list)
    training_stats.append(
        {
            'epoch': epoch + 1,
            'Training Loss': train_loss,
            'Valid. Loss': valid_loss,
            # 'Training Accuracy': train_acc,
            # 'Valid. Accur.': avg_val_accuracy,
            'Training Time': epoch_mins,
        }
    )
    if ((epoch - last_best_loss) > 9):
        print("################")
        print("Termination because of lack of improvement in the last 10 epochs")
        print("################")
        with open(report_address, "a") as f:
            f.write("Termination because of lack of improvement in the last 10 epochs\n")
        break
save_metrics(destination_folder + '/metrics.pt', train_loss_list, valid_loss_list, epoch_counter_list)
print('Finished Training!')


best_model = BiGRU_Model(bert, hidden_size, dropout).to(device)
load_checkpoint(destination_folder + '/model.pt', best_model, optimizer)
optimizer = optim.Adam([param for param in model.parameters() if param.requires_grad == True],lr=learning_rate)
rmse = test_evaluation(best_model, test_iter)
print(rmse)