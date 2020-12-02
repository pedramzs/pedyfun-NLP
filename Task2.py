"""
@author: Pedram ZS
"""


""" Libraries """
import pandas as pd
import numpy as np
import re
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import os.path
from torch.autograd import Variable
import os



""" Create a directory called results """
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)
        

createFolder('./results/') # Creates a folder in the current directory called results



""" Configs """
models_names = ['bert-base-uncased', 'albert-base-v1', 'distilbert-base-uncased', 'roberta-base']
          
print('\nthese pre-trained word-embedding models will be checked for SemEval-2020 task-7:\n')
for model_name in models_names:
    print(model_name)
print('\nfollowed by a 1-layer CNN\n')

seed_val = 1234
np.random.seed(seed_val)
torch.manual_seed(seed_val)

task2_train_df = pd.read_csv("semeval-2020-task-7-data-full/task-2/train.csv")
task2_valid_df = pd.read_csv("semeval-2020-task-7-data-full/task-2/dev.csv")
task2_test_df = pd.read_csv("semeval-2020-task-7-data-full/task-2/test.csv")

num_of_epochs = 4;


""" Functions """
def score_task_2(truth_loc, prediction_loc):
    truth = pd.read_csv(truth_loc, usecols=['id','label'])
    pred = pd.read_csv(prediction_loc, usecols=['id','pred'])
    
    assert(sorted(truth.id) == sorted(pred.id)),"ID mismatch between ground truth and prediction!"
    
    data = pd.merge(truth,pred)
    data = data[data.label != 0]
    accuracy = np.sum(data.label == data.pred)*100/len(data)
    
    print("Accuracy = %.3f %%" % accuracy)
    
    return accuracy

    
def get_sentence_pair(sent_orig, edit_word):
    original_sentence = re.sub("[</>]", "", sent_orig)
    edited_centence = (sent_orig.split("<"))[0] + edit_word + (sent_orig.split(">"))[1]

    return edited_centence, original_sentence



class two_sentence_dataset_1(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.df = dataframe
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.pad = self.tokenizer.pad_token_id
        self.cls = [self.tokenizer.cls_token_id]
        self.sep = [self.tokenizer.sep_token_id]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        label = torch.tensor(self.df["label"][idx])
        
        edit1, original_sentence = get_sentence_pair(self.df["original1"][idx], self.df["edit1"][idx])
        edit2, _ = get_sentence_pair(self.df["original2"][idx], self.df["edit2"][idx])

        edit1_tokens = (self.cls + self.tokenizer.encode(edit1, add_special_tokens=False) + self.sep)
        edit2_tokens = (self.cls + self.tokenizer.encode(edit2, add_special_tokens=False) + self.sep)
        
        original_sentence_tokens = (self.sep + self.tokenizer.encode(original_sentence, add_special_tokens=False) + self.sep)

        sentences_1 = self.tokenizer.encode(edit1, original_sentence, add_special_tokens=True)
        sentences_2 = self.tokenizer.encode(edit2, original_sentence, add_special_tokens=True)
        
        token_type_ids_1 = (torch.tensor([0] * len(edit1_tokens) + [1] * (self.max_len - len(edit1_tokens)))).long()
        token_type_ids_2 = (torch.tensor([0] * len(edit2_tokens) + [1] * (self.max_len - len(edit2_tokens)))).long()
        
        attention_mask_1 = torch.tensor([1] * (len(original_sentence_tokens) + len(edit1_tokens))+ 
                                      [0] * (self.max_len - len(original_sentence_tokens) - len(edit1_tokens)))
        attention_mask_2 = torch.tensor([1] * (len(original_sentence_tokens) + len(edit2_tokens))+ 
                                      [0] * (self.max_len - len(original_sentence_tokens) - len(edit2_tokens)))
        
        attention_mask_1 = attention_mask_1.float()
        attention_mask_2 = attention_mask_2.float()

        input_ids_1 = torch.tensor(sentences_1)
        input_ids_2 = torch.tensor(sentences_2)

        if len(input_ids_1) < self.max_len:
            input_ids_1 = torch.cat((input_ids_1,(torch.ones(self.max_len - len(input_ids_1)) * self.pad).long()))
            input_ids_2 = torch.cat((input_ids_2,(torch.ones(self.max_len - len(input_ids_2)) * self.pad).long()))
            
            token_type_ids_1 = torch.cat((token_type_ids_1,(torch.ones(self.max_len - len(token_type_ids_1)) * self.pad).long()))
            token_type_ids_2 = torch.cat((token_type_ids_2,(torch.ones(self.max_len - len(token_type_ids_2)) * self.pad).long()))
            
        else:
            input_ids_1 = input_ids_1[: self.max_len]
            input_ids_2 = input_ids_2[: self.max_len]
            
            token_type_ids_1 = token_type_ids_1[: self.max_len]
            token_type_ids_2 = token_type_ids_2[: self.max_len]

        return input_ids_1, input_ids_2, token_type_ids_1, token_type_ids_2, attention_mask_1, attention_mask_2, label
    


class two_sentence_dataset_2(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.df = dataframe
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.pad = self.tokenizer.pad_token_id
        self.cls = [self.tokenizer.cls_token_id]
        self.sep = [self.tokenizer.sep_token_id]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        label = torch.tensor(self.df["label"][idx])
        
        edit1, original_sentence = get_sentence_pair(self.df["original1"][idx], self.df["edit1"][idx])
        edit2, _ = get_sentence_pair(self.df["original2"][idx], self.df["edit2"][idx])

        edit1_tokens = (self.cls + self.tokenizer.encode(edit1, add_special_tokens=False) + self.sep)
        edit2_tokens = (self.cls + self.tokenizer.encode(edit2, add_special_tokens=False) + self.sep)
        
        original_sentence_tokens = (self.sep + self.tokenizer.encode(original_sentence, add_special_tokens=False) + self.sep)

        sentences_1 = self.tokenizer.encode(edit1, original_sentence, add_special_tokens=True)
        sentences_2 = self.tokenizer.encode(edit2, original_sentence, add_special_tokens=True)
        
        attention_mask_1 = torch.tensor([1] * (len(original_sentence_tokens) + len(edit1_tokens))+ 
                                      [0] * (self.max_len - len(original_sentence_tokens) - len(edit1_tokens)))
        attention_mask_2 = torch.tensor([1] * (len(original_sentence_tokens) + len(edit2_tokens))+ 
                                      [0] * (self.max_len - len(original_sentence_tokens) - len(edit2_tokens)))
        
        attention_mask_1 = attention_mask_1.float()
        attention_mask_2 = attention_mask_2.float()

        input_ids_1 = torch.tensor(sentences_1)
        input_ids_2 = torch.tensor(sentences_2)

        if len(input_ids_1) < self.max_len:
            input_ids_1 = torch.cat((input_ids_1,(torch.ones(self.max_len - len(input_ids_1)) * self.pad).long()))
            input_ids_2 = torch.cat((input_ids_2,(torch.ones(self.max_len - len(input_ids_2)) * self.pad).long()))
            
        else:
            input_ids_1 = input_ids_1[: self.max_len]
            input_ids_2 = input_ids_2[: self.max_len]

        return input_ids_1, input_ids_2, attention_mask_1, attention_mask_2, label



class ConvNet(nn.Module):
    
    def __init__(self): # input is (32,100,768)
        super(ConvNet, self).__init__()
        
        self.CNN1 = nn.ModuleList(
            [nn.Conv2d(1, 3, (K, 768)) for K in [2, 3, 4, 8]])
        
        self.CNN2 = nn.ModuleList(
            [nn.Conv2d(1, 3, (K, 768)) for K in [2, 3, 4, 8]])
        
        self.drop_out = nn.Dropout(0.5)
        
        self.linear = nn.Linear(4*3*2, 3)
        
    def forward(self, x1, x2):
        x1 = Variable(x1)
        x2 = Variable(x2)
        
        x1 = x1.unsqueeze(1) #(32,1,100,768)
        x2 = x2.unsqueeze(1) #(32,1,100,768)
        
        x1 = [nn.functional.relu(cnn(x1)).squeeze(3) for cnn in self.CNN1] #(32,3,)
        x1 = [nn.functional.max_pool1d(i, i.size(2)).squeeze(2) for i in x1]  # [(N, Co), ...]*len(Ks)
        
        x2 = [nn.functional.relu(cnn(x2)).squeeze(3) for cnn in self.CNN2] #(32,3,)
        x2 = [nn.functional.max_pool1d(i, i.size(2)).squeeze(2) for i in x2]  # [(N, Co), ...]*len(Ks)
        
        x1 = torch.cat(x1, 1)
        x2 = torch.cat(x2, 1)
        
        x = torch.cat((x1, x2), 1)
        
        x = self.drop_out(x) # (32, 24)
        x = self.linear(x)
        
        return x
    
    
""" Main """
criterion = nn.CrossEntropyLoss()
model = ConvNet()

Acc = dict()
for model_name in models_names:
    
    print('model:', model_name)
    
    fname2 = model_name + "_task2"
    
    SAVE_PATH_task2 = os.path.join("results/" + fname2 + "_test.csv");
    
    Model_PATH = os.path.join("./" + fname2 + "_model.pt");
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)   
    
    embedding_model = AutoModel.from_pretrained(model_name)
    
    # Task 2:
    print('Task 2')

    train_df = task2_train_df[["id", "original1", "edit1", "original2", "edit2", "label"]]  
    valid_df = task2_valid_df[["id", "original1", "edit1", "original2", "edit2", "label"]]
    test_df = task2_test_df[["id", "original1", "edit1", "original2", "edit2", "label"]]
    
    
    if model_name == 'bert-base-uncased' or model_name == 'albert-base-v1':
    
        train_dset = two_sentence_dataset_1(train_df, tokenizer, max_len=100)
        valid_dset = two_sentence_dataset_1(valid_df, tokenizer, max_len=100)
        test_dset = two_sentence_dataset_1(test_df, tokenizer, max_len=100)
        
    else:
        
        train_dset = two_sentence_dataset_2(train_df, tokenizer, max_len=100)
        valid_dset = two_sentence_dataset_2(valid_df, tokenizer, max_len=100)
        test_dset = two_sentence_dataset_2(test_df, tokenizer, max_len=100)
            
        
    train_loader = DataLoader(train_dset, batch_size=32, shuffle=False, num_workers=0)
    valid_loader = DataLoader(valid_dset, batch_size=32, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dset, batch_size=32, shuffle=False, num_workers=0)
    
    
    # Train the model 
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, eps=1e-8)  
    min_loss = 10000
    
    if __name__ == '__main__':
        
        for epoch in range(num_of_epochs):
            
            epoch_LOSS = 0;

            model.train()
                
            for ii, data in enumerate(train_loader):
                
                if len(data) == 7:                   
                    input_ids_1, input_ids_2, token_type_ids_1, token_type_ids_2, attention_mask_1, attention_mask_2, label = data
                    
                    with torch.no_grad():
                        output1 = embedding_model(input_ids=input_ids_1, attention_mask=attention_mask_1, token_type_ids=token_type_ids_1)
                        output2 = embedding_model(input_ids=input_ids_2, attention_mask=attention_mask_2, token_type_ids=token_type_ids_2)
                    
                else:
                    input_ids_1, input_ids_2, attention_mask_1, attention_mask_2, label = data
                    
                    with torch.no_grad():
                        output1 = embedding_model(input_ids=input_ids_1, attention_mask=attention_mask_1)
                        output2 = embedding_model(input_ids=input_ids_2, attention_mask=attention_mask_2)
                    
                x1 = output1[0];
                x2 = output2[0];
                
                outputs = model(x1, x2)
                
                optimizer.zero_grad()

                loss = criterion(outputs, label)
        
                loss.backward()
                optimizer.step()
        
                epoch_LOSS += loss.item();
            
                print("batch percentage: %.2f %%,    loss: %f" % (100*min(1, 32*(ii+1)/len(train_dset)), epoch_LOSS/(32*(ii+1))))
                
                if ii % 100 == 99:  # print every 100 mini-batches
                    print("epoch: %d,    used train set: %.2f %%,    loss: %f" 
                          % (epoch+1, 100*min(1, 32*(ii+1)/len(train_dset)), epoch_LOSS/(32*(ii+1))))
                    
            print("\nepoch ", epoch+1, ",    train loss = ", epoch_LOSS/len(train_dset))
            
            
            # Validation Check
            print('checking the new', model_name, 'model validation ...')
            
            validation_loss = 0.0
            model.eval()
            with torch.no_grad():
                for data in valid_loader:
                    
                    if len(data) == 7:                   
                        input_ids_1, input_ids_2, token_type_ids_1, token_type_ids_2, attention_mask_1, attention_mask_2, label = data
                    
                        with torch.no_grad():
                            output1 = embedding_model(input_ids=input_ids_1, attention_mask=attention_mask_1, token_type_ids=token_type_ids_1)
                            output2 = embedding_model(input_ids=input_ids_2, attention_mask=attention_mask_2, token_type_ids=token_type_ids_2)
                    
                    else:
                        input_ids_1, input_ids_2, attention_mask_1, attention_mask_2, label = data
                    
                        with torch.no_grad():
                            output1 = embedding_model(input_ids=input_ids_1, attention_mask=attention_mask_1)
                            output2 = embedding_model(input_ids=input_ids_2, attention_mask=attention_mask_2)
                    
                    x1 = output1[0];
                    x2 = output2[0];
                    
                    outputs = model(x1, x2)

                    loss = criterion(outputs, label)
        
                    validation_loss += loss.item()
        
            VALIDATION_LOSS = validation_loss/len(valid_dset);
            print("epoch ", epoch+1, ",    Validation loss = ", VALIDATION_LOSS, "\n")
        
            if VALIDATION_LOSS < min_loss:
                print("best model for", model_name, "so far was found\n")
                min_loss = VALIDATION_LOSS;
                torch.save(model.state_dict(), Model_PATH)


    # Test the best model
    print('testing the best', model_name, 'model for task 2 ...')
    
    preds = []
    model.load_state_dict(torch.load(Model_PATH))
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            
            if len(data) == 7:                   
                input_ids_1, input_ids_2, token_type_ids_1, token_type_ids_2, attention_mask_1, attention_mask_2, label = data
            
                with torch.no_grad():
                    output1 = embedding_model(input_ids=input_ids_1, attention_mask=attention_mask_1, token_type_ids=token_type_ids_1)
                    output2 = embedding_model(input_ids=input_ids_2, attention_mask=attention_mask_2, token_type_ids=token_type_ids_2)
            
            else:
                input_ids_1, input_ids_2, attention_mask_1, attention_mask_2, label = data
            
                with torch.no_grad():
                    output1 = embedding_model(input_ids=input_ids_1, attention_mask=attention_mask_1)
                    output2 = embedding_model(input_ids=input_ids_2, attention_mask=attention_mask_2)
            
            x1 = output1[0];
            x2 = output2[0];
            
            outputs = model(x1, x2)
            
            _, prediction = torch.max(outputs.data, 1)
    
            preds = np.append(preds, prediction.data.numpy())

    
    prediction_df = pd.DataFrame(columns=["id", "pred"])
    prediction_df["id"] = test_df["id"]
    prediction_df["pred"] = preds
    assert len(prediction_df) == len(test_df)
    
    prediction_df.to_csv(SAVE_PATH_task2, index=False)
    
    Acc[model_name] = score_task_2("semeval-2020-task-7-data-full/task-2/test.csv", SAVE_PATH_task2)
   
    
print('\nAccuracy for Task 2 with the best model of each BERT-based word embedding model:\n')
for key in Acc:
    print(key, ' : ', Acc[key])