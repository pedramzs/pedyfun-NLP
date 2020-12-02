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
          
print('these pre-trained word-embedding models will be checked for SemEval-2020 task-7:\n')
for model_name in models_names:
    print(model_name, '\n')
print('followed by a 1-layer CNN')

seed_val = 1234
np.random.seed(seed_val)
torch.manual_seed(seed_val)

task1_train_df = pd.read_csv("semeval-2020-task-7-data-full/task-1/train.csv")
task1_valid_df = pd.read_csv("semeval-2020-task-7-data-full/task-1/dev.csv")
task1_test_df = pd.read_csv("semeval-2020-task-7-data-full/task-1/test.csv")

task2_test_df = pd.read_csv("semeval-2020-task-7-data-full/task-2/test.csv")

num_of_epochs = 4;


""" Functions """
def score_task_1(truth_loc, prediction_loc):
    truth = pd.read_csv(truth_loc, usecols=['id','meanGrade'])
    pred = pd.read_csv(prediction_loc, usecols=['id','pred'])
    
    assert(sorted(truth.id) == sorted(pred.id)),"ID mismatch between ground truth and prediction!"
    
    data = pd.merge(truth,pred)
    rmse = np.sqrt(np.mean((data['meanGrade'] - data['pred'])**2))
    
    print("RMSE = %.3f" % rmse)
    
    return rmse
    

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

        grade = torch.tensor(self.df["meanGrade"][idx])

        edited_centence, original_sentence = get_sentence_pair(self.df["original"][idx], self.df["edit"][idx])

        edited_centence_tokens = (self.cls + self.tokenizer.encode(edited_centence, add_special_tokens=False) + self.sep)
        
        original_sentence_tokens = (self.sep + self.tokenizer.encode(original_sentence, add_special_tokens=False) + self.sep)

        sentences = self.tokenizer.encode(edited_centence, original_sentence, add_special_tokens=True)
        
        token_type_ids = (torch.tensor([0] * len(edited_centence_tokens) + [1] * (self.max_len - len(edited_centence_tokens)))).long()
        
        attention_mask = torch.tensor([1] * (len(original_sentence_tokens) + len(edited_centence_tokens))+ 
                                      [0] * (self.max_len - len(original_sentence_tokens) - len(edited_centence_tokens)))
        
        attention_mask = attention_mask.float()

        input_ids = torch.tensor(sentences)

        if len(input_ids) < self.max_len:
            input_ids = torch.cat((input_ids,(torch.ones(self.max_len - len(input_ids)) * self.pad).long()))
            
            token_type_ids = torch.cat((token_type_ids,(torch.ones(self.max_len - len(token_type_ids)) * self.pad).long()))
            
        else:
            input_ids = input_ids[: self.max_len]
            token_type_ids = token_type_ids[: self.max_len]

        return input_ids, token_type_ids, attention_mask, grade
    

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

        grade = torch.tensor(self.df["meanGrade"][idx])

        edited_centence, original_sentence = get_sentence_pair(self.df["original"][idx], self.df["edit"][idx])

        edited_centence_tokens = (self.cls + self.tokenizer.encode(edited_centence, add_special_tokens=False) + self.sep)
        
        original_sentence_tokens = (self.sep + self.tokenizer.encode(original_sentence, add_special_tokens=False) + self.sep)

        sentences = self.tokenizer.encode(edited_centence, original_sentence, add_special_tokens=True)
        
        attention_mask = torch.tensor([1] * (len(original_sentence_tokens) + len(edited_centence_tokens))+ 
                                      [0] * (self.max_len - len(original_sentence_tokens) - len(edited_centence_tokens)))
        
        attention_mask = attention_mask.float()

        input_ids = torch.tensor(sentences)

        if len(input_ids) < self.max_len:
            input_ids = torch.cat((input_ids,(torch.ones(self.max_len - len(input_ids)) * self.pad).long()))
            
        else:
            input_ids = input_ids[: self.max_len]

        return input_ids, attention_mask, grade


class ConvNet(nn.Module):
    
    def __init__(self): # input is (32,100,768)
        super(ConvNet, self).__init__()
        
        self.CNN = nn.ModuleList(
            [nn.Conv2d(1, 3, (K, 768)) for K in [2, 3, 4, 8]])
        
        self.drop_out = nn.Dropout(0.5)
        
        self.linear1 = nn.Linear(4*3, 4)
        
        self.linear2 = nn.Linear(4, 1)
        
    def forward(self, x):
        x = Variable(x)
        x = x.unsqueeze(1) #(32,1,100,768)
        x = [nn.functional.relu(cnn(x)).squeeze(3) for cnn in self.CNN] #(32,3,)
        x = [nn.functional.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
        x = torch.cat(x, 1)
        x = self.drop_out(x) # (32, 12)
        x = self.linear1(x)
        x = nn.functional.relu(x)
        x = self.linear2(x)
        x = nn.functional.relu(x)
        return x
    
    
""" Main """
criterion = nn.MSELoss()
model = ConvNet()

Rmse = dict()
Acc = dict()
for model_name in models_names:
    
    print('model:', model_name)
    
    fname1 = model_name + "_task1"
    fname2 = model_name + "_task2"
    
    SAVE_PATH_task1 = os.path.join("results/" + fname1 + "_prediction.csv");
    SAVE_PATH_task2 = os.path.join("results/" + fname2 + "_test.csv");
    
    Model_PATH = os.path.join("./" + fname1 + "_model.pt");
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)   
    
    embedding_model = AutoModel.from_pretrained(model_name)
    
    # Task 1:
    print('Task 1')
    
    task1_train_df = task1_train_df[["original", "edit", "meanGrade"]]
    
    task1_valid_df = task1_valid_df[["original", "edit", "meanGrade"]]
    task1_test_df = task1_test_df[["id", "original", "edit", "meanGrade"]]
    
    if model_name == 'bert-base-uncased' or model_name == 'albert-base-v1':
    
        train_dset = two_sentence_dataset_1(task1_train_df, tokenizer, max_len=100)
        valid_dset = two_sentence_dataset_1(task1_valid_df, tokenizer, max_len=100)
        test_dset = two_sentence_dataset_1(task1_test_df, tokenizer, max_len=100)
        
    else:
        
        train_dset = two_sentence_dataset_2(task1_train_df, tokenizer, max_len=100)
        valid_dset = two_sentence_dataset_2(task1_valid_df, tokenizer, max_len=100)
        test_dset = two_sentence_dataset_2(task1_test_df, tokenizer, max_len=100)
            
 
    train_loader = DataLoader(train_dset, batch_size=32, shuffle=True, num_workers=0)
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
                
                if len(data) == 4:                   
                    input_ids, token_type_ids, attention_mask, grade = data
                    
                    with torch.no_grad():
                        output = embedding_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                    
                else:
                    input_ids, attention_mask, grade = data
                    
                    with torch.no_grad():
                        output = embedding_model(input_ids=input_ids, attention_mask=attention_mask)
                    
                x = output[0];
                
                outputs = model(x)
                
                optimizer.zero_grad()

                grade = grade.reshape(grade.size(0), -1)

                loss = criterion(outputs, grade)
        
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
                    
                    if len(data) == 4:                   
                        input_ids, token_type_ids, attention_mask, grade = data
                        
                        with torch.no_grad():
                            output = embedding_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                        
                    else:
                        input_ids, attention_mask, grade = data
                        
                        with torch.no_grad():
                            output = embedding_model(input_ids=input_ids, attention_mask=attention_mask)
                    
                    x = output[0];
                    
                    outputs = model(x)
                    
                    grade = grade.reshape(grade.size(0), -1)

                    loss = criterion(outputs, grade)
        
                    validation_loss += loss.item()
        
            VALIDATION_LOSS = validation_loss/len(valid_dset);
            print("epoch ", epoch+1, ",    Validation loss = ", VALIDATION_LOSS, "\n")
        
            if VALIDATION_LOSS < min_loss:
                print("best model for", model_name, "so far was found\n")
                min_loss = VALIDATION_LOSS;
                torch.save(model.state_dict(), Model_PATH)


    # Test the best model for task 1
    print('testing the', model_name, 'model for task 1 ...')
    
    preds = []
    model.load_state_dict(torch.load(Model_PATH))
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            
            if len(data) == 4:                   
                input_ids, token_type_ids, attention_mask, grade = data
                
                with torch.no_grad():
                    output = embedding_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                
            else:
                input_ids, attention_mask, grade = data
                
                with torch.no_grad():
                    output = embedding_model(input_ids=input_ids, attention_mask=attention_mask)
                        
            x = output[0];
            
            outputs = model(x)
    
            for result in outputs:
                preds.append(result.item())
    
    prediction_df = pd.DataFrame(columns=["id", "pred"])
    prediction_df["id"] = task1_test_df["id"]
    prediction_df["pred"] = preds
    assert len(prediction_df) == len(task1_test_df)
    prediction_df.to_csv(SAVE_PATH_task1, index=False)
    
    Rmse[model_name] = score_task_1("semeval-2020-task-7-data-full/task-1/test.csv", SAVE_PATH_task1)
    
    
    # Task 2:
    print('testing the', model_name, 'model for task 2 ...')
    
    test_df_1 = task2_test_df[["id", "original1", "edit1", "meanGrade1"]]
    test_df_2 = task2_test_df[["id", "original2", "edit2", "meanGrade2"]]
    
    test_df_1.rename(columns={"original1": "original", "edit1": "edit", "meanGrade1": "meanGrade"},inplace=True)
    test_df_2.rename(columns={"original2": "original", "edit2": "edit", "meanGrade2": "meanGrade"},inplace=True)
    
    if model_name == 'bert-base-uncased' or model_name == 'albert-base-v1':
    
        test_dset_1 = two_sentence_dataset_1(test_df_1, tokenizer, max_len=100)
        test_dset_2 = two_sentence_dataset_1(test_df_2, tokenizer, max_len=100)
        test_sets = [test_dset_1,test_dset_1];
            
    else:
        
        test_dset_1 = two_sentence_dataset_2(test_df_1, tokenizer, max_len=100)
        test_dset_2 = two_sentence_dataset_2(test_df_2, tokenizer, max_len=100)
        test_sets = [test_dset_1,test_dset_1];
            
    test_loader_1 = DataLoader(test_dset_1, batch_size=32, shuffle=False, num_workers=0)
    test_loader_2 = DataLoader(test_dset_2, batch_size=32, shuffle=False, num_workers=0)

    for jj, test_loader in enumerate([test_loader_1, test_loader_2]):
        
        preds = [];
        test_dset = test_sets[jj];
        with torch.no_grad():
            for ii, data in enumerate(test_loader):
                
                if len(data) == 4:                   
                    input_ids, token_type_ids, attention_mask, grade = data
                    
                    with torch.no_grad():
                        output = embedding_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                    
                else:
                    input_ids, attention_mask, grade = data
                    
                    with torch.no_grad():
                        output = embedding_model(input_ids=input_ids, attention_mask=attention_mask)
                        
                x = output[0];
                
                outputs = model(x)
    
                for result in outputs:
                    preds.append(result.item())
                
                print("prediction for edit %d,    batch percentage: %.2f %%" % (jj+1, 100*min(1, 32*(ii+1)/len(test_dset))))
                
        if jj==0:
            pred1 = preds;
            
        else:
            pred2 = preds;
    
    final_preds = []
    for ii,_ in enumerate(pred1):
        if pred1[ii] > pred2[ii]:
            final_preds.append(1)
        elif pred1[ii] < pred2[ii]:
            final_preds.append(2)
        else:
            final_preds.append(0)
    
    prediction_df = pd.DataFrame(columns=["id", "pred"])
    prediction_df["id"] = test_df_1["id"]
    prediction_df["pred"] = final_preds
    assert len(prediction_df) == len(test_df_1)
    
    prediction_df.to_csv(SAVE_PATH_task2, index=False)
    
    Acc[model_name] = score_task_2("semeval-2020-task-7-data-full/task-2/test.csv", SAVE_PATH_task2)
    
    
print('\nRMSE for Task 1 with the best model of each BERT-based word embedding model:\n')
for key in Rmse:
    print(key, ' : ', Rmse[key])

print('\nAccuracy for Task 2 with the best model of each BERT-based word embedding model:\n')
for key in Acc:
    print(key, ' : ', Acc[key])