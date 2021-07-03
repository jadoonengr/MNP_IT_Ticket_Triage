# Board Testing Package
### General Packages ###
import pandas as pd
import random
from datetime import datetime

### For Input Processing ###
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizer, RobertaModel, AdamW, get_linear_schedule_with_warmup

### Performance Metrics ###
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report

### For Model Exporting ###
from joblib import dump, load

### Repress SettingWithCopyWarning ###
from ast import literal_eval
pd.options.mode.chained_assignment = None 

### Training Packages ROBERTA ###
import torch
from torchtext.legacy.data import Field, TabularDataset, BucketIterator, Iterator

##############################################################################################
### Global Variables ###
max_len = 512
MAX_SEQ_LEN = 256
BATCH_SIZE = 16
NUM_FEATURES = 5

# RoBERTa Tokenization
random.seed(datetime.now())

# Initialize tokenizer.
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

# Set tokenizer hyperparameters.
PAD_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
UNK_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)


########################################
# Set random seed and set device to GPU.
torch.manual_seed(17)

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
else:
    device = torch.device('cpu')

print("Compute engine used: ", device)



##############################################################################################
##############################################################################################
# Functions for saving and loading model parameters and metrics.
def save_checkpoint(path, model, valid_loss):
    torch.save({'model_state_dict': model.state_dict(),
                  'valid_loss': valid_loss}, path)

    
def load_checkpoint(path, model):    
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict['model_state_dict'])
    
    return state_dict['valid_loss']


def save_metrics(path, train_loss_list, valid_loss_list, global_steps_list):   
    state_dict = {'train_loss_list': train_loss_list,
                  'valid_loss_list': valid_loss_list,
                  'global_steps_list': global_steps_list}
    
    torch.save(state_dict, path)


def load_metrics(path):    
    state_dict = torch.load(path, map_location=device)
    return state_dict['train_loss_list'], state_dict['valid_loss_list'], state_dict['global_steps_list']


##############################################################################################
##############################################################################################
# Model with extra layers on top of RoBERTa
class ROBERTAClassifier(torch.nn.Module):
    def __init__(self, num_feature_size, dropout_rate=0.3):
        super(ROBERTAClassifier, self).__init__()
        
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.d1 = torch.nn.Dropout(dropout_rate)
        self.l1 = torch.nn.Linear(768, 64)
        self.bn1 = torch.nn.LayerNorm(64)
        self.d2 = torch.nn.Dropout(dropout_rate)
        self.l2 = torch.nn.Linear(64, 3)
        # self.l2 = torch.nn.Linear(64+num_feature_size, 3)
        # self.softmax = torch.nn.Softmax(dim=1)
        
    def forward(self, input_ids, attention_mask, nn_input_meta):
        _, x = self.roberta(input_ids=input_ids, attention_mask=attention_mask,return_dict = False)
        x = self.d1(x)
        x = self.l1(x)
        x = self.bn1(x)
        x = torch.nn.Tanh()(x)
        x = self.d2(x)
        # x = torch.cat((x, nn_input_meta[0].reshape(-1,1).float(),
        #               nn_input_meta[1].reshape(-1,1).float(),
        #               nn_input_meta[2].reshape(-1,1).float(),
        #               nn_input_meta[3].reshape(-1,1).float(),
        #               nn_input_meta[4].reshape(-1,1).float()), dim=1)
        x = self.l2(x)
        # x = self.softmax(x)
        return x


##############################################################################################
##############################################################################################
def pretrain(model, 
            train_iter, 
            valid_iter,
            optimizer,
            valid_period,
            scheduler = None,
            num_epochs = 5):

    # Pretrain linear layers, do not train bert
    for param in model.roberta.parameters():
        param.requires_grad = False
    
    model.train()
    
    # Initialize losses and loss histories
    train_loss = 0.0
    valid_loss = 0.0   
    global_step = 0  
    
    # Train loop
    for epoch in range(num_epochs):
        for (txt, target), _ in train_iter:
            mask = (txt != PAD_INDEX).type(torch.uint8)
            y_pred = model(input_ids=txt,  
                          attention_mask=mask,
                          nn_input_meta=[])
            loss = torch.nn.CrossEntropyLoss()(y_pred, target)
  
            loss.backward()
            
            # Optimizer and scheduler step
            optimizer.step()    
            scheduler.step()
                
            optimizer.zero_grad()
            
            # Update train loss and global step
            train_loss += loss.item()
            global_step += 1

            # Validation loop. Save progress and evaluate model performance.
            if global_step % valid_period == 0:
                model.eval()
                
                with torch.no_grad():                    
                    for (txt, target), _ in valid_iter:
                        mask = (txt != PAD_INDEX).type(torch.uint8)
                        
                        y_pred = model(input_ids=txt, 
                                      attention_mask=mask,
                                      nn_input_meta=[])
                        
                        loss = torch.nn.CrossEntropyLoss()(y_pred, target)
                        
                        valid_loss += loss.item()

                # Store train and validation loss history
                train_loss = train_loss / valid_period
                valid_loss = valid_loss / len(valid_iter)
                
                model.train()

                # print summary
#                 print('Epoch [{}/{}], global step [{}/{}], PT Loss: {:.4f}, Val Loss: {:.4f}'
#                       .format(epoch+1, num_epochs, global_step, num_epochs*len(train_iter),
#                               train_loss, valid_loss))
                
                train_loss = 0.0                
                valid_loss = 0.0
    
    # Set bert parameters back to trainable
    for param in model.roberta.parameters():
        param.requires_grad = True
        
    print('Pre-training done!')


##############################################################################################
##############################################################################################
# Training Function

def train(model,
          train_iter,
          valid_iter,
          optimizer,
          valid_period,
          scheduler = None,
          num_epochs = 5,
          output_path = "",
          model_name = "roberta_board"):

    # Initialize losses and loss histories
    train_loss = 0.0
    valid_loss = 0.0
    train_loss_list = []
    valid_loss_list = []
    best_valid_loss = float('Inf')
    
    global_step = 0
    global_steps_list = []
    
    model.train()
    
    # Train loop
    for epoch in range(num_epochs):
        for (txt, target), _ in train_iter:
            mask = (txt != PAD_INDEX).type(torch.uint8)
            
            y_pred = model(input_ids=txt, 
                            attention_mask=mask,
                            nn_input_meta=[])
                        
            #output = model(input_ids=source,
            #              labels=target,
            #              attention_mask=mask)
            
            loss = torch.nn.CrossEntropyLoss()(y_pred, target)
            #loss = output[0]
            
            loss.backward()
            
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            
            # Optimizer and scheduler step
            optimizer.step()    
            scheduler.step()
                
            optimizer.zero_grad()
            
            # Update train loss and global step
            train_loss += loss.item()
            global_step += 1

            # Validation loop. Save progress and evaluate model performance.
            if global_step % valid_period == 0:
                model.eval()
                
                with torch.no_grad():                    
                    for (txt, target), _ in valid_iter:
                        mask = (txt != PAD_INDEX).type(torch.uint8)
                        
                        y_pred = model(input_ids=txt, 
                                      attention_mask=mask,
                                      nn_input_meta=[])
                        #output = model(input_ids=source,
                        #               labels=target,
                        #               attention_mask=mask)
                        
                        loss = torch.nn.CrossEntropyLoss()(y_pred, target)
                        #loss = output[0]
                        
                        valid_loss += loss.item()

                # Store train and validation loss history
                train_loss = train_loss / valid_period
                valid_loss = valid_loss / len(valid_iter)
                train_loss_list.append(train_loss)
                valid_loss_list.append(valid_loss)
                global_steps_list.append(global_step)

                # print summary
#                 print('Epoch [{}/{}], global step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}'
#                       .format(epoch+1, num_epochs, global_step, num_epochs*len(train_iter),
#                               train_loss, valid_loss))
                
                # checkpoint
                if best_valid_loss > valid_loss:
                    best_valid_loss = valid_loss
                    save_checkpoint(output_path + '/'+model_name+'.pkl', model, best_valid_loss)
                    save_metrics(output_path + '/'+model_name+'_metric.pkl', train_loss_list, valid_loss_list, global_steps_list)
                        
                train_loss = 0.0                
                valid_loss = 0.0
                model.train()
    
    save_metrics(output_path + '/'+model_name+'_metric.pkl', train_loss_list, valid_loss_list, global_steps_list)
    print('Training done!')

##############################################################################################
##############################################################################################
# Evaluation Function

def evaluate(model, test_loader):
    y_pred = []
    y_prob = []
    y_true = []
    sm = torch.nn.Softmax(dim=1)

    model.eval()
    with torch.no_grad():
        for (txt, target), _ in test_loader:
            mask = (txt != PAD_INDEX).type(torch.uint8)
            
            output = model(input_ids=txt, 
                            attention_mask=mask,
                            nn_input_meta=[])
            # print(output)       
            d = sm(output)
            y_pred.extend(torch.argmax(output, axis=-1).tolist())
            
            y_prob.extend([max(sublist) for sublist in d.tolist()])

            y_true.extend(target.tolist())
    
#     print('Classification Report:')
#     res = classification_report(y_true, y_pred, labels=[0,1,2], digits=4)
#     print(res)
    
#     print('Confusion Matrix:')
#     res = confusion_matrix(y_true, y_pred, labels=[0,1,2])
#     print(res)

    return pd.DataFrame(zip(y_pred, y_prob), columns=['Board_Predicted', 'Board_Probability'])





##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################


def format_inputs(df, split=0.1):
    '''
    Prepare input and tokenize for RoBERTa
    ---
    Parameters:
        df (dataframe): Dataframe that contains the Text.
        split (float): Percent of values used for model validation. Between 0 and 1.
    Returns:
        train_iter (dataframe): Training data iterator object.
        valid_iter (dataframe): Validation data iterator object.
    '''
    max_len = 512
    ##############################################################################################
    # Roberta only allows labels starting with 0,1,2
    df['Board'] = df.Board.map({36:0, 41:1, 43:2})
    df['combined_text'] = df['combined_text'].apply(lambda x: " ".join(x.split()[:max_len]))
    # d = {'DeskDirector':'s1',	'Email':'s2',	'Email Connector':'s3',	'Escalation':'s4',	'Renewal':'s5'}
    # src = pd.get_dummies(df.Source).rename(columns = d, inplace = False)
    # df = pd.concat([df, src], axis =1)

    train, valid = train_test_split(df, test_size=split, random_state=42)
    train.to_csv("./RoBERTa_Files/train.csv")
    valid.to_csv("./RoBERTa_Files/valid.csv")

    ##############################################################################################
    # Define columns to read.
    LABEL = Field(preprocessing=lambda x: float(x), sequential=False, use_vocab=False, batch_first=True)
    SOURCE = Field(preprocessing=lambda x: float(x), sequential=False, use_vocab=False, batch_first=True)
    TEXT = Field(use_vocab=False, 
                      tokenize=tokenizer.encode, 
                      include_lengths=False, 
                      batch_first=True,
                      fix_length=MAX_SEQ_LEN, 
                      pad_token=PAD_INDEX, 
                      unk_token=UNK_INDEX)

    fields = {'combined_text' : ('combined_text', TEXT), 
                # 's1' : ('s1', SOURCE), 
                # 's2' : ('s2', SOURCE), 
                # 's3' : ('s3', SOURCE), 
                # 's4' : ('s4', SOURCE), 
                # 's5' : ('s5', SOURCE), 
                'Board' : ('Board', LABEL)}


    # Read preprocessed CSV into TabularDataset and split it into train, test and valid.
    train_data = TabularDataset(path=f"./RoBERTa_Files/train.csv", format='csv', fields=fields, skip_header=False)
    valid_data = TabularDataset(path=f"./RoBERTa_Files/valid.csv", format='csv', fields=fields, skip_header=False)

    # Create train and validation iterators.
    train_iter, valid_iter = BucketIterator.splits((train_data, valid_data),
                                                batch_size=BATCH_SIZE,
                                                device=device,
                                                shuffle=True,
                                                sort_key=lambda x: len(x.combined_text), 
                                                sort=True, 
                                                sort_within_batch=False)


    return train_iter, valid_iter


##############################################################################################
##############################################################################################    
def train_roberta(train_iter, valid_iter, model_name = 'roberta_board', model_path =".", pretrain_epoch=6, train_epoch=12):
    '''
    Train a RoBERTA classifier model using Board.
    ---
    Parameters:
        train_iter (iterator object): RoBERTa Encoded Text for training data.
        valid_iter (iterator object): RoBERTa Encoded Text for validation data.
        model_path (str): Output path of model
        pretrain_epoch (int): No. of epochs to train model linear layers only.
        train_epoch (int): No. of epochs to train all model layers.
    ---
    Returns:
        None: Model internally saves trained model as .pkl file.
    '''
    
    # Main training loop
    NUM_EPOCHS = pretrain_epoch
    steps_per_epoch = len(train_iter)

    model = ROBERTAClassifier(num_feature_size=NUM_FEATURES, dropout_rate=0.4)
    model = model.to(device)


    opt1 = AdamW(model.parameters(), lr=1e-4)
    sch1 = get_linear_schedule_with_warmup(opt1, 
                                                num_warmup_steps=steps_per_epoch*1, 
                                                num_training_steps=steps_per_epoch*NUM_EPOCHS)

    print("======================= Start pretraining ==============================")

    pretrain(model=model,
            train_iter=train_iter,
            valid_iter=valid_iter,
            optimizer=opt1,
            valid_period = len(train_iter),
            scheduler=sch1,
            num_epochs=NUM_EPOCHS)

    NUM_EPOCHS = train_epoch
    print("======================= Start training =================================")
    opt2 = AdamW(model.parameters(), lr=2e-6)
    sch2 = get_linear_schedule_with_warmup(opt2, 
                                                num_warmup_steps=steps_per_epoch*2, 
                                                num_training_steps=steps_per_epoch*NUM_EPOCHS)
    train(model=model, 
        train_iter=train_iter, 
        valid_iter=valid_iter, 
        optimizer=opt2, 
        valid_period = len(train_iter),
        scheduler=sch2, 
        num_epochs=NUM_EPOCHS,
        output_path = model_path,
        model_name = model_name)

    # Display training results
#     print('\n=====Training Metrics=====')
#     train_set = evaluate(model, train_iter)
#############################################################################################
#############################################################################################
def predict_roberta(df,path="./Saved_Models/Board/roberta_board.pkl"):
    '''
    Predicts the output classification and probability of certainty for Board.
    ---
    Parameters:
        X_predict (dataframe): Dataframe containing plain text.
        path (str): Path to where pre-trained model is located.
    ---
    Returns:
        Output (dataframe): Original dataframe containing two extra columns for the final prediction (Board_Prediction) and the classification probability for the most likely outcome (Board_Probability).
    '''
    
    ##############################################################################################
    # Roberta only allows labels starting with 0,1,2
    df['Board'] = df.Board.map({36:0, 41:1, 43:2})
    df['combined_text'] = df['combined_text'].apply(lambda x: " ".join(x.split()[:max_len]))
    # d = {'DeskDirector':'s1',	'Email':'s2',	'Email Connector':'s3',	'Escalation':'s4',	'Renewal':'s5'}
    # src = pd.get_dummies(df.Source).rename(columns = d, inplace = False)
    # df = pd.concat([df, src], axis =1)
    df.to_csv("./RoBERTa_Files/test.csv")

    # Define columns to read.
    LABEL = Field(preprocessing=lambda x: float(x), sequential=False, use_vocab=False, batch_first=True)
    # SOURCE = Field(preprocessing=lambda x: float(x), sequential=False, use_vocab=False, batch_first=True)
    TEXT = Field(use_vocab=False, 
                      tokenize=tokenizer.encode, 
                      include_lengths=False, 
                      batch_first=True,
                      fix_length=MAX_SEQ_LEN, 
                      pad_token=PAD_INDEX, 
                      unk_token=UNK_INDEX)

    fields = {'combined_text' : ('combined_text', TEXT), 
                # 's1' : ('s1', SOURCE), 
                # 's2' : ('s2', SOURCE), 
                # 's3' : ('s3', SOURCE), 
                # 's4' : ('s4', SOURCE), 
                # 's5' : ('s5', SOURCE), 
                'Board' : ('Board', LABEL)}


    # Read preprocessed CSV into TabularDataset.
    test_data = TabularDataset(path=f"./RoBERTa_Files/test.csv", format='csv', fields=fields, skip_header=False)

    # Test iterator, no shuffling or sorting required.
    test_iter = Iterator(test_data, batch_size=BATCH_SIZE, device=device, train=False, shuffle=False, sort=False)
    
    # Evalution
    model = ROBERTAClassifier(num_feature_size=NUM_FEATURES)
    model = model.to(device)

    load_checkpoint(path, model)

    #print('\n=====Prediction Metrics=====')
    test_set = evaluate(model, test_iter)
    
    return pd.concat([df, test_set], axis=1)