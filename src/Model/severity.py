# Severity Testing Package
### General Packages ###
import pandas as pd
import datetime

### For Embeddings ###
from transformers import DistilBertModel, DistilBertTokenizer
from bert_tokenizer import BERT_Tokenizer

### For Model Exporting ###
from joblib import dump, load

### Training Packages ###
from sklearn.svm import SVC

### Prediction Packages ###
import model_functions as mf


def add_board_predictions(X,board_predict):
    ''' 
    Combined Board predictions into Dataframe.
    ---
    Parameters:
        X (dataframe): Cleaned Data with Text, Source, Severity. 
        board_predict (list): List of predicted Board values from the roBERTa model.
    ---
    Returns:
        df (datafrane): X dataframe with predicted board appeneded.
    '''
    #Copy dataframe
    df = X.copy()
    Y_board_predicted = board_predict
    #One-Hot-Encode Board Predictions
    board36 = [0]*len(df)
    board41 = [0]*len(df)
    board43 = [0]*len(df)
    
    for i in range(0,len(df)):
        if Y_board_predicted[i] == 36:
            board36[i] = 1
        elif Y_board_predicted[i] == 41:
            board41[i] = 1
        elif Y_board_predicted[i] == 43:
            board43[i] = 1
    df.brd36 = board36
    df.brd41 = board41
    df.brd43 = board43
    
    return df

def format_inputs(df,max_len=100,verbose = 2):
    '''
    BERT Encode text and combine with Board Predictions and Source
    ---
    Parameters:
        df (dataframe): Dataframe that contains the Text, one-hot-encoded (OHE) predicted Board predictions and Source. Typically the return object of `add_board_predictions()`
        max_len (int): Max token length for the BERT encoder. Values are between 0 and 512 (maximum length for BERT).
    Returns:
        X_train (dataframe): Dataframe of the features used for modelling. Number of columns will be equal to the 768 embeddings from the BERT encode + 5 Sources (OHE) + 3 Boards (OHE). Number of rows will be equal to the number of observations.
    '''
    if max_len > 512:
        print("BERT Encoder has a max length of 512. Max length has been set to 512.")
        max_len = 512
    if max_len > 250:
        print("BERT Encoding can be very expensive/taxing depending on max_length size and dataset")
    
    #Initializing BERT Tokenization
    if verbose == 2:
        print("--Importing pre-trained BERT model and tokenizer (1/3)--")
    model_class, tokenizer_class, pretrained_weights = (DistilBertModel, DistilBertTokenizer, 'distilbert-base-uncased')
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    bert_model = model_class.from_pretrained(pretrained_weights)
    
    #Tokenize Text
    if verbose == 2:
        print("--Tokenizing Text. May take a while. (2/3)--")
    X_text = BERT_Tokenizer(model=bert_model,tokenizer=tokenizer,text=df.combined_text,max_len=max_len)
    X_train = pd.DataFrame(X_text)
    
    #Combine Text with Source and Board
    if verbose == 2:
        print("--Combining Source and Board (3/3)--")
    source_board = ["email_connector","deskdirector","email","renewal","escalation","brd36","brd41","brd43"]
    for i in source_board:
        X_train[i] = df[i]  
    
    return X_train

    
def train_svm(X_train,Y_actual,save_model = "Y", model_name = "severity_train",export_path = "./Saved_Models/Severity", verbose=2):
    '''
    Train an SVM model using Board, BERT Encoded Text and Source. SVM model will be exported as 'severity_train.joblib'.
    ---
    Parameters:
        X_train (dataframe): BERT Encoded Text, one-hot-encoded (OHE) predicted Board and Source.
        Y_actual (list,array): True labels for Severity, between 0-2 (low-high).
        save_model (str): If 'Y', model will be saved as a `.joblib` file at export_path. If 'N', model will not be saved.
        model_name (str): Specifies the name of the .joblib file.
        export_path (str): Export path for the resulting model.
        verbose (int): 2 for all the print statements. 1 for no print statements.
    ---
    Returns:
        svm_model (sklearn.svc object): Returns trained SVM model.
        prob_df (dataframe): Dataframe containing the classification probabilities and final prediction (class with the max probability).
    '''
    
    #Train SVM model
    if save_model == "Y" and verbose == 2:
        print("--Fitting Model (1/2)--")
    elif verbose == 2:
        print("--Fitting Model--")
    svm_model  = SVC(C=0.06, class_weight='balanced', gamma=0.01, kernel='linear')
    svm_model.fit(X_train,Y_actual)
    
    #Export model
    if save_model == "Y":
        if verbose == 2:
            print("--Exporting Model (2/2)--")
        file_name = export_path+"/"+model_name+".joblib"
        dump(svm_model, file_name) 
    
    if verbose == 2:
        print("--Done--")
    
    prob_df = mf.extract_probabilities(model = svm_model,X = X_train,Y= Y_actual)
    
    return svm_model, prob_df

def predict_svm(X_predict, import_path = "./Saved_Models/severity_train.joblib", verbose = 2):
    '''
    Applies a previously constructed SVM model to a dataset. The output is a dataframe with the predicted values.
    ---
    Parameters:
        X_predict (dataframe): BERT Encoded Text, one-hot-encoded (OHE) predicted Board and Source.
        import_path (str): Export path for the resulting model.
        verbose (int): 2 for all the print statements. 1 for no print statements.
    ---
    Returns:
        prob_df (dataframe): Dataframe containing the classification probabilities and final prediction (class with the max probability).
    '''

    #Load pre-trained model
    if verbose == 2:
        print("--Load pre-trained model (1/2)--")
    svm_model = load(import_path)
    #Get prediction probabilities
    if verbose ==2:
        print("--Predicting and Extracting probabilities (2/2)--")
    prob_df = mf.extract_probabilities(model = svm_model,X = X_predict,Y = None)
    
    if verbose ==2:
        print("--Done--")
    
    return prob_df