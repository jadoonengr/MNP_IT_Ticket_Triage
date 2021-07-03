import time

import redis
from flask import Flask

app = Flask(__name__)
cache = redis.Redis(host='redis', port=6379)

def get_hit_count():
    retries = 5
    while True:
        try:
            return cache.incr('hits')
        except redis.exceptions.ConnectionError as exc:
            if retries == 0:
                raise exc
            retries -= 1
            time.sleep(0.5)

# @app.route('/')
# Train All Packages
import pandas as pd
import datetime
import warnings
import numpy as np

### For Model Exporting ###
from joblib import dump, load

### Ticket triage functions ###
import sys
import os.path
sys.path.append("src/auxiliary/")
sys.path.append("src/cleaning/")
sys.path.append("src/model/")
sys.path.append("src/tokenizer/")

### For pre-processing ###
import ticket_cleaner
import bert_tokenizer

### For Modelling ###
import board, severity, impact

### For Probabilities, Modified Accuracy Score ###
import model_functions

#############################################################################
# Metrics Package
### Metrics for Evaluation ###
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, accuracy_score, f1_score

### To Write out results ###
import csv
#############################################################################
def step_1_read_in(datapath = "./Data/Tickets with Classifications.xlsx"):
    '''
    Read in data and clean it using ticket_cleaner's clean_ticket function.
    ---
    Parameters:
        datapath (str/dataframe): Either the path to the datafile or a dataframe. MUST include the following headers (ticketNbr, contact_name, company_name, Summary, Initial_Description, SR_Impact_RecID, SR_Severity_RecID, SR_Board_RecID), Source and date_entered).
    Returns:
        data (dataframe): Cleaned dataframe with cleaned text and One Hot Encoded (OHE) Source and Board.
    '''
    #Check if xlsx, csv or dataframe is input
    if isinstance(datapath,pd.DataFrame):
        rawdata = datapath.copy()
    elif datapath.split(".")[-1] == "xlsx":
        rawdata = pd.read_excel(datapath)
    elif datapath.split(".")[-1] == "csv":
        rawdata = pd.read_csv(datapath)
        
    data = ticket_cleaner.clean_tickets(ticketNbr = rawdata.ticketNbr, 
                                        contact_name = rawdata.contact_name, 
                                        company_name = rawdata.company_name, 
                                        Summary = rawdata.Summary, 
                                        Initial_Description = rawdata.Initial_Description, 
                                        Impact = rawdata.SR_Impact_RecID, 
                                        Severity = rawdata.SR_Severity_RecID, 
                                        Board = rawdata.SR_Board_RecID, 
                                        Source = rawdata.Source, 
                                        date_entered = rawdata.date_entered)
    
    return data
#############################################################################
def board_train(dataset,
                mod_name = "roberta_board", 
                mod_path = "./Saved_Models/Board",
                pt_epoch = 1, 
                t_epoch = 1,
                verbose = 2):
    '''
    Train the RoBERTa model for Board. Will create a mod_name.pkl file at the mod_path location.
    ---
    Parameters:
        dataset (dataframe): Cleaned dataframe from ticket_cleaner.clean_tickets function. 
        mod_name (str): Name of the model. If name is the same as an exisiting model, it will overwrite it. If you don't want to store multiple models just keep default value.
        mod_path (str): Path to where you want to save model. If you don't want to store multiple models just keep default value.
        pt_epoch (int): Pre-training epoch. Suggested 6, but will increase run-time.
        t_epoch (int): Training epoch. Suggested 12, but will increase run-time.
        verbose (int): Control amount of text displayed during training.
    Returns:
        board_results (dataframe): Dataframe displaying text, actual labels, predicted labels and prediction probabilities. 
    '''
    #Functions alter the dataset you put in so need to make copies 
    copied_dataset = dataset.copy()
    if verbose == 2:
        print('--Start RoBERTa Training--')
    #Create Train/Valid split on Dataset x
    train_iter, valid_iter = board.format_inputs(copied_dataset, split = 0.1)
    #Pre-train/Train roBERTa which outputs .pkl model
    board.train_roberta(train_iter, valid_iter,model_name = mod_name, model_path = mod_path,pretrain_epoch = pt_epoch, train_epoch = t_epoch)
    if verbose == 2:
        print('--Done RoBERTa Training--')
    #Get Board Predictions
    if verbose == 2:
        print('--Board Predictions for Severity and Imapct--')
    #Functions alter the dataset you put in so need to make copies 
    predict_dataset = dataset.copy()
    board_results = board.predict_roberta(df = predict_dataset,path=mod_path + "/" + mod_name + ".pkl")
    if verbose == 2:
        print("--DONE--")
    #Returns df with Actual Board and Predicted Board
    board_results['Board'] = board_results.Board.map({0:36, 1:41, 2:43})
    board_results['Board_Predicted'] = board_results.Board_Predicted.map({0:36, 1:41, 2:43})
    return board_results
#############################################################################
def berttext_board_source(dataset,predicted_board,max_len=100,verbose = 2):
    '''
    Combined the Board predictions from model to dataset.
    ---
    Parameters:
        dataset (dataframe): Cleaned dataframe from ticket_cleaner.clean_tickets function. 
        predicted_board (list): List of Board predictions for the dataset.
        max_len (int): Maximum length for BERT embedding. Max value is 512 but we found 100 to be a good balance between performance and run-time.
        verbose (int): Control amount of text displayed during training.
    Returns:
        berttext_board_source (dataframe): Dataframe with embedded text, OHE board and source/.
    '''
    if verbose == 2:
        print('--Merge Board Predictions with Text and Source for Severity and Impact--')
    #Combined Text and Source with Board Predictions
    text_board_source = severity.add_board_predictions(dataset,board_predict=predicted_board)

    #BERT Encode Text
    berttext_board_source = severity.format_inputs(text_board_source,max_len, verbose = verbose)
    
    if verbose == 2:
        print("--DONE--")
    
    return berttext_board_source
#############################################################################
def severity_train(features, 
                   Y_varibles, 
                   mod_name = "svm_severity", 
                   mod_path = "./Saved_Models/Severity",
                   save = "Y",
                   verbose = 2):
    '''
    Train the SVM model for Severity. Returns trained model and trained predictions. Will create a mod_name.joblib file at the mod_path location. 
    ---
    Parameters:
        features (dataframe): Dataframe with BERT embedded text, OHE Board and Source from berttext_board_source function.
        Y_variable (list): List of Actual Severity Labels for observations.
        mod_name (str): Name of the model. If name is the same as an exisiting model, it will overwrite it. If you don't want to store multiple models just keep default value.
        mod_path (str): Path to where you want to save model. If you don't want to store multiple models just keep default value.
        save (str): "Y" to save model at mod_path. "N" to not save the model. 
        verbose (int): Control amount of text displayed during training.
    Returns:
        severity_model (sklearn.svc object): Trained SVM model.
        severity_train (dataframe): Dataframe containing the classification probabilities and final prediction (class with the max probability).
    '''
    #Train SVM model which outputs .joblib file
    severity_model, severity_train = severity.train_svm(features,
                                                        Y_varibles, 
                                                        model_name = mod_name, 
                                                        save_model = save, 
                                                        export_path = mod_path, 
                                                        verbose = verbose)
    
    return severity_model, severity_train
#############################################################################
def impact_train(features, 
                 Y_varibles, 
                 mod_name = "svm_impact",
                 mod_path = "./Saved_Models/Impact",
                 save = "Y",
                 verbose = 2):
    '''
    Train the SVM model for Impact. Returns trained model and trained predictions. Will create a mod_name.joblib file at the mod_path location. 
    ---
    Parameters:
        features (dataframe): Dataframe with BERT embedded text, OHE Board and Source from berttext_board_source function.
        Y_variable (list): List of Actual Impact Labels for observations.
        mod_name (str): Name of the model. If name is the same as an exisiting model, it will overwrite it. If you don't want to store multiple models just keep default value.
        mod_path (str): Path to where you want to save model. If you don't want to store multiple models just keep default value.
        save (str): "Y" to save model at mod_path. "N" to not save the model. 
        verbose (int): Control amount of text displayed during training.
    Returns:
        impact_model (sklearn.svc object): Trained SVM model.
        impact_train (dataframe): Dataframe containing the classification probabilities and final prediction (class with the max probability).
    '''
    #Train SVM model which outputs .joblib file
    impact_model, impact_train = impact.train_svm(features,
                                                  Y_varibles, 
                                                  model_name = mod_name, 
                                                  save_model = save, 
                                                  export_path = mod_path, 
                                                  verbose= verbose)
    
    return impact_model, impact_train
#############################################################################
def train_all(path_to_data,
              model_name="mod1",
              pretrain_epoch=1,
              train_epoch=1,
              max_token_len = 100, 
              save_models = "Y",
              verbose = 2):
    '''
    Train on ALL data.
    Function to: Intake ALL training data -> Clean data -> Model Board -> Model Severity -> Model Impact. Models will be stored in Ticket_Triage/Saved_Models/
    ---
    Parameters:
        path_to_data (str/dataframe): Either the path to the datafile or a dataframe. MUST include the following headers (ticketNbr, contact_name, company_name, Summary, Initial_Description, SR_Impact_RecID, SR_Severity_RecID, SR_Board_RecID), Source and date_entered).
        model_name (str): Name of the model. If name is the same as an exisiting model, it will overwrite it. If you don't want to store multiple models just keep default value.
        pretrain_epoch (int): Pre-training epoch. Suggested 6, but will increase run-time.
        train_epoch (int): Training epoch. Suggested 12, but will increase run-time.
        max_token_len (int): Maximum length for BERT embedding. Max value is 512 but we found 100 to be a good balance between performance and run-time.
        verbose (int): Control amount of text displayed during training.
    Returns:
        train_results (dataframe): Dataframe with Text, Actual and Predicted labels for Board, Severity and Impact, Prediction Probabilities.
        sev_model (sklearn.svc object): Trained SVM model.
        imp_model (sklearn.svc object): Trained SVM model.
    '''
    warnings.filterwarnings('ignore')
    
    print("***Begin Training***")
    print("***Step 1/5 : Read/Clean Data***")
    data = step_1_read_in(datapath = path_to_data)
    
    print("***Step 2/5 : Train Board***")
    board_predictions = board_train(dataset=data, 
                                    mod_name = model_name+"_board", 
                                    mod_path = "./Saved_Models/Board", 
                                    pt_epoch = pretrain_epoch,
                                    t_epoch = train_epoch,
                                    verbose = verbose)
    
    print("***Step 3/5 : Combine Predictied Board with Text and Source***")
    X_features = berttext_board_source(dataset=data,
                                       predicted_board=list(board_predictions.Board_Predicted),
                                       max_len=max_token_len,
                                       verbose = verbose)
    
    print("***Step 4/5 : Train Severity***")
    sev_model, sev_train = severity_train(X_features,
                                          data.Severity, 
                                          mod_name =  model_name+"_severity", 
                                          mod_path = "./Saved_Models/Severity",
                                          save = save_models,
                                          verbose = verbose)
    
    print("***Step 5/5 : Train Impact***")
    imp_model, imp_train = impact_train(X_features,
                                          data.Impact, 
                                          mod_name =  model_name+"_impact", 
                                          mod_path = "./Saved_Models/Impact",
                                          save = save_models,
                                          verbose = verbose)
    
    print("Models are Saved in `Saved_Models` file.")
    
    train_results = pd.DataFrame({"combined_text":board_predictions.combined_text,
                                 "Board":board_predictions.Board,
                                 "Severity":sev_train.Actual,
                                 "Impact":imp_train.Actual,
                                 "Predicted_Board":board_predictions.Board_Predicted,
                                 "Predicted_Severity":sev_train.Predict,
                                 "Predicted_Impact":imp_train.Predict})
    
    return train_results, sev_model, imp_model
#############################################################################
def train_test_metrics(model_name = "Train_Test_1",
                       train_proportion = 0.8, 
                       data_path = "./Data/Tickets with Classifications.xlsx",
                       file_output = "./Model_History/model_metrics.csv",
                       pt_epoch = 1,
                       t_epoch = 1,
                       max_len = 100,
                       verbose = 2):
    '''
    Train on TRAINING_PROPORTION and report testing metrics. 
    Function to: Intake training data -> Split the data into training/testing -> Clean data -> Model Board -> Model Severity -> Model Impact -> Test models. Models will be stored in Ticket_Triage/Saved_Models/
    ---
    Parameters:
        model_name (str): Name of the model. If name is the same as an exisiting model, it will overwrite it. If you don't want to store multiple models just keep default value. Creates 3 models. 
        training_proportion (float): Training size. Must be between 0 and 1.
        data_path (str/dataframe): Either the path to the datafile or a dataframe. MUST include the following headers (ticketNbr, contact_name, company_name, Summary, Initial_Description, SR_Impact_RecID, SR_Severity_RecID, SR_Board_RecID), Source and date_entered).
        file_output (str): Path to file to store model metrics. If no file is present, will create the file "model_metrics.csv"
        pt_epoch (int): Pre-training epoch. Suggested 6, but will increase run-time.
        t_epoch (int): Training epoch. Suggested 12, but will increase run-time.
        max_len (int): Maximum length for BERT embedding. Max value is 512 but we found 100 to be a good balance between performance and run-time.
        verbose (int): Control amount of text displayed during training.
    Returns:
        none
    '''
    #Custom Scorer
    triage_metric = make_scorer(model_functions.modified_accuracy_score, greater_is_better=True)
    print("**Train-Test Metrics")
    
    ###Step 1: Read in Data and Split by Train/Test
    print("**Step 1: Read-in Data and Train/Test Split**")
    #Check if xlsx, csv or dataframe is input
    if isinstance(data_path,pd.DataFrame):
        data = data_path.copy()
    elif data_path.split(".")[-1] == "xlsx":
        data = pd.read_excel(data_path)
    elif data_path.split(".")[-1] == "csv":
        data = pd.read_csv(data_path)
        
    #Train-Test Split
    train, test = train_test_split(data,
                           shuffle = True,
                           train_size = train_proportion,
                           random_state = 1)
    
    #train.to_csv("./Model_History/Model_Train_Test_Split/"+model_name+".csv",index=False)
    train = train.reset_index().iloc[:,1:]
    test = test.reset_index().iloc[:,1:]
    
    ###Step 2: Train Models on Train Set
    print("**Step 2: Use train_all function**")
    train_results, sev_model, imp_model = train_all(path_to_data = train,
          model_name = model_name,
          pretrain_epoch = pt_epoch,
          train_epoch = t_epoch,
          max_token_len = max_len,
          save_models = "Y",
          verbose = 1)    
        
#     train_results, sev_model, imp_model = train_all(path_to_data = "./Model_History/Model_Train_Test_Split/"+model_name+".csv",
#           model_name = model_name,
#           pretrain_epoch = pt_epoch,
#           train_epoch = t_epoch,
#           max_token_len = max_len,
#           save_models = "Y",
#           verbose = 1)
    
    ###Step 3: Predict on Testing
    #Clean the test text 
    test = test.reset_index()
    test_data = ticket_cleaner.clean_tickets(ticketNbr = test.ticketNbr, 
                                    contact_name = test.contact_name, 
                                    company_name = test.company_name, 
                                    Summary = test.Summary, 
                                    Initial_Description = test.Initial_Description, 
                                    Impact = test.SR_Impact_RecID, 
                                    Severity = test.SR_Severity_RecID, 
                                    Board = test.SR_Board_RecID, 
                                    Source = test.Source, 
                                    date_entered = test.date_entered)
    
    print("**Step 3: Predict on Testing")
    print("**Step 3.1: Predict Board on Testing")
    #Functions alter the dataset you put in so need to make copies 
    board_test_input = test_data.copy()
    #Get Board Predictions
    predictions_board = board.predict_roberta(df = board_test_input,path="./Saved_Models/Board" + "/" + model_name + "_board.pkl")
    #Remap Predictions to Normal Labels
    predictions_board['Board'] = predictions_board.Board.map({0:36, 1:41, 2:43})  
    predictions_board['Board_Predicted'] = predictions_board.Board_Predicted.map({0:36, 1:41, 2:43})    
    
    print("**Step 3.2: BERT Embedded Text and Board and Source for Severity and Impact")

    #Combined Embedded Text and Board and Source
    X_test = berttext_board_source(dataset=test_data,predicted_board = list(predictions_board.Board_Predicted), max_len=max_len, verbose = 1)
    
    print("**3.3: Predict Severity on Testing")
    Y_severity_test = test_data.Severity
    predictions_severity = sev_model.predict(X_test)
    
    print("**3.4: Predict Impact on Testing")
    Y_impact_test = test_data.Impact
    predictions_impact = imp_model.predict(X_test)
    
    ###Step 4: Compile Results
    ### Compile and Output Results
    print("**4: Compile Results")
    
    ### Ticket Predictions
    # Train Results
    train_results["Subset"] = "Train"
    # Test Results
    test_results = test_data.loc[:,["combined_text","Impact","Severity","Board"]].assign(
        Predicted_Board = predictions_board.Board_Predicted, 
        Predicted_Severity = predictions_severity, 
        Predicted_Impact = predictions_impact,
        )
    test_results["Subset"] = "Test"
    # Both Results
    output_df = train_results.append(test_results)
    
    ##Train Metrics
    Y_train_board = train_results.Predicted_Board
    board_train = train_results.Board
    Y_train_severity = train_results.Predicted_Severity
    severity_train = train_results.Severity
    Y_train_impact = train_results.Predicted_Impact
    impact_train = train_results.Impact
    
    #Train Board Custom Acc. and F1-Weighted
    train_board_accuracy = accuracy_score(Y_train_board, board_train)
    train_board_f1_score = f1_score(Y_train_board, board_train, average = "weighted")
    #Train Severity Custom Acc. and F1-Weighted
    train_severity_custom_metric = model_functions.modified_accuracy_score(severity_train,Y_train_severity)
    train_severity_f1_weighted = f1_score(Y_train_severity, severity_train, average = "weighted")
    #Train Impact Custom Acc. and F1-Weighted
    train_impact_custom_metric = model_functions.modified_accuracy_score(impact_train,Y_train_impact)
    train_impact_f1_weighted = f1_score(Y_train_impact, impact_train, average = "weighted")
    
    ##Test Metrics
    predictions_board['Board']
    predictions_board['Board_Predicted']
    Y_severity_test
    Y_impact_test
    #Test Board Custom Acc. and F1-Weighted
    test_board_accuracy = accuracy_score(predictions_board['Board_Predicted'], predictions_board['Board'])
    test_board_f1_score = f1_score(predictions_board['Board_Predicted'], predictions_board['Board'], average = "weighted")
    #Test Severity Custom Acc. and F1-Weighted
    test_severity_custom_metric = np.mean(cross_val_score(sev_model, X_test, Y_severity_test, scoring=triage_metric, cv=5))
    test_severity_f1_weighted = np.mean(cross_val_score(sev_model, X_test, Y_severity_test, scoring="f1_weighted", cv=5))
    #Test Impact Custom Acc. and F1-Weighted
    test_impact_custom_metric = np.mean(cross_val_score(imp_model, X_test, Y_impact_test, scoring=triage_metric, cv=5))
    test_impact_f1_weighted = np.mean(cross_val_score(imp_model, X_test, Y_impact_test, scoring="f1_weighted", cv=5))
    
    #Compile Metrics into DF
    metrics = [train_board_accuracy,
              train_board_f1_score,
              test_board_accuracy,
              test_board_f1_score,
              train_severity_custom_metric,
              train_severity_f1_weighted,
              test_severity_custom_metric,
              test_severity_f1_weighted,
              train_impact_custom_metric,
              train_impact_f1_weighted,
              test_impact_custom_metric,
              test_impact_f1_weighted]
    
    #Date Train-Test was ran
    run_date = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    #Dataset Name
    dataset_name = data_path.split("/")[-1]
    #Training/Testing Stats
    train_size = len(train_results)
    train_brd_36 = len([yes for yes in train_results.Board if yes == 36]) 
    train_brd_41 = len([yes for yes in train_results.Board if yes == 41]) 
    train_brd_43 = len([yes for yes in train_results.Board if yes == 43]) 
    train_low_sev = len([yes for yes in train_results.Severity if yes == 0]) 
    train_med_sev = len([yes for yes in train_results.Severity if yes == 1])
    train_high_sev = len([yes for yes in train_results.Severity if yes == 2]) 
    train_low_imp = len([yes for yes in train_results.Impact if yes == 0])
    train_med_imp = len([yes for yes in train_results.Impact if yes == 1])
    train_high_imp = len([yes for yes in train_results.Impact if yes == 2])
    test_size = len(test_results)
    test_brd_36 = len([yes for yes in test_results.Board if yes == 36]) 
    test_brd_41 = len([yes for yes in test_results.Board if yes == 41]) 
    test_brd_43 = len([yes for yes in test_results.Board if yes == 43]) 
    test_low_sev = len([yes for yes in test_results.Severity if yes == 0]) 
    test_med_sev = len([yes for yes in test_results.Severity if yes == 1])
    test_high_sev = len([yes for yes in test_results.Severity if yes == 2]) 
    test_low_imp = len([yes for yes in test_results.Impact if yes == 0])
    test_med_imp = len([yes for yes in test_results.Impact if yes == 1])
    test_high_imp = len([yes for yes in test_results.Impact if yes == 2])
    
    output_metrics = pd.DataFrame(columns = ["model_name",
                                             "data_set",
                                             "date_entered",
                                             "train_board_accuracy",
                                             "train_board_f1_score",
                                             "test_board_accuracy",
                                             "test_board_f1_score",
                                             "train_severity_custom_metric",
                                             "train_severity_f1_weighted",
                                             "test_severity_custom_metric",
                                             "test_severity_f1_weighted",
                                             "train_impact_custom_metric",
                                             "train_impact_f1_weighted",
                                             "test_impact_custom_metric",
                                             "test_impact_f1_weighted",
                                             "train_size",
                                             "train_brd_36",
                                             "train_brd_41",
                                             "train_brd_43",
                                             "train_low_sev",
                                             "train_med_sev",
                                             "train_high_sev",
                                             "train_low_imp",
                                             "train_med_imp",
                                             "train_high_imp",
                                             "test_size",
                                             "test_brd_36",
                                             "test_brd_41",
                                             "test_brd_43",
                                             "test_low_sev",
                                             "test_med_sev",
                                             "test_high_sev",
                                             "test_low_imp",
                                             "test_med_imp",
                                             "test_high_imp"])
    output_metrics.loc[0] = [model_name,
                            dataset_name,
                            run_date,
                            train_board_accuracy,
                            train_board_f1_score,
                            test_board_accuracy,
                            test_board_f1_score,
                            train_severity_custom_metric,
                            train_severity_f1_weighted,
                            test_severity_custom_metric,
                            test_severity_f1_weighted,
                            train_impact_custom_metric,
                            train_impact_f1_weighted,
                            test_impact_custom_metric,
                            test_impact_f1_weighted,
                            train_size,
                            train_brd_36,
                            train_brd_41,
                            train_brd_43,
                            train_low_sev,
                            train_med_sev,
                            train_high_sev,
                            train_low_imp,
                            train_med_imp,
                            train_high_imp,
                            test_size,
                            test_brd_36,
                            test_brd_41,
                            test_brd_43,
                            test_low_sev,
                            test_med_sev,
                            test_high_sev,
                            test_low_imp,
                            test_med_imp,
                            test_high_imp]
    
    #Check if CSV Exists - If not create one with initialized Column
    if not(os.path.isfile(file_output)):
        print("Creating model_metrics.csv")
        output_metrics.to_csv(file_output, index=False)
    elif os.path.isfile(file_output):
        print("Appending model_metrics.csv to",file_output)
        old_metrics = pd.read_csv(file_output)
        old_metrics = old_metrics.append(output_metrics)
        old_metrics.to_csv(file_output, index=False)
        
     #Output Predictions to Model_History/Model_Predictions folder
    output_df.to_csv("./Model_History/Model_Predictions/"+model_name+"_train_test_predictions.csv",index=False)
#############################################################################
def predict_all(path_to_data = "./Data/Tickets with Classifications.xlsx",
                model_to_use = "2k_dataset",
                max_token_len = 100,
                verbose = 1):
    '''
    Predict on DATASET 
    Function to: Intake data -> Clean data -> Predict using pre-trained models stoed in Ticket_Triage/Saved_Models
    ---
    Parameters:
        path_to_data (str/dataframe): Either the path to the datafile or a dataframe. MUST include the following headers (contact_name, company_name, Summary, Initial_Description and Source).
        model_to_use (str): Name of the models. Will look for model_to_use_board.pkl for Board, model_to_use_severity.joblib for Severity and model_to_use_impact.joblib for Impact.
        file_output (str): Path to file to store model metrics. If no file is present, will create the file "model_metrics.csv"
        max_len (int): Maximum length for BERT embedding. Max value is 512 but we found 100 to be a good balance between performance and run-time.
        verbose (int): Control amount of text displayed during training.
    Returns:
        prediction_df (dataframe): Dataframe with predicted labels and prediction probabilities. 
    '''
    #Check if xlsx or csv
    if isinstance(path_to_data,pd.DataFrame):
        data = path_to_data.copy()
    elif path_to_data.split(".")[-1] == "xlsx":
        data = pd.read_excel(path_to_data)
    elif path_to_data.split(".")[-1] == "csv":
        data = pd.read_csv(path_to_data)
    
    #For the cleaner to work we need values for 'ticketNbr','Impact','Severity','Board','date_entered'
    #This will check if they exist - if not: fill it with proxy values
    #RoBERTa is very pick with its preprocessing function: Therefore pass 1 for Sev and Imp. and 36 for Board
    if "ticketNbr" not in data.columns:
        data["ticketNbr"] = 0
    if "SR_Board_RecID" not in data.columns:
        data["SR_Board_RecID"] = 36
    if "SR_Severity_RecID" not in data.columns:
        data["SR_Severity_RecID"] = 1
    if "SR_Impact_RecID" not in data.columns:
        data["SR_Impact_RecID"] = 1    
    if "date_entered" not in data.columns:
        data["date_entered"] = 0   
        
    print('--Cleaning Text-')
    #Keep TicketNbrs
    ticket_numbers = data.ticketNbr
    
    #Clean text
    data = ticket_cleaner.clean_tickets(ticketNbr = data.ticketNbr, 
                                        contact_name = data.contact_name, 
                                        company_name = data.company_name, 
                                        Summary = data.Summary, 
                                        Initial_Description = data.Initial_Description, 
                                        Impact = data.SR_Impact_RecID, 
                                        Severity = data.SR_Severity_RecID, 
                                        Board = data.SR_Board_RecID, 
                                        Source = data.Source, 
                                        date_entered = data.date_entered)

    ###Board Predictions
    #Copy Data for Board Predictions
    board_data_copy = data.copy()

    #Predict Board
    print('--Predicting Board--')
    predictions_board = board.predict_roberta(df = board_data_copy,
                                             path = "./Saved_Models/Board/"+model_to_use+"_board.pkl")
    #Store Predictions and Probability
    predictions_df = pd.DataFrame({"ticketNbr":ticket_numbers,
                                   "Board_Predicted":predictions_board.Board_Predicted.map({0:36, 1:41, 2:43}),
                                  "Board_Probability":predictions_board.Board_Probability})

    ###Combined Predictions
    print('--BERT Embedding text and combining with Board and Source--')
    X_features = berttext_board_source(dataset=data,
                                       predicted_board=list(predictions_df.Board_Predicted),
                                       max_len=max_token_len,
                                       verbose = verbose)

    ###Severity Predictions
    #Predict Severity 
    print('--Predicting Severity--')
    predictions_severity = severity.predict_svm(X_features, 
                                            import_path = "./Saved_Models/Severity/"+model_to_use+"_severity.joblib",
                                            verbose = verbose)
    #Store Predictions and Probability
    predictions_df = predictions_df.assign(Severity_Predicted = predictions_severity.Predict,
                         Severity_Probability = [max(i) for i in predictions_severity.iloc[:,0:3].values])

    ###Impact Predictions
    #Predict Impact
    print('--Predicting Impact--')
    predictions_impact = impact.predict_svm(X_features, 
                                            import_path = "./Saved_Models/Impact/"+model_to_use+"_impact.joblib",
                                            verbose = verbose)
    #Store Predictions and Probability
    predictions_df = predictions_df.assign(Impact_Predicted = predictions_impact.Predict,
                         Impact_Probability = [max(i) for i in predictions_impact.iloc[:,0:3].values])

    return predictions_df