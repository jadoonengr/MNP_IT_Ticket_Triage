### General Packages ###
import pandas as pd
import datetime

### For Model Exporting ###
from joblib import dump, load

### Metrics for Evaluation ###
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, accuracy_score, f1_score

### For Board Modelling ###
from sklearn.svm import SVC

### Ticket triage functions ###
import sys
sys.path.append("src/Auxiliary/")
sys.path.append("src/Cleaning/")
sys.path.append("src/Model/")
sys.path.append("src/Tokenizer/")

### For pre-processing ###
import ticket_cleaner
import bert_tokenizer

### For Board Modelling ###
import board

### For Severity Modelling ###
import severity

### For Impact Modelling ###
import impact

### For Prs, Modified Accuracy Score ###
import model_functions
import numpy as np

### For SVM Board ###
from bert_tokenizer import BERT_Tokenizer
from transformers import DistilBertModel, DistilBertTokenizer

### To Write out results ###
import csv

def train_test(run_name = "Train_Test",train_proportion = 0.8, file_address = "./Data/Tickets with Classifications.xlsx",file_output = "./Model_History/dashboard_triage_metrics.csv"):

    #Custom Scorer
    triage_metric = make_scorer(model_functions.modified_accuracy_score, greater_is_better=True)
    
    #Read in Training Data
    train_set = pd.read_excel(file_address)
    data = ticket_cleaner.clean_tickets(ticketNbr = train_set.ticketNbr, contact_name = train_set.contact_name, company_name = train_set.company_name, Summary = train_set.Summary, Initial_Description = train_set.Initial_Description, Impact = train_set.SR_Impact_RecID, Severity = train_set.SR_Severity_RecID, Board = train_set.SR_Board_RecID, Source = train_set.Source, date_entered = train_set.date_entered)
####################################################################################################################################################
    #BERT Pre-loading
    print("--Pre-loading BERT Tokenizers (1/7)--")
    max_length = 100
    model_class, tokenizer_class, pretrained_weights = (DistilBertModel, DistilBertTokenizer, 'distilbert-base-uncased')
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    model = model_class.from_pretrained(pretrained_weights)
    
    #Train-Test Split
    print("--Train-Test Split (2/7)--")
    train, test = train_test_split(data,
                           shuffle = True,
                           train_size = train_proportion,
                           random_state = 1)

    #BERT Embeddings
    print("--Embedding Text with BERT. Will take a few minutes. (3/7)--")
    X_text_train = BERT_Tokenizer(model = model, tokenizer = tokenizer, text = train.combined_text, max_len = max_length)
    X_text_test = BERT_Tokenizer(model = model, tokenizer = tokenizer, text = test.combined_text, max_len = max_length)
####################################################################################################################################################
    ### SVM Board Predicitons ###
    print("--Board Modelling (4/7)--")
    #Extract Acutal Board from Train/Test set
    Y_train_board = train.Board
    Y_test_board = test.Board
    
    #Model Board
    board_svm = SVC(C=1, kernel='linear', class_weight='balanced', gamma=0.1)
    board_svm.fit(X_text_train,Y_train_board)
    
    #Predict Board
    board_train = board_svm.predict(X_text_train)
    board_test = board_svm.predict(X_text_test)
####################################################################################################################################################          
    ### Attach Board and Source to Text ###
    #Needed for Severity and Impact Modelling
    #Creating Empty Columns for brd36,brd41,brd43
    X_text_train = pd.DataFrame(X_text_train).assign(brd36 = [0]*len(X_text_train),brd41 = [0]*len(X_text_train),brd43 = [0]*len(X_text_train))
    X_text_test = pd.DataFrame(X_text_test).assign(brd36 = [0]*len(X_text_test),brd41 = [0]*len(X_text_test),brd43 = [0]*len(X_text_test))

    #Add predictions from Board Model to Text Embeddings
    X_text_src_board_train = severity.add_board_predictions(X_text_train, board_predict=board_train)
    X_text_src_board_test = severity.add_board_predictions(X_text_test, board_predict=board_test)
    
    X_text_src_board_train = X_text_src_board_train.set_index(train.index)
    X_text_src_board_test = X_text_src_board_test.set_index(test.index)
    
    #Add Source Text Embeddings
    source_board = ["email_connector","deskdirector","email","renewal","escalation"]
    for i in source_board:
            X_text_src_board_test[i] = test[i]
            X_text_src_board_train[i] = train[i]
####################################################################################################################################################     
    ### Severity ###
    print("--Severity Modelling (5/7)--")
          
    #Extract Acutal Severity from Train/Test set
    Y_severity_train = train.Severity
    Y_severity_test = test.Severity
          
    #Model Severity 
    #Don't save the model - use return object
    severity_svm_model, severity_train = severity.train_svm(X_text_src_board_train,Y_severity_train, model_name = "sev_model1", save_model = "N", export_path = "./Saved_Models/Severity", verbose=2)
    
    #Predict Severity
    predictions_severity = severity_svm_model.predict(X_text_src_board_test)
####################################################################################################################################################      
    ### Impact ###
    print("--Impact Modelling (6/7)--")
    #Extract Acutal Impact from Train/Test set
    Y_impact_train = train.Impact
    Y_impact_test = test.Impact

    #Model Impact 
    #Don't save the model - use return object
    impact_svm_model, impact_train = impact.train_svm(X_text_src_board_train,Y_impact_train, model_name = "imp_model1", save_model = "N", export_path = "./Saved_Models/Impact", verbose=2)

    #Predict Impact
    predictions_impact = impact_svm_model.predict(X_text_src_board_test)
####################################################################################################################################################
    ### Compile and Output Results ###
    print("--Compile and Output results to Model_History (7/7)--")
    # Train Results
    train = train.loc[:,["Impact","Severity","Board"]].assign(Board_Predictions = board_train, Severity_Predictions = severity_train.Predict, Impact_Predictions = impact_train.Predict)
    train["Subset"] = "Train"
    # Test Results
    test = test.loc[:,["Impact","Severity","Board"]].assign(Board_Predictions = board_test, Severity_Predictions = predictions_severity, Impact_Predictions = predictions_impact)
    test["Subset"] = "Test"
    # Both Results
    output_df = train.append(test)

     ##Train Metrics
    #Train Board Custom Acc. and F1-Weighted
    train_board_accuracy = accuracy_score(Y_train_board, board_train)
    train_board_f1_score = f1_score(Y_train_board, board_train, average = "weighted")
    #Train Severity Custom Acc. and F1-Weighted
    train_severity_custom_metric = np.mean(cross_val_score(severity_svm_model, X_text_src_board_train, Y_severity_train, scoring=triage_metric, cv=5))
    train_severity_f1_weighted = np.mean(cross_val_score(severity_svm_model, X_text_src_board_train, Y_severity_train, scoring="f1_weighted", cv=5))
    #Train Impact Custom Acc. and F1-Weighted
    train_impact_custom_metric = np.mean(cross_val_score(impact_svm_model, X_text_src_board_train, Y_impact_train, scoring=triage_metric, cv=5))
    train_impact_f1_weighted = np.mean(cross_val_score(impact_svm_model, X_text_src_board_train, Y_impact_train, scoring="f1_weighted", cv=5))
    
    ##Test Metrics
    #Test Board Custom Acc. and F1-Weighted
    test_board_accuracy = accuracy_score(Y_test_board, board_test)
    test_board_f1_score = f1_score(Y_test_board, board_test, average = "weighted")
    #Test Severity Custom Acc. and F1-Weighted
    test_severity_custom_metric = np.mean(cross_val_score(severity_svm_model, X_text_src_board_test, Y_severity_test, scoring=triage_metric, cv=5))
    test_severity_f1_weighted = np.mean(cross_val_score(severity_svm_model, X_text_src_board_test, Y_severity_test, scoring="f1_weighted", cv=5))
    #Test Impact Custom Acc. and F1-Weighted
    test_impact_custom_metric = np.mean(cross_val_score(impact_svm_model, X_text_src_board_test, Y_impact_test, scoring=triage_metric, cv=5))
    test_impact_f1_weighted = np.mean(cross_val_score(impact_svm_model, X_text_src_board_test, Y_impact_test, scoring="f1_weighted", cv=5))

    #Impact proportions - Low vs Medium and High
    train_impact_proportion_low_vs_mediumhigh = Y_impact_train[Y_impact_train == 0].count()/Y_impact_train.count()
    test_impact_proportion_low_vs_mediumhigh = Y_impact_test[Y_impact_test == 0].count()/Y_impact_test.count()
    test_severity_proportion_low_vs_mediumhigh = Y_severity_test[Y_severity_test == 0].count()/Y_severity_test.count()
    train_severity_proportion_low_vs_mediumhigh = Y_severity_train[Y_severity_train == 0].count()/Y_severity_train.count()


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
              test_impact_f1_weighted,
              train_impact_proportion_low_vs_mediumhigh,
              test_impact_proportion_low_vs_mediumhigh,
              test_severity_proportion_low_vs_mediumhigh,
              train_severity_proportion_low_vs_mediumhigh]
    #Date Train-Test was ran
    run_date = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    
    output_metrics = pd.DataFrame(columns = ["model_name",
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
                                             "train_impact_proportion_low_vs_mediumhigh",
                                             "test_impact_proportion_low_vs_mediumhigh",
                                             "test_severity_proportion_low_vs_mediumhigh",
                                             "train_severity_proportion_low_vs_mediumhigh"
                                             ])
    output_metrics.loc[0] = [run_name,
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
                            train_impact_proportion_low_vs_mediumhigh,
                            test_impact_proportion_low_vs_mediumhigh,
                            test_severity_proportion_low_vs_mediumhigh,
                            train_severity_proportion_low_vs_mediumhigh]
    
    #Append results to CSV or Create CSV if it does not exit
    with open(file_output, "a") as file:
        csv_writter  = csv.writer(file)
        csv_writter.writerow(list(output_metrics.loc[0]))
        
    #Output Predictions to Model_History/Model_Predictions folder
    output_df.to_csv("./Model_History/"+run_name+"_sample_output.csv")