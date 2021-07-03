# Auxiliary functions for evaluating models

###General Imports ###
import pandas as pd
import numpy as np

#Scoring Function
def modified_accuracy_score(y_true,y_predict,close_label_value = 1, penalty_label_value = 0):
    '''
    Modified accuracy metric that considers misclassification of the Low class (0) as Medium (1) and misclassifications of the Medium class (1) as High (2) as correct.
    Penalizes miclassifications of the High class (2) as Medium (1) and Low (0) by a penalty value.
    Returns the proportion of correct - penalized labels. 
    ---
    Paramters:
        y_true (list/array): List/array of true labels of Severity/Impact ranging from 0-2.
        y_predict (list/array): List/array of predicted labels of Severity/Impact ranging from 0-2.
        close_label_value (int): Value given to predictions that are 'correct'. Default value = 1 (i.e. same weight as perfectly predicted).
        penalty_label_value (int): Value given to penalized classifications. Default value = 0 (i.e. no penalty).
    ---
    Returns:
        (float): Proportion of correctly predicted labels.
    '''
    
    correct = 0 

    if str(type(y_true)) == '''<class 'pandas.core.series.Series'>''':
        y_true = y_true.reset_index().iloc[:,1]
    if str(type(y_predict)) == '''<class 'pandas.core.series.Series'>''':
        y_predict = y_predict.reset_index().iloc[:,1]
        
    for i in range(0,len(y_true)):
        if y_true[i] == y_predict[i]: #Labels are the same
            correct += 1
        elif y_true[i] == 0 and y_predict[i] == 1: #Low -> Medium
            correct += close_label_value
        elif y_true[i] == 1 and y_predict[i] == 2: #Medium -> High
            correct += close_label_value
        elif y_true[i] == 2 and y_predict[i] == 1: #High -> Medium 
            correct -= penalty_label_value/2
        elif y_true[i] == 2 and y_predict[i] == 0: #High -> Low 
            correct -= penalty_label_value
  
    return (correct/len(y_true))

#Combine Probabilities for Model
def combine_probs(model_list,threshold=0.5,default_value=1):
    average_probs = []
    df1 = model_list[0]
    df2 = model_list[1]
    df3 = model_list[2]
    
    Y = df1.Actual.values

    for i in range(0,3): #Pr_0, Pr_1 and Pr_2 columns
        intermediate = pd.DataFrame({"df_1":df1.iloc[:,i],"df_2":df2.iloc[:,i],"df_3":df3.iloc[:,i]})
        average_probs.append(list(intermediate.mean(axis=1).values))
    
    average_prob_valid_results = pd.DataFrame({"Sev_0":average_probs[0],"Sev_1":average_probs[1],"Sev_2":average_probs[2],"Actual":Y})
    average_prob_valid_results["max_pr"] = [max(i) for i in average_prob_valid_results.iloc[:,0:3].values]
    average_prob_valid_results = average_prob_valid_results.reset_index().iloc[:,1:]

    predict = []
    for i in range(0,len(average_prob_valid_results)):
        max_pr = list(average_prob_valid_results.iloc[i,0:3]).index(average_prob_valid_results.max_pr[i])
        if average_prob_valid_results.max_pr[i] < threshold:
            predict.append(default_value)
        else:
            predict.append(list(average_prob_valid_results.iloc[i,0:3]).index(average_prob_valid_results.max_pr[i]))

    average_prob_valid_results["Predict"] = predict
    return average_prob_valid_results

def extract_probabilities(model,X,Y=None):
    '''
    Extract the probabilities from a SVM model.
    ---
    Paramters:
        model (sklearn.svm._classes.SVC): Scikit SVM model used for prediction.
        X (dataframe): Features used for modelling.
        Y (array): Predicted labels from model. Max 3 labels.
    ---
    Returns:
        prob_df(dataframe): Dataframe with prediction probabilities of each of the 3 labels, the final predicted label and the actual label. 
    '''    
    #Extracting Votes from SVM
    votes = np.array(model.decision_function(X)) #Votes
    
    #Convert Votes to Probabilities 
    prob = np.exp(votes)/np.sum(np.exp(votes),axis=1, keepdims=True) #Probabilities
    
    #Getting Probabilities (essentially max probability)
    predictions = model.predict(X)
    
    if Y is None: #We are NOT given actual labels
        labels = [0,1,2] #Labels
        prob_df = pd.DataFrame({"Pr_"+str(labels[0]):prob[:,0],"Pr_"+str(labels[1]):prob[:,1],"Pr_"+str(labels[2]):prob[:,2],"Predict":predictions})
        
    else: #We are given actual labels
        labels = sorted(list(Y.unique())) #Labels
        prob_df = pd.DataFrame({"Pr_"+str(labels[0]):prob[:,0],"Pr_"+str(labels[1]):prob[:,1],"Pr_"+str(labels[2]):prob[:,2],"Predict":predictions,"Actual":Y})
        
    return prob_df