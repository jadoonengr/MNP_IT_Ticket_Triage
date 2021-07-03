# Table of contents
1. Project Description
2. Package Structure
3. Main Functions
4. How To Use (TLDR)
5. Packages Used
6. Docker Implementation
<br />

# 1. Project Description 

MNP ITMS receives 100s of tickets per day and they must be manually classified by technicians and engineers to prioritize and assign to a team.

Tickets with a higher severity and impact are assigned to be solved faster than lower impacts and lower severity issues. Misclassification in these two categories leads to higher waiting times for more important issues, significantly lower client satisfaction. In classifying boards, tickets are sent to different teams for processing depending on the responsible area. A misclassification in this category provokes further escalations and changes of assignment, increasing waiting times.

To accomplish this, the package contains different functions that perform the necessary steps preprocessing, modelling and analysis steps:
- Preprocess the data into a clean state that eliminates irrelevant information such as email signatures, DeskDirector questions and mentions of customer names.
- Output metrics on alternate models to test their performance using subsets of the data.
- Model the data using the full available dataset and export the result into a file.
- Apply the model files to additional datasets.  
<br />

# 2. Package Structure
Shown below is the general file structure of the package and a breakdown of its contents.
- **Data**: Contains the data files used for training.
- **Model History**: Contains the outputs from the 'train_test_metrics' function.
    - [`model_metrics.csv`](Model_History/model_metrics.csv) : contains the metrics by model trained.
    - [*Model_Predictions*](Model_History/Model_Predictions) : CSV files by model trained with individual observation predictions.
- **Saved_Models**: Contains .pkl and .joblib files for saved models stored in their respective folders.
- **src**: Contains code stored in .py files.
    - `main.py` : 
        -  `train_all()` function to train and store Board, Severity and Impact models.  
        - `train_test_metrics()` function to split data into train/test and store Board, Severity and Impact models plus return metrics on the test set.  
        - `predict_all()` function to predict labels for data given pre-trained models.
    - [*auxiliary*](src/auxiliary) : Contains .py files for extracting probabilities from predictions, calculating metrics. 
    - [*cleaning*](src/cleaning) : Contains .py file for pre-processing text.
    - [*model*](src/model) : Contains .py files for formatting inputs, training and predicting with the Board, Severity and Impact models.
    - [*tokenizer*](src/tokenizer) : Contains .py file for BERT embedding text.  
<br />

# 3. Main Functions
Each of the 3 models (Board, Severity and Impact) contain individual functions for formatting inputs unique to the model, training the model and using pre-trained models for prediction. To simplify training and using models for prediction, we packaged all these functions into 3 main functions outlined below.
- `train_test_metrics()`: This function will split the data found at your ***data path*** (typically in the [Data](Data) folder and pre-process the text. The training portion will then be used for training and model will be save in the [Saved_Models](Saved_Models) folder under `model_name_board.pkl` or `model_name_severity/impact.joblib`. Trained model will then be evaluated using the testing set, metrics will be stored in the file specified in ***file_output*** (typically model_metrics.csv). A more detailed breakdown with the Text, Actual Label and Predicted Label for each observation can be found in the [Model_Predictions](Model_History/Model_Predictions) folder.   
### **If no model_metrics.csv file is present - one will be created. If one already exists, results will be appeneded to existing file.** ###
```python
train_test_metrics(model_name = "2k_dataset",
                    train_proportion = 0.8, 
                    data_path = "./Data/Tickets with Classifications.xlsx",
                    file_output = "./Model_History/model_metrics.csv",
                    pt_epoch = 6,
                    t_epoch = 12,
                    max_len = 100)
```
- `train_all()`: Similar to the `train_test_metric()` function but will use the **WHOLE** dataset for training, therefore there are no testing metrics to output. Data will follow the same pipeline: Preprocessing - Model Fitting - Save models to [Saved_Models](Saved_Models).folder.
```python
train_all(path_to_data = "./Data/Tickets with Classifications.xlsx",
          model_name = "2k_dataset",
          pretrain_epoch = 1,
          train_epoch = 1,
          max_token_len = 5,
          save_models = "Y",
          verbose = 1)
```
### **If you enter a model_name that already exists, it will overwrite/replace that model. Therefore if you want to keep a history of models - change model_name.**

- `predict_all()`: Read in data located at path_to_data, clean the data and used pre-trained models for predictions. Will return a dataframe with: TicketNbr, Predicted labels and their probabilities. 
```python
predict_all(path_to_data = "./Data/Tickets with Classifications.xlsx",
                model_to_use = "2k_dataset",
                max_token_len = 100,
                verbose = 1)
```
<br />

# 4. How to use (TLDR)
**example.ipynb** demonstrates how to use the 3 main functions. But in general, follow the steps below:   

### To Train:
1. Acquire Training set from Ticket Database as a CSV, XLSX or pd.DataFrame.

If you want to train using the **WHOLE** training set:  

2. Use ***train_test_metrics()*** to document your model's performance on the training set
3. Use ***train_all()*** with the same model name to train on the whole set.

If you want you are ok with training with a Train/Test subset:  

2. Use ***train_test_metrics()*** to document your model's performance on the training set

### To Predict:
1. Acquire Prediction set from Ticket Database as a CSV, XLSX or pd.DataFrame.
2. Use ***predict_all()*** specifying the name of the pre-trained models to use.
Returns a dataframe with predicted labels and prediction probabilities.  
<br />  

# 5. Packages Used
Below is a list of Python packages used and their function. 

Library |Purpose|
--- | --- |
numpy, pandas, counter | Data wrangling/manipulation |
re, nltk | Text pre-processing | 
datetime | Estimate run-times |
pytorch | RoBERTa modelling and BERT embeddings | 
transformers | Pre-trained BERT and RoBERTa | 
sklearn | Modelling, Metrics
joblib | Model export/import | 
flask, redis | Docker deployment
<br />

# 6. Docker Implementation
The structure for the Docker implementation is in the main folder:
requirements.txt
Dockerfile
.dockerignore

First, set Ticket_Triage as your current working directory. In terminal: `cd path/Ticket_Triage`
```console
foo@bar ~ $ cd Ticket_Triage
```
With these files, a Docker image for the app.py file can be built using: `docker build -t ticket-triage .`
```console
foo@bar ~ Ticket_Triage $ docker build -t ticket-triage .
```
And the session can be started with: `docker run -p 5000:5000 ticket-triage`
```console
foo@bar ~ Ticket_Triage $ docker run -p 5000:5000 ticket-triage
```

You can then enter `http://localhost:5000/` into your web-browser to see the results from the `app.py` file.  
Changes to the Dockerfile may be necessary to make it work in the desired port (default is 5000).  
`app.py` contains a placeholder file with the tickets that were used in prediction. Line 15 has to be adjusted in case a new file is to be used.  
We followed/found the following resource helpful for Docker Deployment: https://docs.docker.com/language/python/run-containers/
