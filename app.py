from flask import Flask, request
import pandas as pd
import src.main as main
import logging
import warnings 

warnings.filterwarnings('ignore')

logger = logging.getLogger()
logger.setLevel(logging.INFO)

app = Flask(__name__)

@app.route('/')
def predict_container():

    input_docker = pd.DataFrame({"contact_name":["Sam Velez"],
                          "company_name":["MNP"],
                          "Summary":["No internet connection"],
                          "Initial_Description":["No one has internet connection in the office"],
                          "Source":["Email Connector"]})

    test_predictions = main.predict_all(path_to_data = input_docker,
                model_to_use = "2k_dataset",
                max_token_len = 100,
                verbose = 1)
    
    return test_predictions.to_html(classes='data')

if __name__ == '__main__':
    app.run(host='0.0.0.0')