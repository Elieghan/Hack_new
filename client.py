import pandas as pd
import numpy as np
import requests
import pickle

filename = 'churn_model.pkl'
model = pickle.load(open(filename, 'rb'))

# https://itc-heroku-ex-2021.herokuapp.com/

def main():
    """
    Send requests to the inference server for predicting some samples with our model.
    Author:  Elie Ghanassia
    """
    NB_SAMPLE = 5
    X_test = pd.read_csv("X_test.csv")
    y_pred = np.loadtxt('preds.csv')
    sample = X_test.sample(NB_SAMPLE)
    index = sample.index.tolist()
    list_params = X_test.columns.tolist()
    pred_sample = y_pred[index].tolist()
    res_sample = []

    for i in range(NB_SAMPLE):
        params = dic = dict(zip(list_params,sample.iloc[i].tolist()))
        x = requests.get('https://itc-heroku-ex-2021.herokuapp.com/predict_churn',params=params)
        print(f'Predictions for sample {params}:')
        print(x.text)
        print('---------------------')
        res_sample.append(int(x.text))

    if np.array_equal(pred_sample,res_sample):
        print('Predictions are exactly the same as before')
    else:
        print('Predictions are not the same as before')


if __name__ == '__main__':
    main()