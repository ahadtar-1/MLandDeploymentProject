"""
The module comprises of the driver program for the function used to create an API and the function.
"""

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/api', methods = ['GET', 'POST'])
def make_prediction()-> None:
    """
    The api function that generates a prediction from a model

    Parameters
    ----------
    None

    Returns
    -------
    json
        The prediction generated from the model
    
    """

    if request.method == 'POST':
        model = pickle.load(open('model.pkl', 'rb'))
        jsonData = request.json
        queryDf = pd.DataFrame(jsonData)
        prediction = model.predict(queryDf)
        return jsonify({"Prediction": list(prediction)})


if __name__ == '__main__':
    app.run(port = 5000, debug = True)
