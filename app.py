import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import statsmodels.api as sm
from sklearn import linear_model
from pycaret.classification import predict_model

app = Flask(__name__)
model = pickle.load(open('trained_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]

    pred = predict_model(model, pd.DataFrame(data=[final_features], 
                                             columns=pd.Index(['male', 'age', 'education', 'currentSmoker', 'cigsPerDay', 'BPMeds',
                                                               'prevalentStroke', 'prevalentHyp', 'diabetes', 'totChol', 'sysBP',
                                                               'diaBP', 'BMI', 'heartRate', 'glucose'],
                                                                dtype='object')))
    prediction = pred["Score"][0] if pred["Label"][0] == "1.0" else 1 - pred["Score"][0]

    return render_template("index.html", prediction_text=f'Khả năng mắc bệnh Tim mạch vành trong 10 năm tới là {prediction[0][0] * 100:.2f}%')

if __name__ == "__main__":
    app.run(debug=True)
