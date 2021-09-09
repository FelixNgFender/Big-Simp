from flask import Flask, request, jsonify, render_template
import pickle
import sklearn

app = Flask(__name__)
model = pickle.load(open('trained_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    int_features = [int(x) for x in request.form.values()]
    normed_features = sklearn.preprocessing.normalize([int_features], axis=1)
    prediction = model.predict_proba(normed_features)

    return render_template("index.html", prediction_text=f'Khả năng mắc bệnh Tim mạch vành trong 10 năm tới là {prediction[0][0] * 100:.2f}%')

if __name__ == "__main__":
    app.run(debug=True)
