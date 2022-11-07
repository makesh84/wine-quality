import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods = ['POST'])
def predict():
    
    features = [float(x) for x in request.form.values()]
    final_features = np.array(features).reshape(1,-1)
    prediction = model.predict(final_features)
    return render_template('index.html',prediction_text = "output is {}".format(prediction))

if __name__ =="__main__":
    app.run(debug=True)