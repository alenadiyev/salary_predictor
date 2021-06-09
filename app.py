import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
# model = pickle.load(open('model_econ.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    if list(request.form.values())[0] == "economist":
        model = pickle.load(open('model_econ.pkl', 'rb'))
    elif list(request.form.values())[0] == "data scientist":
        model = pickle.load(open('model_ds.pkl', 'rb'))
    elif list(request.form.values())[0] == "developer":
        model = pickle.load(open('model_dev.pkl', 'rb'))
    elif list(request.form.values())[0] == "junior developer":
        model = pickle.load(open('model_jundev.pkl', 'rb'))
    elif list(request.form.values())[0] == "senior developer":
        model = pickle.load(open('model_sendev.pkl', 'rb'))
    elif list(request.form.values())[0] == "robotics engineer":
        model = pickle.load(open('model_rob.pkl', 'rb'))
    else:
        return render_template('index.html', prediction_text='Enter a valid profession: data scientist, economist, developer, junior developer, senior developer, robotics engineer')
    int_features = list(request.form.values())[1:]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = prediction

    if output[0] < 0:
        output[0] = 0

    return render_template('index.html', prediction_text='Employee Salary should be {} tenge'.format(round(output[0])))

if __name__ == "__main__":
    app.run(debug=True)