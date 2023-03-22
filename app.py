from __future__ import division, print_function

from flask import Flask, render_template, request
import joblib
import numpy as np

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)

model = joblib.load('model.sav')


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict-traffic')
def predict_traffic():
    return render_template('predict.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':

        val0, val1, val2, val3, val4, val5, val6, val7 = str(request.form['0']), float(request.form['1']), float(request.form['2']), float(
            request.form['3']), float(request.form['4']), float(request.form['5']), float(request.form['6']), float(request.form['7'])
        final4 = np.array([val1, val2, val3, val4, val5,
                          val6, val7]).reshape(1, -1)
        #int_features= [float(x) for x in request.form.values()]
        # final4=[np.array(int_features)]
        predict = model.predict(final4)

        if predict == 0:
            output = 'Based on the Speed and Vehicle Count, The Network Traffic is Congested!'
        elif predict == 1:
            output = 'Based on the Speed and Vehicle Count, The Network Traffic is Free-Flow!'
        elif predict == 2:
            output = 'Based on the Speed and Vehicle Count, The Network Traffic is Impossible!'
        elif predict == 3:
            output = 'Based on the Speed and Vehicle Count, The Network Traffic is Unknown'

    return render_template("index.html", output=output)


@app.route("/notebook")
def notebook():
    return render_template("Notebook.html")


if __name__ == "__main__":
    app.run(debug=True)
