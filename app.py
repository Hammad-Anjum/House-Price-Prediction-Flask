import numpy as np
import joblib
from flask import Flask , request , render_template

app = Flask(__name__)

model = joblib.load('model.pkl')

@app.route('/')
def home():
    return render_template('page.html')

@app.route('/predict', methods=['POST'])
def predict():
    medinc = request.form['medinc']
    houseage = request.form['houseage']
    avgrooms = request.form['avgrooms']
    avgbdrms = request.form['avgbdrms']
    pop = request.form['pop']
    avgocp = request.form['avgoccp']
    lat = request.form['lat']
    long = request.form['long']

    arr = np.array([medinc, houseage, avgrooms, avgbdrms , pop , avgocp , lat , long])
    arr = arr.astype(np.float64)
    pred = model.predict([arr])
    pred = np.round(pred * 100000 , 2) 

    return render_template('page.html', data=float(pred))


if __name__ == '__main__':
    app.run(debug=True)