import numpy as np
import pickle
from flask import Flask , request , jsonify , render_template

app = Flask(__name__)

with open('rfr.pkl' , 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('page.html')

@app.route('/predict', methods=['POST'])
def predict():
    medinc = request.form['medinc']
    houseage = request.form['houseage']
    avgrooms = request.form['avgrooms']
    avgbdrms = request.form['avgbdrms']
    pop = request.form['population']
    avgocp = request.form['avgoccp']
    lat = request.form['latitude']
    long = request.form['longitude']

    arr = np.array([medinc, houseage, avgrooms, avgbdrms , pop , avgocp , lat , long])
    arr = arr.astype(np.float64)
    pred = model.predict([arr])

    return render_template('page.html', data=int(pred))


if __name__ == '__main__':
    app.run(debug=True)
