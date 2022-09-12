from flask import Flask, request, render_template
from artifacts.utils import car_price

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():
    data = request.form
    car_price_obj = car_price(data)
    result = car_price_obj.predict()
    return render_template('index.html', pred = result)

if __name__ == "__main__":
    app.run(debug = True)