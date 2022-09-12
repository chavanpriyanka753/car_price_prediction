import pickle
import numpy as np

class car_price():
    def __init__(self, data):
        self.data = data

    def load_model(self):
        with open(r'artifacts/model.pkl', 'rb') as file:
            self.model = pickle.load(file)

    def predict(self):
        self.load_model()

        Year = float(self.data['Year'])
        Present_Price = float(self.data['Present_Price'])
        Kms_Driven = float(self.data['Kms_Driven'])
        Fuel_Type = float(self.data['Fuel_Type'])
        Seller_Type = float(self.data['Seller_Type'])
        Transmission = float(self.data['Transmission'])
        Owner = float(self.data['Owner'])

        array = np.array([Year, Present_Price, Kms_Driven, Fuel_Type, Seller_Type, Transmission, Owner], ndmin = 2)
        result = np.around(self.model.predict(array), 2)[0]
        return result



   
