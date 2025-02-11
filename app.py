from flask import Flask, request,make_response
import joblib
import pandas as pd
import numpy as np
import os 
from flask_cors import CORS
# from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

CORS(app)



model = joblib.load('models/crop_recommendation_model.pkl')

model = joblib.load('models/crop_recommendation_model.pkl')
district_encoder = joblib.load('models/District_encoder.pkl')
village_encoder = joblib.load('models/Village_encoder.pkl')
soil_quality_encoder = joblib.load('models/Soil_Quality_encoder.pkl')


def value_encoder(district, village, soil_quality):
  return [district_encoder.transform([district])[0], village_encoder.transform([village])[0],soil_quality_encoder.transform([soil_quality])[0]]

@app.route('/', methods=["GET"])
def home():
  return make_response("Backend for Crop Recommendation System",200)


@app.route('/predict', methods=['POST'])
def predict():
  try:

    data = request.get_json()
    
    if(not data):
      return make_response({
        "Error" : {
          "code": 400,
          "message": "Bad request"
        }
      }, 400)
    encodedValues = value_encoder(data["district"], data['village'], data['soil_quality'])
    feature_array = np.array([[data['ph'], data['n'], data['p'], data['k'], encodedValues[0], encodedValues[1], encodedValues[2]]])
    featured_array_df = pd.DataFrame(feature_array, columns=["pH", "N", "P", "K", "District", "Village", "Soil_Quality"])
    prediction = model.predict(featured_array_df)
    
    
    return make_response({
      "Success": {
      "code": 200,
      "prediction": prediction[0]
    }}, 200)
  except Exception as e:
      return make_response({"Error": {"code": 500, "message": str(e)}}, 500)

if __name__ == '__main__':
    app.run(debug=True)
