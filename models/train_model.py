import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os


DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

data_path = os.path.join(DATA_DIR, 'CropData.csv')
print(data_path)

data = pd.read_csv(data_path)
district_encoder = LabelEncoder()
village_encoder = LabelEncoder()
soilQuality_encoder = LabelEncoder()

data['District'] = district_encoder.fit_transform(data['District'])
data['Village'] = village_encoder.fit_transform(data['Village'])
data['Soil_Quality'] = soilQuality_encoder.fit_transform(data['Soil_Quality'])

X = data[["pH", 'N', 'P', 'K', 'District', 'Village', 'Soil_Quality']]
y = data["Crop"]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state = 42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


joblib.dump(model, 'models/crop_recommendation_model.pkl')




# Save encoders to disk
joblib.dump(district_encoder, 'models/District_encoder.pkl')
joblib.dump(village_encoder, 'models/Village_encoder.pkl')
joblib.dump(soilQuality_encoder, 'models/Soil_Quality_encoder.pkl')





