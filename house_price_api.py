from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

app = FastAPI()

class HouseFeatures(BaseModel):
    square_feet: float  # GrLivArea
    bedrooms: int      # BedroomAbvGr
    bathrooms: float   # combining FullBath and HalfBath
    year_built: int    # YearBuilt

class HousePricePredictor:
    def __init__(self):
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        self.feature_columns = ['square_feet', 'bedrooms', 'bathrooms', 'year_built']
        
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        return self.scaler.fit_transform(data)
    
    def train(self, X: pd.DataFrame, y: pd.Series):
        X_processed = self.preprocess_data(X)
        self.model.fit(X_processed, y)
        
    def predict(self, features: pd.DataFrame) -> float:
        features_processed = self.scaler.transform(features)
        return self.model.predict(features_processed)[0]

# Load and prepare the data
def prepare_training_data():
    # Load the training data
    df = pd.read_csv('train.csv')
    
    # Prepare features
    X = pd.DataFrame({
        'square_feet': df['GrLivArea'],
        'bedrooms': df['BedroomAbvGr'],
        'bathrooms': df['FullBath'] + 0.5 * df.get('HalfBath', 0),  # Convert half baths to equivalent full baths
        'year_built': df['YearBuilt']
    })
    
    # Target variable
    y = df['SalePrice']
    
    return X, y

# Initialize predictor
predictor = HousePricePredictor()

# Load and train the model
try:
    X, y = prepare_training_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    predictor.train(X_train, y_train)
    print("Model trained successfully!")
except Exception as e:
    print(f"Error loading or training model: {str(e)}")

@app.post("/predict")
async def predict_price(features: HouseFeatures):
    try:
        # Convert input features to DataFrame
        input_data = pd.DataFrame([[
            features.square_feet,
            features.bedrooms,
            features.bathrooms,
            features.year_built
        ]], columns=predictor.feature_columns)
        
        # Make prediction
        predicted_price = predictor.predict(input_data)
        
        return {
            "predicted_price": round(predicted_price, 2)
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
def read_root():
    print("Root endpoint accessed")  # Debugging line
    return {"message": "Welcome to House Price Predictor API"}
