import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# FastAPI initialization
app = FastAPI()

# Define input model for FastAPI
class HousePriceRequest(BaseModel):
    features: dict

class HousePricePredictor:
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.model = LinearRegression()
        self.imputer = SimpleImputer(strategy='mean')  # Imputer for numeric data
        self.cat_imputer = SimpleImputer(strategy='most_frequent')  # Imputer for categorical data
        self.df = pd.read_csv("/app/train.csv")
        self.train_model()  # Train model when the app starts

    def load_and_prepare_data(self):
        # Separate numeric and categorical columns
        numeric_cols = self.df.select_dtypes(include=['number']).columns
        categorical_cols = self.df.select_dtypes(include=['object']).columns

        # Impute missing values for numeric columns with mean strategy
        self.df[numeric_cols] = self.imputer.fit_transform(self.df[numeric_cols])

        # Impute missing values for categorical columns with most frequent strategy
        self.df[categorical_cols] = self.cat_imputer.fit_transform(self.df[categorical_cols])

        # One-hot encode categorical features
        self.df = pd.get_dummies(self.df, drop_first=True)

        # Features and target variable
        X = self.df.drop(columns='SalePrice')  # Assuming 'SalePrice' is the target column
        y = self.df['SalePrice']
        
        return X, y

    def train_model(self):
        X, y = self.load_and_prepare_data()
        # Train a linear regression model
        self.model.fit(X, y)
        joblib.dump(self.model, 'house_price_model.pkl')  # Save the trained model
        joblib.dump(self.imputer, 'numeric_imputer.pkl')  # Save the imputer for numeric data
        joblib.dump(self.cat_imputer, 'categorical_imputer.pkl')  # Save the imputer for categorical data

    def predict(self, input_data: dict):
        # Preprocess and impute input data as per the training set
        input_df = pd.DataFrame([input_data])
        numeric_cols = input_df.select_dtypes(include=['number']).columns
        categorical_cols = input_df.select_dtypes(include=['object']).columns
        
        input_df[numeric_cols] = self.imputer.transform(input_df[numeric_cols])
        input_df[categorical_cols] = self.cat_imputer.transform(input_df[categorical_cols])

        # One-hot encode the input data in the same way as the training data
        input_df = pd.get_dummies(input_df, drop_first=True)

        # Align the columns of input_df with the trained model's feature space
        X_train, _ = self.load_and_prepare_data()  # Get training data to align columns
        input_df = input_df.reindex(columns=X_train.columns, fill_value=0)

        # Load the trained model and predict
        model = joblib.load('house_price_model.pkl')
        prediction = model.predict(input_df)
        return prediction[0]

# Initialize the predictor
predictor = HousePricePredictor(dataset_path="C:/Users/Admin/OneDrive/Desktop/python_mlops/train.csv")

# FastAPI endpoint for prediction
@app.post("/predict/")
async def predict(data: dict):
    try:
        # Extract features from the request
        features = data.get('features', {})
        # Predict using the model
        prediction = predictor.predict(features)
        return {"prediction": prediction}
    except Exception as e:
        return {"error": str(e)}

@app.get("/")
def read_root():
    return {"message": "Welcome to House Price Predictor API"}
