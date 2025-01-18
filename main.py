Creating a complete Python program for a project like `traffic-optimizer` involves several components, including data acquisition, processing, predictive analytics, and real-time traffic management. Below is an illustrative example of such a program using placeholder components. This program will provide a framework, but you'll need to incorporate specific data sources and predictive models to suit your needs.

```python
import pandas as pd
import numpy as np
import requests
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def fetch_real_time_traffic_data(api_url):
    """
    Fetch real-time traffic data from a given API.
    Returns JSON data if successful or raises an error on failure.
    """
    try:
        response = requests.get(api_url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching traffic data: {e}")
        return None

def preprocess_data(data):
    """
    Preprocess raw traffic data for analysis.
    Convert to pandas DataFrame, handle missing values, and extract features.
    """
    try:
        # Assume data comes in as a list of dictionaries
        df = pd.DataFrame(data)
        
        # Example of feature extraction, replace with actual traffic data features
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        
        # Handle missing values
        df.fillna(df.mean(), inplace=True)
        
        return df
    except Exception as e:
        print(f"Error preprocessing data: {e}")
        return None

def train_predictive_model(df):
    """
    Train a predictive model to forecast traffic congestion.
    Splits the data into training and test sets, and returns the trained model.
    """
    try:
        # Define features and target variable
        X = df[['hour', 'day_of_week']]
        y = df['traffic_congestion']  # Replace with actual field for congestion

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize and train model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Evaluate model
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f"Model Mean Squared Error: {mse:.2f}")

        return model
    except Exception as e:
        print(f"Error training model: {e}")
        return None

def optimize_traffic(predictions):
    """
    Implement traffic optimization strategies based on the model's predictions.
    Placeholder function: replaces with actual optimization logic.
    """
    try:
        # Example: Optimize traffic lights, suggest alternate routes, etc.
        # This is a placeholder for your optimization logic.
        for prediction in predictions:
            if prediction > 0.8:  # Threshold for high congestion
                print("High congestion predicted. Implementing optimization strategies.")
            else:
                print("Traffic flow is normal.")
    except Exception as e:
        print(f"Error optimizing traffic: {e}")

def main():
    # Replace the API URL with an actual endpoint for real-time traffic data
    api_url = "https://api.example.com/traffic"  
    data = fetch_real_time_traffic_data(api_url)
    
    if data:
        # Preprocess data
        df = preprocess_data(data)
        
        if df is not None:
            # Train model
            model = train_predictive_model(df)
            
            if model:
                # Predict future traffic for optimization
                future_data = df[['hour', 'day_of_week']]  # Replace with actual future data
                predictions = model.predict(future_data)
                
                # Optimize traffic
                optimize_traffic(predictions)

if __name__ == "__main__":
    main()
```

### Key Notes:
- **Data Source**: Replace the dummy API URL with a real API endpoint providing traffic data.
- **Feature Engineering**: Adapt the preprocessing to include meaningful features specific to your traffic data.
- **Predictive Model**: The model here is a simple linear regression for illustration; consider more sophisticated models like time series forecasting or machine learning algorithms tailored to traffic data.
- **Optimization Logic**: The placeholder logic should be replaced with real strategies (e.g., dynamic traffic signal timings, route suggestions).

This framework can be further developed to create a sophisticated traffic optimization system by integrating more complex models and real-time control logic.