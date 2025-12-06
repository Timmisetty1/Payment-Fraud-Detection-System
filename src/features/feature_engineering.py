"""
Feature Engineering Module for Payment Fraud Detection
Implements transaction velocity, merchant patterns, and device characteristics
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


class FraudFeatureEngineer:
    """Engineer features for fraud detection"""
    
    def __init__(self):
        """Initialize feature engineer"""
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def engineer_velocity_features(self, df):
        """
        Engineer transaction velocity features
        
        Args:
            df (pd.DataFrame): Input transaction data
            
        Returns:
            pd.DataFrame: DataFrame with velocity features added
        """
        df = df.copy()
        
        # Sort by customer and timestamp
        df = df.sort_values(['customer_id', 'timestamp'])
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Set timestamp as index for rolling operations
        df_indexed = df.set_index('timestamp')
        
        # Transaction count per customer in last 24 hours
        tx_count_24h = df_indexed.groupby('customer_id')['amount'].rolling(
            window='24h', closed='left'
        ).count().reset_index(level=0, drop=True)
        df['tx_count_24h'] = tx_count_24h.values
        
        # Total amount spent per customer in last 24 hours
        tx_amount_24h = df_indexed.groupby('customer_id')['amount'].rolling(
            window='24h', closed='left'
        ).sum().reset_index(level=0, drop=True)
        df['tx_amount_24h'] = tx_amount_24h.values
        
        # Average transaction amount per customer
        df['avg_tx_amount'] = df.groupby('customer_id')['amount'].transform('mean')
        
        # Transaction amount vs customer average (normalized)
        df['amount_vs_avg'] = (df['amount'] - df['avg_tx_amount']) / (df['avg_tx_amount'] + 1e-5)
        
        # Velocity: transactions per hour
        df['tx_velocity'] = df['tx_count_24h'] / 24.0
        
        # Fill NaN values with 0 (for first transactions)
        velocity_cols = ['tx_count_24h', 'tx_amount_24h', 'tx_velocity']
        df[velocity_cols] = df[velocity_cols].fillna(0)
        
        return df
    
    def engineer_merchant_features(self, df):
        """
        Engineer merchant pattern features
        
        Args:
            df (pd.DataFrame): Input transaction data
            
        Returns:
            pd.DataFrame: DataFrame with merchant features added
        """
        df = df.copy()
        
        # Merchant fraud rate (calculated on training data)
        merchant_fraud_rate = df.groupby('merchant_id')['is_fraud'].transform('mean')
        df['merchant_fraud_rate'] = merchant_fraud_rate
        
        # Number of transactions per merchant
        df['merchant_tx_count'] = df.groupby('merchant_id')['merchant_id'].transform('count')
        
        # Average transaction amount per merchant
        df['merchant_avg_amount'] = df.groupby('merchant_id')['amount'].transform('mean')
        
        # Customer's transaction count with this merchant
        df['customer_merchant_tx_count'] = df.groupby(
            ['customer_id', 'merchant_id']
        )['merchant_id'].transform('count')
        
        # Is this a new merchant for the customer?
        df['is_new_merchant'] = (df['customer_merchant_tx_count'] == 1).astype(int)
        
        return df
    
    def engineer_device_features(self, df):
        """
        Engineer device characteristic features
        
        Args:
            df (pd.DataFrame): Input transaction data
            
        Returns:
            pd.DataFrame: DataFrame with device features added
        """
        df = df.copy()
        
        # Number of unique customers per device
        df['device_unique_customers'] = df.groupby('device_id')['customer_id'].transform('nunique')
        
        # Number of transactions per device
        df['device_tx_count'] = df.groupby('device_id')['device_id'].transform('count')
        
        # Average transaction amount per device
        df['device_avg_amount'] = df.groupby('device_id')['amount'].transform('mean')
        
        # Device fraud rate
        device_fraud_rate = df.groupby('device_id')['is_fraud'].transform('mean')
        df['device_fraud_rate'] = device_fraud_rate
        
        # Is device used by multiple customers? (potential fraud indicator)
        df['device_shared'] = (df['device_unique_customers'] > 1).astype(int)
        
        return df
    
    def engineer_time_features(self, df):
        """
        Engineer time-based features
        
        Args:
            df (pd.DataFrame): Input transaction data
            
        Returns:
            pd.DataFrame: DataFrame with time features added
        """
        df = df.copy()
        
        # Convert timestamp to datetime if not already
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Hour of day
        df['hour_of_day'] = df['timestamp'].dt.hour
        
        # Day of week (0=Monday, 6=Sunday)
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        
        # Is weekend?
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Is night time? (10 PM - 6 AM)
        df['is_night'] = ((df['hour_of_day'] >= 22) | (df['hour_of_day'] <= 6)).astype(int)
        
        return df
    
    def engineer_all_features(self, df, is_training=True):
        """
        Apply all feature engineering steps
        
        Args:
            df (pd.DataFrame): Input transaction data
            is_training (bool): Whether this is training data
            
        Returns:
            pd.DataFrame: DataFrame with all engineered features
        """
        print("Engineering features...")
        
        # Make a copy
        df = df.copy()
        
        # Engineer different feature groups
        df = self.engineer_time_features(df)
        df = self.engineer_velocity_features(df)
        df = self.engineer_merchant_features(df)
        df = self.engineer_device_features(df)
        
        # Encode categorical variables
        categorical_cols = ['transaction_type', 'country']
        for col in categorical_cols:
            if is_training:
                self.label_encoders[col] = LabelEncoder()
                df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col])
            else:
                # Handle unseen categories in test data
                df[f'{col}_encoded'] = df[col].apply(
                    lambda x: self.label_encoders[col].transform([x])[0] 
                    if x in self.label_encoders[col].classes_ 
                    else -1
                )
        
        # Convert boolean to int
        df['card_present'] = df['card_present'].astype(int)
        
        print(f"Feature engineering complete. Total features: {len(df.columns)}")
        return df
    
    def get_feature_columns(self):
        """
        Get list of feature columns to use for modeling
        
        Returns:
            list: List of feature column names
        """
        feature_cols = [
            # Original features
            'amount',
            'distance_from_home_km',
            'time_since_last_transaction_hours',
            'card_present',
            
            # Time features
            'hour_of_day',
            'day_of_week',
            'is_weekend',
            'is_night',
            
            # Velocity features
            'tx_count_24h',
            'tx_amount_24h',
            'avg_tx_amount',
            'amount_vs_avg',
            'tx_velocity',
            
            # Merchant features
            'merchant_fraud_rate',
            'merchant_tx_count',
            'merchant_avg_amount',
            'customer_merchant_tx_count',
            'is_new_merchant',
            
            # Device features
            'device_unique_customers',
            'device_tx_count',
            'device_avg_amount',
            'device_fraud_rate',
            'device_shared',
            
            # Encoded categorical features
            'transaction_type_encoded',
            'country_encoded',
        ]
        
        return feature_cols
    
    def prepare_features(self, df, feature_cols, fit_scaler=True):
        """
        Prepare features for modeling (scaling, etc.)
        
        Args:
            df (pd.DataFrame): Input DataFrame with engineered features
            feature_cols (list): List of feature columns to use
            fit_scaler (bool): Whether to fit the scaler (True for training)
            
        Returns:
            np.ndarray: Scaled feature matrix
        """
        X = df[feature_cols].values
        
        # Replace any remaining NaN or inf values
        X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
        
        if fit_scaler:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        return X_scaled


if __name__ == '__main__':
    # Example usage
    from src.data_generator import generate_and_save_data
    
    # Generate sample data
    df = generate_and_save_data(n_samples=1000)
    
    # Engineer features
    engineer = FraudFeatureEngineer()
    df_features = engineer.engineer_all_features(df, is_training=True)
    
    # Get feature columns
    feature_cols = engineer.get_feature_columns()
    print(f"\nFeature columns: {feature_cols}")
    
    # Prepare features
    X = engineer.prepare_features(df_features, feature_cols, fit_scaler=True)
    print(f"\nFeature matrix shape: {X.shape}")
