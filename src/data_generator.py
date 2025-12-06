"""
Data Generator for Payment Fraud Detection System
Generates simulated payment transaction data with various features
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


class TransactionDataGenerator:
    """Generate simulated payment transaction data"""
    
    def __init__(self, n_samples=10000, fraud_ratio=0.02, random_state=42):
        """
        Initialize the data generator
        
        Args:
            n_samples (int): Number of transactions to generate
            fraud_ratio (float): Proportion of fraudulent transactions
            random_state (int): Random seed for reproducibility
        """
        self.n_samples = n_samples
        self.fraud_ratio = fraud_ratio
        self.random_state = random_state
        np.random.seed(random_state)
        
    def generate_data(self):
        """
        Generate synthetic transaction dataset
        
        Returns:
            pd.DataFrame: Generated transaction data
        """
        n_fraud = int(self.n_samples * self.fraud_ratio)
        n_normal = self.n_samples - n_fraud
        
        # Generate timestamps
        start_date = datetime(2023, 1, 1)
        timestamps = [start_date + timedelta(minutes=np.random.randint(0, 525600)) 
                     for _ in range(self.n_samples)]
        
        # Generate transaction amounts
        # Normal transactions: smaller amounts
        normal_amounts = np.random.lognormal(mean=3.5, sigma=1.2, size=n_normal)
        # Fraudulent transactions: larger amounts on average
        fraud_amounts = np.random.lognormal(mean=4.5, sigma=1.5, size=n_fraud)
        amounts = np.concatenate([normal_amounts, fraud_amounts])
        
        # Generate merchant IDs (500 unique merchants)
        n_merchants = 500
        # Normal transactions: distributed across many merchants
        normal_merchants = np.random.randint(0, n_merchants, size=n_normal)
        # Fraudulent transactions: concentrated in fewer merchants
        fraud_merchants = np.random.choice(
            np.random.randint(0, n_merchants // 5), size=n_fraud
        )
        merchant_ids = np.concatenate([normal_merchants, fraud_merchants])
        
        # Generate device IDs (1000 unique devices)
        n_devices = 1000
        normal_devices = np.random.randint(0, n_devices, size=n_normal)
        # Fraudulent transactions: more likely to use same device multiple times
        fraud_devices = np.random.choice(
            np.random.randint(0, n_devices // 10), size=n_fraud
        )
        device_ids = np.concatenate([normal_devices, fraud_devices])
        
        # Generate customer IDs (2000 unique customers)
        n_customers = 2000
        customer_ids = np.random.randint(0, n_customers, size=self.n_samples)
        
        # Generate transaction types
        transaction_types = np.random.choice(
            ['online', 'in_store', 'mobile', 'atm'], 
            size=self.n_samples,
            p=[0.4, 0.3, 0.25, 0.05]
        )
        # Fraudulent transactions more likely to be online
        fraud_type_mask = np.random.random(n_fraud) < 0.7
        transaction_types[-n_fraud:][fraud_type_mask] = 'online'
        
        # Generate countries (mostly domestic, some international)
        countries = np.random.choice(
            ['US', 'UK', 'CA', 'DE', 'FR', 'CN', 'BR', 'IN'],
            size=self.n_samples,
            p=[0.7, 0.08, 0.07, 0.05, 0.04, 0.03, 0.02, 0.01]
        )
        # Fraudulent transactions more likely to be international
        fraud_country_mask = np.random.random(n_fraud) < 0.4
        fraud_countries = np.random.choice(['CN', 'BR', 'IN'], size=np.sum(fraud_country_mask))
        transaction_types_temp = transaction_types.copy()
        countries_list = list(countries)
        for i, mask_val in enumerate(fraud_country_mask):
            if mask_val and i < len(fraud_countries):
                countries_list[-(n_fraud-i)] = fraud_countries[i]
        countries = np.array(countries_list)
        
        # Generate card present flag
        card_present = np.random.choice([True, False], size=self.n_samples, p=[0.4, 0.6])
        # Fraudulent transactions less likely to have card present
        card_present[-n_fraud:] = np.random.choice([True, False], size=n_fraud, p=[0.1, 0.9])
        
        # Generate distance from home (in km)
        # Normal transactions: mostly local
        normal_distance = np.random.gamma(shape=2, scale=10, size=n_normal)
        # Fraudulent transactions: potentially further from home
        fraud_distance = np.random.gamma(shape=3, scale=50, size=n_fraud)
        distance_from_home = np.concatenate([normal_distance, fraud_distance])
        
        # Generate time since last transaction (in hours)
        # Normal transactions: varied intervals
        normal_time_since = np.random.exponential(scale=24, size=n_normal)
        # Fraudulent transactions: often in quick succession
        fraud_time_since = np.random.exponential(scale=2, size=n_fraud)
        time_since_last = np.concatenate([normal_time_since, fraud_time_since])
        
        # Generate labels
        is_fraud = np.concatenate([np.zeros(n_normal), np.ones(n_fraud)])
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': timestamps,
            'amount': amounts,
            'merchant_id': merchant_ids,
            'device_id': device_ids,
            'customer_id': customer_ids,
            'transaction_type': transaction_types,
            'country': countries,
            'card_present': card_present,
            'distance_from_home_km': distance_from_home,
            'time_since_last_transaction_hours': time_since_last,
            'is_fraud': is_fraud.astype(int)
        })
        
        # Shuffle the dataset
        df = df.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        
        # Sort by timestamp for realistic ordering
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df


def generate_and_save_data(output_path='data/transactions.csv', n_samples=10000):
    """
    Generate and save transaction data to CSV
    
    Args:
        output_path (str): Path to save the CSV file
        n_samples (int): Number of samples to generate
    """
    generator = TransactionDataGenerator(n_samples=n_samples)
    df = generator.generate_data()
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} transactions and saved to {output_path}")
    print(f"Fraud ratio: {df['is_fraud'].mean():.2%}")
    return df


if __name__ == '__main__':
    df = generate_and_save_data()
    print("\nDataset preview:")
    print(df.head())
    print("\nDataset info:")
    print(df.info())
