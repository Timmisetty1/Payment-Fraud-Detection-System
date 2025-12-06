"""
Risk Scoring Pipeline
Processes transactions and assigns fraud risk scores
"""

import numpy as np
import pandas as pd
import pickle
import os


class RiskScoringPipeline:
    """Pipeline for scoring transaction fraud risk"""
    
    def __init__(self, model, feature_engineer, feature_cols, threshold=0.5):
        """
        Initialize risk scoring pipeline
        
        Args:
            model: Trained fraud detection model
            feature_engineer: Feature engineering instance
            feature_cols (list): List of feature columns
            threshold (float): Probability threshold for fraud classification
        """
        self.model = model
        self.feature_engineer = feature_engineer
        self.feature_cols = feature_cols
        self.threshold = threshold
        
    def assign_risk_category(self, probability):
        """
        Assign risk category based on fraud probability
        
        Args:
            probability (float): Fraud probability
            
        Returns:
            str: Risk category
        """
        if probability < 0.2:
            return 'Low'
        elif probability < 0.5:
            return 'Medium'
        elif probability < 0.8:
            return 'High'
        else:
            return 'Critical'
    
    def process_transaction(self, transaction_df):
        """
        Process a single transaction or batch of transactions
        
        Args:
            transaction_df (pd.DataFrame): Transaction data
            
        Returns:
            pd.DataFrame: Transaction data with risk scores
        """
        # Engineer features
        df_features = self.feature_engineer.engineer_all_features(
            transaction_df, 
            is_training=False
        )
        
        # Prepare features for prediction
        X = self.feature_engineer.prepare_features(
            df_features, 
            self.feature_cols, 
            fit_scaler=False
        )
        
        # Get fraud probabilities
        fraud_probabilities = self.model.predict_proba(X)[:, 1]
        
        # Add risk scores to DataFrame
        results_df = transaction_df.copy()
        results_df['fraud_probability'] = fraud_probabilities
        results_df['fraud_prediction'] = (fraud_probabilities >= self.threshold).astype(int)
        results_df['risk_category'] = [
            self.assign_risk_category(prob) for prob in fraud_probabilities
        ]
        
        return results_df
    
    def process_batch(self, transactions_df, output_path=None):
        """
        Process a batch of transactions
        
        Args:
            transactions_df (pd.DataFrame): Batch of transactions
            output_path (str): Optional path to save results
            
        Returns:
            pd.DataFrame: Processed transactions with risk scores
        """
        print(f"Processing batch of {len(transactions_df)} transactions...")
        
        results_df = self.process_transaction(transactions_df)
        
        # Print summary
        print("\nRisk Scoring Summary:")
        print(f"Total transactions: {len(results_df)}")
        print(f"Predicted frauds: {results_df['fraud_prediction'].sum()}")
        print(f"Fraud rate: {results_df['fraud_prediction'].mean():.2%}")
        print("\nRisk Category Distribution:")
        print(results_df['risk_category'].value_counts().sort_index())
        
        # Save results if path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            results_df.to_csv(output_path, index=False)
            print(f"\nResults saved to {output_path}")
        
        return results_df
    
    def get_high_risk_transactions(self, results_df, risk_level='High'):
        """
        Filter high-risk transactions for review
        
        Args:
            results_df (pd.DataFrame): Processed transactions
            risk_level (str): Minimum risk level ('Medium', 'High', 'Critical')
            
        Returns:
            pd.DataFrame: High-risk transactions
        """
        risk_order = ['Low', 'Medium', 'High', 'Critical']
        min_risk_idx = risk_order.index(risk_level)
        
        high_risk = results_df[
            results_df['risk_category'].apply(lambda x: risk_order.index(x) >= min_risk_idx)
        ]
        
        return high_risk.sort_values('fraud_probability', ascending=False)
    
    def generate_report(self, results_df, save_path='output/risk_scoring_report.txt'):
        """
        Generate a detailed risk scoring report
        
        Args:
            results_df (pd.DataFrame): Processed transactions with risk scores
            save_path (str): Path to save the report
        """
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("RISK SCORING PIPELINE REPORT")
        report_lines.append("="*80)
        report_lines.append("")
        
        # Overall statistics
        report_lines.append("OVERALL STATISTICS")
        report_lines.append("-"*80)
        report_lines.append(f"Total transactions processed: {len(results_df)}")
        report_lines.append(f"Predicted fraudulent transactions: {results_df['fraud_prediction'].sum()}")
        report_lines.append(f"Fraud rate: {results_df['fraud_prediction'].mean():.2%}")
        report_lines.append(f"Average fraud probability: {results_df['fraud_probability'].mean():.4f}")
        report_lines.append("")
        
        # Risk category distribution
        report_lines.append("RISK CATEGORY DISTRIBUTION")
        report_lines.append("-"*80)
        risk_dist = results_df['risk_category'].value_counts()
        for category in ['Low', 'Medium', 'High', 'Critical']:
            count = risk_dist.get(category, 0)
            percentage = count / len(results_df) * 100
            report_lines.append(f"{category:<12}: {count:>6} ({percentage:>5.1f}%)")
        report_lines.append("")
        
        # High-risk transactions
        high_risk = self.get_high_risk_transactions(results_df, 'High')
        report_lines.append("HIGH-RISK TRANSACTIONS (Top 10)")
        report_lines.append("-"*80)
        report_lines.append(f"{'Transaction ID':<15} {'Amount':<12} {'Risk Category':<15} {'Fraud Prob':<12}")
        report_lines.append("-"*80)
        
        for idx, row in high_risk.head(10).iterrows():
            report_lines.append(
                f"{idx:<15} ${row['amount']:<11.2f} {row['risk_category']:<15} {row['fraud_probability']:<12.4f}"
            )
        report_lines.append("")
        
        # Save report
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"\nRisk scoring report saved to {save_path}")
        
        # Also print to console
        print('\n'.join(report_lines))
    
    def save_pipeline(self, filepath='output/risk_scoring_pipeline.pkl'):
        """
        Save the entire pipeline to disk
        
        Args:
            filepath (str): Path to save the pipeline
        """
        pipeline_data = {
            'model': self.model,
            'feature_engineer': self.feature_engineer,
            'feature_cols': self.feature_cols,
            'threshold': self.threshold
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(pipeline_data, f)
        
        print(f"Risk scoring pipeline saved to {filepath}")
    
    @classmethod
    def load_pipeline(cls, filepath='output/risk_scoring_pipeline.pkl'):
        """
        Load a saved pipeline from disk
        
        Args:
            filepath (str): Path to load the pipeline from
            
        Returns:
            RiskScoringPipeline: Loaded pipeline instance
        """
        with open(filepath, 'rb') as f:
            pipeline_data = pickle.load(f)
        
        pipeline = cls(
            model=pipeline_data['model'],
            feature_engineer=pipeline_data['feature_engineer'],
            feature_cols=pipeline_data['feature_cols'],
            threshold=pipeline_data['threshold']
        )
        
        print(f"Risk scoring pipeline loaded from {filepath}")
        return pipeline


def create_risk_scoring_pipeline(model, feature_engineer, feature_cols, threshold=0.5):
    """
    Factory function to create a risk scoring pipeline
    
    Args:
        model: Trained fraud detection model
        feature_engineer: Feature engineering instance
        feature_cols (list): List of feature columns
        threshold (float): Probability threshold for fraud classification
        
    Returns:
        RiskScoringPipeline: Configured pipeline instance
    """
    return RiskScoringPipeline(model, feature_engineer, feature_cols, threshold)


if __name__ == '__main__':
    print("Risk scoring pipeline module loaded successfully")
