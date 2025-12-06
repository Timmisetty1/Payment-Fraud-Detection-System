"""
Main Script for Payment Fraud Detection System
Orchestrates data generation, feature engineering, model training, evaluation, and risk scoring
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_generator import generate_and_save_data
from features.feature_engineering import FraudFeatureEngineer
from models.fraud_classifiers import train_all_models
from utils.model_evaluation import (
    plot_precision_recall_curves,
    plot_roc_curves,
    plot_confusion_matrices,
    plot_feature_importance,
    generate_model_comparison_report
)
from risk_scoring_pipeline import create_risk_scoring_pipeline


def main():
    """Main execution function"""
    
    print("="*80)
    print("PAYMENT FRAUD DETECTION SYSTEM")
    print("="*80)
    print()
    
    # Configuration
    N_SAMPLES = 10000
    TEST_SIZE = 0.3
    RANDOM_STATE = 42
    
    # Step 1: Generate transaction data
    print("\n" + "="*80)
    print("STEP 1: GENERATING TRANSACTION DATA")
    print("="*80)
    
    data_path = 'data/transactions.csv'
    if os.path.exists(data_path):
        print(f"Loading existing data from {data_path}")
        df = pd.read_csv(data_path)
    else:
        print(f"Generating {N_SAMPLES} simulated transactions...")
        df = generate_and_save_data(output_path=data_path, n_samples=N_SAMPLES)
    
    print(f"\nDataset shape: {df.shape}")
    print(f"Fraud ratio: {df['is_fraud'].mean():.2%}")
    print(f"Fraudulent transactions: {df['is_fraud'].sum()}")
    print(f"Legitimate transactions: {(df['is_fraud'] == 0).sum()}")
    
    # Step 2: Feature Engineering
    print("\n" + "="*80)
    print("STEP 2: FEATURE ENGINEERING")
    print("="*80)
    
    feature_engineer = FraudFeatureEngineer()
    df_features = feature_engineer.engineer_all_features(df, is_training=True)
    
    # Get feature columns
    feature_cols = feature_engineer.get_feature_columns()
    print(f"\nNumber of engineered features: {len(feature_cols)}")
    print(f"Feature columns: {feature_cols}")
    
    # Step 3: Prepare data for modeling
    print("\n" + "="*80)
    print("STEP 3: PREPARING DATA FOR MODELING")
    print("="*80)
    
    # Split data
    train_df, test_df = train_test_split(
        df_features, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE,
        stratify=df_features['is_fraud']
    )
    
    print(f"Training set size: {len(train_df)}")
    print(f"Test set size: {len(test_df)}")
    print(f"Training fraud ratio: {train_df['is_fraud'].mean():.2%}")
    print(f"Test fraud ratio: {test_df['is_fraud'].mean():.2%}")
    
    # Prepare features
    X_train = feature_engineer.prepare_features(train_df, feature_cols, fit_scaler=True)
    X_test = feature_engineer.prepare_features(test_df, feature_cols, fit_scaler=False)
    y_train = train_df['is_fraud'].values
    y_test = test_df['is_fraud'].values
    
    print(f"\nX_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    
    # Step 4: Train and evaluate models
    print("\n" + "="*80)
    print("STEP 4: TRAINING AND EVALUATING MODELS")
    print("="*80)
    
    results = train_all_models(
        X_train, y_train, 
        X_test, y_test, 
        random_state=RANDOM_STATE
    )
    
    # Step 5: Generate evaluation visualizations
    print("\n" + "="*80)
    print("STEP 5: GENERATING EVALUATION VISUALIZATIONS")
    print("="*80)
    
    # Precision-Recall curves
    plot_precision_recall_curves(results)
    
    # ROC curves
    plot_roc_curves(results)
    
    # Confusion matrices
    plot_confusion_matrices(results)
    
    # Feature importance for tree-based models
    if 'random_forest' in results:
        plot_feature_importance(
            results['random_forest']['model'],
            feature_cols,
            save_path='output/feature_importance_rf.png'
        )
    
    if 'xgboost' in results:
        plot_feature_importance(
            results['xgboost']['model'],
            feature_cols,
            save_path='output/feature_importance_xgb.png'
        )
    
    # Generate model comparison report
    generate_model_comparison_report(results)
    
    # Step 6: Select best model and create risk scoring pipeline
    print("\n" + "="*80)
    print("STEP 6: CREATING RISK SCORING PIPELINE")
    print("="*80)
    
    # Select best model based on average precision
    best_model_name = max(
        results.keys(), 
        key=lambda k: results[k]['metrics']['avg_precision']
    )
    best_model = results[best_model_name]['model']
    
    print(f"Best model selected: {best_model.model_name}")
    print(f"Average Precision: {results[best_model_name]['metrics']['avg_precision']:.4f}")
    
    # Create risk scoring pipeline
    pipeline = create_risk_scoring_pipeline(
        model=best_model,
        feature_engineer=feature_engineer,
        feature_cols=feature_cols,
        threshold=0.5
    )
    
    # Step 7: Test risk scoring pipeline on sample data
    print("\n" + "="*80)
    print("STEP 7: TESTING RISK SCORING PIPELINE")
    print("="*80)
    
    # Use test data for demonstration
    sample_transactions = test_df.head(100)[df.columns]
    results_df = pipeline.process_batch(
        sample_transactions,
        output_path='output/sample_risk_scores.csv'
    )
    
    # Generate risk scoring report
    pipeline.generate_report(results_df)
    
    # Save pipeline for future use
    pipeline.save_pipeline()
    
    # Step 8: Save trained models
    print("\n" + "="*80)
    print("STEP 8: SAVING TRAINED MODELS")
    print("="*80)
    
    os.makedirs('output/models', exist_ok=True)
    for name, result in results.items():
        model = result['model']
        model_path = f'output/models/{name}.pkl'
        model.save_model(model_path)
    
    print("\n" + "="*80)
    print("FRAUD DETECTION SYSTEM SETUP COMPLETE")
    print("="*80)
    print("\nSummary:")
    print(f"  - Generated and processed {N_SAMPLES} transactions")
    print(f"  - Engineered {len(feature_cols)} features")
    print(f"  - Trained {len(results)} models (Logistic Regression, Random Forest, XGBoost)")
    print(f"  - Best model: {best_model.model_name}")
    print(f"  - Created risk scoring pipeline")
    print(f"\nOutputs saved to 'output/' directory:")
    print(f"  - Precision-Recall curves")
    print(f"  - ROC curves")
    print(f"  - Confusion matrices")
    print(f"  - Feature importance plots")
    print(f"  - Model comparison report")
    print(f"  - Risk scoring report")
    print(f"  - Trained models")
    print(f"  - Risk scoring pipeline")
    
    return results, pipeline


if __name__ == '__main__':
    results, pipeline = main()
