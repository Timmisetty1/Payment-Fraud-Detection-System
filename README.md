# Payment Fraud Detection System

A comprehensive machine learning system for detecting fraudulent payment transactions using Python and scikit-learn. This system implements advanced feature engineering, multiple classification algorithms, and a production-ready risk scoring pipeline.

## Features

### 🔍 Advanced Feature Engineering
- **Transaction Velocity Features**: Track transaction frequency and amounts over time windows
- **Merchant Pattern Analysis**: Identify suspicious merchant behaviors and customer-merchant relationships
- **Device Characteristics**: Analyze device usage patterns and multi-customer device sharing
- **Temporal Features**: Extract hour of day, day of week, and time-based patterns

### 🤖 Multiple Classification Models
- **Logistic Regression**: Fast, interpretable baseline model with balanced class weights
- **Random Forest**: Ensemble method capturing complex non-linear patterns
- **XGBoost**: Gradient boosting with advanced handling of imbalanced data

### 📊 Comprehensive Evaluation
- **Precision-Recall Curves**: Optimal for imbalanced fraud detection scenarios
- **ROC Curves**: Traditional performance visualization across thresholds
- **Confusion Matrices**: Detailed breakdown of model predictions
- **Feature Importance**: Understand which features drive fraud detection

### 🎯 Risk Scoring Pipeline
- Real-time transaction risk assessment
- Four-tier risk categorization (Low, Medium, High, Critical)
- Batch processing capabilities
- Detailed reporting and high-risk transaction flagging

## Project Structure

```
Payment-Fraud-Detection-System/
├── main.py                          # Main orchestration script
├── requirements.txt                 # Python dependencies
├── README.md                        # Project documentation
├── data/                           # Generated transaction data
│   └── transactions.csv
├── src/                            # Source code
│   ├── data_generator.py          # Simulated transaction data generator
│   ├── risk_scoring_pipeline.py   # Production risk scoring pipeline
│   ├── features/                  # Feature engineering modules
│   │   └── feature_engineering.py
│   ├── models/                    # Classification models
│   │   └── fraud_classifiers.py
│   └── utils/                     # Evaluation utilities
│       └── model_evaluation.py
└── output/                         # Results and visualizations
    ├── precision_recall_curves.png
    ├── roc_curves.png
    ├── confusion_matrices.png
    ├── feature_importance_rf.png
    ├── feature_importance_xgb.png
    ├── model_comparison.txt
    ├── risk_scoring_report.txt
    ├── sample_risk_scores.csv
    ├── risk_scoring_pipeline.pkl
    └── models/                     # Saved trained models
        ├── logistic_regression.pkl
        ├── random_forest.pkl
        └── xgboost.pkl
```

## Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Timmisetty1/Payment-Fraud-Detection-System.git
cd Payment-Fraud-Detection-System
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

Run the complete fraud detection pipeline:

```bash
python main.py
```

This will:
1. Generate 10,000 simulated payment transactions
2. Engineer 27 features including velocity, merchant, and device patterns
3. Train three classification models (Logistic Regression, Random Forest, XGBoost)
4. Evaluate models using precision-recall curves
5. Create a risk scoring pipeline with the best model
6. Generate comprehensive reports and visualizations

### Using Individual Components

#### 1. Generate Transaction Data

```python
from src.data_generator import generate_and_save_data

# Generate 5000 transactions with 2% fraud rate
df = generate_and_save_data(
    output_path='data/transactions.csv',
    n_samples=5000
)
```

#### 2. Feature Engineering

```python
from src.features.feature_engineering import FraudFeatureEngineer

# Initialize feature engineer
engineer = FraudFeatureEngineer()

# Engineer all features
df_features = engineer.engineer_all_features(df, is_training=True)

# Get feature columns
feature_cols = engineer.get_feature_columns()

# Prepare features for modeling
X = engineer.prepare_features(df_features, feature_cols, fit_scaler=True)
```

#### 3. Train Models

```python
from src.models.fraud_classifiers import train_all_models
from sklearn.model_selection import train_test_split

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Train all models
results = train_all_models(X_train, y_train, X_test, y_test)
```

#### 4. Evaluate Models

```python
from src.utils.model_evaluation import (
    plot_precision_recall_curves,
    generate_model_comparison_report
)

# Generate precision-recall curves
plot_precision_recall_curves(results)

# Generate comparison report
generate_model_comparison_report(results)
```

#### 5. Risk Scoring Pipeline

```python
from src.risk_scoring_pipeline import create_risk_scoring_pipeline

# Create pipeline with best model
pipeline = create_risk_scoring_pipeline(
    model=best_model,
    feature_engineer=engineer,
    feature_cols=feature_cols,
    threshold=0.5
)

# Score transactions
results_df = pipeline.process_batch(new_transactions)

# Get high-risk transactions
high_risk = pipeline.get_high_risk_transactions(results_df, 'High')
```

## Model Performance

The system trains and compares three models:

| Model | Key Strengths | Use Case |
|-------|--------------|----------|
| **Logistic Regression** | Fast, interpretable, good baseline | Quick deployment, regulatory requirements |
| **Random Forest** | Handles non-linearity, robust to outliers | Complex patterns, feature importance |
| **XGBoost** | Best performance, handles imbalance well | Production deployment, highest accuracy |

Models are evaluated using:
- **Average Precision Score**: Primary metric for imbalanced data
- **ROC AUC Score**: Overall discrimination ability
- **Precision-Recall Curves**: Trade-off visualization
- **Confusion Matrices**: Detailed prediction breakdown

## Engineered Features

### Transaction Velocity (5 features)
- `tx_count_24h`: Number of transactions in last 24 hours
- `tx_amount_24h`: Total amount spent in last 24 hours
- `avg_tx_amount`: Customer's average transaction amount
- `amount_vs_avg`: Current amount vs. customer average
- `tx_velocity`: Transactions per hour

### Merchant Patterns (5 features)
- `merchant_fraud_rate`: Historical fraud rate for merchant
- `merchant_tx_count`: Total transactions at merchant
- `merchant_avg_amount`: Average amount at merchant
- `customer_merchant_tx_count`: Customer's history with merchant
- `is_new_merchant`: First time customer uses merchant

### Device Characteristics (5 features)
- `device_unique_customers`: Number of customers using device
- `device_tx_count`: Total transactions from device
- `device_avg_amount`: Average amount from device
- `device_fraud_rate`: Historical fraud rate for device
- `device_shared`: Whether device is shared

### Time Features (4 features)
- `hour_of_day`: Hour of transaction (0-23)
- `day_of_week`: Day of week (0-6)
- `is_weekend`: Weekend indicator
- `is_night`: Night time indicator (10 PM - 6 AM)

### Base Features (8 features)
- Transaction amount
- Distance from home
- Time since last transaction
- Card present indicator
- Transaction type (encoded)
- Country (encoded)
- Merchant ID patterns
- Device ID patterns

## Risk Scoring

The pipeline assigns risk scores to transactions:

| Risk Category | Probability Range | Action |
|---------------|------------------|--------|
| **Low** | 0% - 20% | Auto-approve |
| **Medium** | 20% - 50% | Standard review |
| **High** | 50% - 80% | Priority review |
| **Critical** | 80% - 100% | Immediate investigation |

## Output Files

After running `main.py`, the following outputs are generated:

### Visualizations
- `precision_recall_curves.png`: Compare model performance
- `roc_curves.png`: ROC curve comparison
- `confusion_matrices.png`: Prediction breakdowns
- `feature_importance_rf.png`: Random Forest feature importance
- `feature_importance_xgb.png`: XGBoost feature importance

### Reports
- `model_comparison.txt`: Detailed model performance comparison
- `risk_scoring_report.txt`: Risk scoring statistics and high-risk transactions

### Data
- `transactions.csv`: Generated transaction dataset
- `sample_risk_scores.csv`: Sample scored transactions

### Models
- `risk_scoring_pipeline.pkl`: Complete production-ready pipeline
- `logistic_regression.pkl`: Trained logistic regression model
- `random_forest.pkl`: Trained random forest model
- `xgboost.pkl`: Trained XGBoost model

## Technical Details

### Dependencies
- **numpy 1.24.3**: Numerical computations
- **pandas 2.0.3**: Data manipulation
- **scikit-learn 1.3.0**: Machine learning algorithms
- **xgboost 1.7.6**: Gradient boosting
- **matplotlib 3.7.2**: Visualization
- **seaborn 0.12.2**: Statistical visualization
- **imbalanced-learn 0.11.0**: Handling imbalanced datasets

### Key Algorithms
- **Logistic Regression**: L-BFGS solver with balanced class weights
- **Random Forest**: 100 trees, max depth 10, balanced weights
- **XGBoost**: 100 estimators, scale_pos_weight=50 for imbalance

### Data Characteristics
- Default: 10,000 transactions
- Fraud ratio: ~2% (realistic imbalance)
- Features: 27 engineered features
- Categories: Online, in-store, mobile, ATM transactions
- Geographic: 8 countries (US-dominant)

## Future Enhancements

Potential improvements for production deployment:

1. **Real-time Processing**: Stream processing with Apache Kafka
2. **Model Monitoring**: Drift detection and automated retraining
3. **Deep Learning**: LSTM networks for sequential patterns
4. **Explainability**: SHAP values for individual predictions
5. **A/B Testing**: Framework for testing new models
6. **API Integration**: REST API for real-time scoring
7. **Database Integration**: PostgreSQL/MongoDB for transaction storage
8. **Alert System**: Automated notifications for high-risk transactions

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is available for educational and research purposes.

## Acknowledgments

- Built with scikit-learn and XGBoost
- Inspired by real-world fraud detection challenges
- Designed for educational purposes and prototype development

## Contact

For questions or feedback, please open an issue on GitHub.

---

**Note**: This system uses simulated data for demonstration purposes. For production use, integrate with real transaction data and conduct thorough validation.