"""
Fraud Classification Models
Implements Logistic Regression, Random Forest, and XGBoost classifiers
"""

import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_auc_score


class FraudClassifier:
    """Base class for fraud classification models"""
    
    def __init__(self, model_name, model):
        """
        Initialize classifier
        
        Args:
            model_name (str): Name of the model
            model: Scikit-learn compatible model
        """
        self.model_name = model_name
        self.model = model
        self.is_trained = False
        
    def train(self, X_train, y_train):
        """
        Train the model
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        print(f"\nTraining {self.model_name}...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        print(f"{self.model_name} training complete.")
        
    def predict(self, X):
        """
        Make predictions
        
        Args:
            X: Feature matrix
            
        Returns:
            np.ndarray: Predicted labels
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Predict probabilities
        
        Args:
            X: Feature matrix
            
        Returns:
            np.ndarray: Predicted probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            dict: Dictionary of evaluation metrics
        """
        print(f"\nEvaluating {self.model_name}...")
        
        # Make predictions
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        print(f"\n{self.model_name} Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud']))
        
        print(f"\n{self.model_name} Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        # Calculate precision-recall metrics
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        avg_precision = average_precision_score(y_test, y_proba)
        roc_auc = roc_auc_score(y_test, y_proba)
        
        metrics = {
            'model_name': self.model_name,
            'precision': precision,
            'recall': recall,
            'avg_precision': avg_precision,
            'roc_auc': roc_auc,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_proba': y_proba
        }
        
        print(f"\n{self.model_name} Metrics:")
        print(f"Average Precision Score: {avg_precision:.4f}")
        print(f"ROC AUC Score: {roc_auc:.4f}")
        
        return metrics
    
    def save_model(self, filepath):
        """
        Save model to disk
        
        Args:
            filepath (str): Path to save the model
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"{self.model_name} saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load model from disk
        
        Args:
            filepath (str): Path to load the model from
        """
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
        self.is_trained = True
        print(f"{self.model_name} loaded from {filepath}")


class LogisticRegressionClassifier(FraudClassifier):
    """Logistic Regression classifier for fraud detection"""
    
    def __init__(self, random_state=42):
        """
        Initialize Logistic Regression classifier
        
        Args:
            random_state (int): Random seed
        """
        model = LogisticRegression(
            random_state=random_state,
            max_iter=1000,
            class_weight='balanced',
            solver='liblinear'
        )
        super().__init__("Logistic Regression", model)


class RandomForestClassifier_(FraudClassifier):
    """Random Forest classifier for fraud detection"""
    
    def __init__(self, random_state=42):
        """
        Initialize Random Forest classifier
        
        Args:
            random_state (int): Random seed
        """
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=4,
            random_state=random_state,
            class_weight='balanced',
            n_jobs=-1
        )
        super().__init__("Random Forest", model)


class XGBoostClassifier_(FraudClassifier):
    """XGBoost classifier for fraud detection"""
    
    def __init__(self, random_state=42):
        """
        Initialize XGBoost classifier
        
        Args:
            random_state (int): Random seed
        """
        model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            scale_pos_weight=50,  # Handle class imbalance
            use_label_encoder=False,
            eval_metric='logloss'
        )
        super().__init__("XGBoost", model)


def train_all_models(X_train, y_train, X_test, y_test, random_state=42):
    """
    Train and evaluate all models
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        random_state (int): Random seed
        
    Returns:
        dict: Dictionary of trained models and their metrics
    """
    # Initialize models
    models = {
        'logistic_regression': LogisticRegressionClassifier(random_state=random_state),
        'random_forest': RandomForestClassifier_(random_state=random_state),
        'xgboost': XGBoostClassifier_(random_state=random_state)
    }
    
    results = {}
    
    # Train and evaluate each model
    for name, model in models.items():
        model.train(X_train, y_train)
        metrics = model.evaluate(X_test, y_test)
        results[name] = {
            'model': model,
            'metrics': metrics
        }
    
    return results


if __name__ == '__main__':
    # Example usage with synthetic data
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    # Generate synthetic data
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        weights=[0.98, 0.02],
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Train all models
    results = train_all_models(X_train, y_train, X_test, y_test)
    
    print("\n" + "="*50)
    print("Model Performance Summary")
    print("="*50)
    for name, result in results.items():
        metrics = result['metrics']
        print(f"\n{metrics['model_name']}:")
        print(f"  Average Precision: {metrics['avg_precision']:.4f}")
        print(f"  ROC AUC: {metrics['roc_auc']:.4f}")
