"""
Model Evaluation Utilities
Creates precision-recall curves and other evaluation visualizations
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_recall_curve, 
    average_precision_score,
    roc_curve,
    roc_auc_score,
    confusion_matrix
)
import os


def plot_precision_recall_curves(results, save_path='output/precision_recall_curves.png'):
    """
    Plot precision-recall curves for multiple models
    
    Args:
        results (dict): Dictionary containing model results with metrics
        save_path (str): Path to save the plot
    """
    plt.figure(figsize=(12, 8))
    
    colors = ['blue', 'green', 'red', 'orange', 'purple']
    
    for i, (name, result) in enumerate(results.items()):
        metrics = result['metrics']
        precision = metrics['precision']
        recall = metrics['recall']
        avg_precision = metrics['avg_precision']
        
        plt.plot(
            recall, 
            precision, 
            color=colors[i % len(colors)],
            lw=2,
            label=f'{metrics["model_name"]} (AP={avg_precision:.3f})'
        )
    
    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.title('Precision-Recall Curves - Fraud Detection Models', fontsize=16, fontweight='bold')
    plt.legend(loc='best', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nPrecision-Recall curves saved to {save_path}")
    plt.close()


def plot_roc_curves(results, save_path='output/roc_curves.png'):
    """
    Plot ROC curves for multiple models
    
    Args:
        results (dict): Dictionary containing model results with metrics
        save_path (str): Path to save the plot
    """
    plt.figure(figsize=(12, 8))
    
    colors = ['blue', 'green', 'red', 'orange', 'purple']
    
    for i, (name, result) in enumerate(results.items()):
        metrics = result['metrics']
        y_test = metrics['y_test']
        y_proba = metrics['y_proba']
        
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = metrics['roc_auc']
        
        plt.plot(
            fpr, 
            tpr, 
            color=colors[i % len(colors)],
            lw=2,
            label=f'{metrics["model_name"]} (AUC={roc_auc:.3f})'
        )
    
    # Plot diagonal line
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('ROC Curves - Fraud Detection Models', fontsize=16, fontweight='bold')
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"ROC curves saved to {save_path}")
    plt.close()


def plot_confusion_matrices(results, save_path='output/confusion_matrices.png'):
    """
    Plot confusion matrices for all models
    
    Args:
        results (dict): Dictionary containing model results with metrics
        save_path (str): Path to save the plot
    """
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
    
    if n_models == 1:
        axes = [axes]
    
    for idx, (name, result) in enumerate(results.items()):
        metrics = result['metrics']
        y_test = metrics['y_test']
        y_pred = metrics['y_pred']
        
        cm = confusion_matrix(y_test, y_pred)
        
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            ax=axes[idx],
            cbar=True,
            xticklabels=['Legitimate', 'Fraud'],
            yticklabels=['Legitimate', 'Fraud']
        )
        
        axes[idx].set_title(f'{metrics["model_name"]}', fontsize=14, fontweight='bold')
        axes[idx].set_ylabel('True Label', fontsize=12)
        axes[idx].set_xlabel('Predicted Label', fontsize=12)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrices saved to {save_path}")
    plt.close()


def plot_feature_importance(model, feature_names, top_n=20, save_path='output/feature_importance.png'):
    """
    Plot feature importance for tree-based models
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names (list): List of feature names
        top_n (int): Number of top features to display
        save_path (str): Path to save the plot
    """
    if not hasattr(model.model, 'feature_importances_'):
        print(f"Model {model.model_name} does not support feature importance")
        return
    
    importances = model.model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    
    plt.figure(figsize=(12, 8))
    plt.title(f'Top {top_n} Feature Importances - {model.model_name}', fontsize=16, fontweight='bold')
    plt.barh(range(top_n), importances[indices], color='steelblue')
    plt.yticks(range(top_n), [feature_names[i] for i in indices])
    plt.xlabel('Importance', fontsize=14)
    plt.ylabel('Features', fontsize=14)
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Feature importance plot saved to {save_path}")
    plt.close()


def generate_model_comparison_report(results, save_path='output/model_comparison.txt'):
    """
    Generate a text report comparing all models
    
    Args:
        results (dict): Dictionary containing model results with metrics
        save_path (str): Path to save the report
    """
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("FRAUD DETECTION MODEL COMPARISON REPORT")
    report_lines.append("="*80)
    report_lines.append("")
    
    # Collect metrics for all models
    comparison_data = []
    for name, result in results.items():
        metrics = result['metrics']
        comparison_data.append({
            'Model': metrics['model_name'],
            'Avg Precision': metrics['avg_precision'],
            'ROC AUC': metrics['roc_auc']
        })
    
    # Sort by average precision
    comparison_data.sort(key=lambda x: x['Avg Precision'], reverse=True)
    
    report_lines.append("Model Performance Ranking (by Average Precision):")
    report_lines.append("-"*80)
    report_lines.append(f"{'Rank':<6} {'Model':<25} {'Avg Precision':<20} {'ROC AUC':<15}")
    report_lines.append("-"*80)
    
    for rank, data in enumerate(comparison_data, 1):
        report_lines.append(
            f"{rank:<6} {data['Model']:<25} {data['Avg Precision']:<20.4f} {data['ROC AUC']:<15.4f}"
        )
    
    report_lines.append("")
    report_lines.append("="*80)
    report_lines.append("DETAILED MODEL METRICS")
    report_lines.append("="*80)
    
    for name, result in results.items():
        metrics = result['metrics']
        report_lines.append("")
        report_lines.append(f"Model: {metrics['model_name']}")
        report_lines.append("-"*80)
        report_lines.append(f"Average Precision Score: {metrics['avg_precision']:.4f}")
        report_lines.append(f"ROC AUC Score: {metrics['roc_auc']:.4f}")
        report_lines.append("")
    
    report_lines.append("="*80)
    report_lines.append("RECOMMENDATIONS")
    report_lines.append("="*80)
    report_lines.append("")
    
    best_model = comparison_data[0]
    report_lines.append(f"Best performing model: {best_model['Model']}")
    report_lines.append(f"  - Average Precision: {best_model['Avg Precision']:.4f}")
    report_lines.append(f"  - ROC AUC: {best_model['ROC AUC']:.4f}")
    report_lines.append("")
    report_lines.append("This model should be used for the production risk scoring pipeline.")
    report_lines.append("")
    
    # Write report to file
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"\nModel comparison report saved to {save_path}")
    
    # Also print to console
    print('\n'.join(report_lines))


if __name__ == '__main__':
    print("Model evaluation utilities loaded successfully")
