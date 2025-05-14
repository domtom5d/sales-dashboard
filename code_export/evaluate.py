"""
evaluate.py - Model evaluation and analysis functions

This module contains functions for evaluating machine learning models,
particularly for lead scoring and conversion prediction.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, 
    f1_score, 
    precision_score, 
    recall_score,
    roc_curve, 
    auc, 
    precision_recall_curve,
)

def calculate_model_metrics(y_true, y_pred_proba):
    """
    Calculate comprehensive model metrics for binary classification.
    
    Args:
        y_true (array-like): True binary labels
        y_pred_proba (array-like): Predicted probabilities for the positive class
    
    Returns:
        dict: Dictionary of model metrics including:
            - roc_auc: Area under the ROC curve
            - pr_auc: Area under the Precision-Recall curve
            - fpr: False positive rates for ROC curve
            - tpr: True positive rates for ROC curve
            - precision: Precision values for PR curve
            - recall: Recall values for PR curve
            - best_threshold: Optimal probability threshold
            - j_threshold: Youden's J threshold
            - f1_threshold: F1-optimized threshold
            - confusion_matrix: Confusion matrix at optimal threshold
            - y_pred_proba: Predicted probabilities for all samples
            - y_true: True labels
            - won_scores: Score distribution for won leads
            - lost_scores: Score distribution for lost leads
    """
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Calculate Precision-Recall curve
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    # Find optimal threshold using Youden's J statistic (sensitivity + specificity - 1)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    j_threshold = thresholds[best_idx]
    
    # Calculate F1 scores for different thresholds to find optimal F1 score threshold
    f1_scores = []
    thresholds_to_test = np.linspace(0.01, 0.99, 99)
    
    for threshold in thresholds_to_test:
        y_pred = (y_pred_proba >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        f1_scores.append(f1)
    
    f1_df = pd.DataFrame({
        'threshold': thresholds_to_test,
        'f1_score': f1_scores
    })
    
    # Find threshold that maximizes F1 score
    max_f1_score = max(f1_scores)
    max_f1_idx = f1_scores.index(max_f1_score)
    f1_threshold = thresholds_to_test[max_f1_idx]
    
    # Use F1 threshold as the best threshold
    best_threshold = f1_threshold
    
    # Compute confusion matrix at best threshold
    y_pred = (y_pred_proba >= best_threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    
    # Separate scores for won and lost leads
    won_scores = y_pred_proba[y_true == 1]
    lost_scores = y_pred_proba[y_true == 0]
    
    # Return metrics
    return {
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds,
        'precision': precision,
        'recall': recall,
        'pr_thresholds': pr_thresholds,
        'best_threshold': best_threshold,
        'j_threshold': j_threshold,
        'f1_threshold': f1_threshold,
        'best_idx': best_idx,
        'max_f1_score': max_f1_score,
        'confusion_matrix': cm,
        'y_pred_proba': y_pred_proba,
        'y_true': y_true,
        'won_scores': won_scores,
        'lost_scores': lost_scores
    }

def plot_roc_curve(metrics, ax=None):
    """
    Plot ROC curve for model evaluation
    
    Args:
        metrics (dict): Dictionary of model metrics from calculate_model_metrics
        ax (matplotlib.axes.Axes, optional): Axes to plot on. If None, creates new figure.
    
    Returns:
        matplotlib.axes.Axes: The axes with the ROC curve plotted
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot ROC curve
    ax.plot(metrics['fpr'], metrics['tpr'], 'b-', label=f'ROC Curve (AUC = {metrics["roc_auc"]:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    
    # Plot the optimal threshold point
    best_idx = metrics['best_idx']
    ax.plot(metrics['fpr'][best_idx], metrics['tpr'][best_idx], 'ro', 
            label=f'Optimal threshold: {metrics["best_threshold"]:.3f}')
    
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    return ax

def plot_precision_recall_curve(metrics, ax=None):
    """
    Plot Precision-Recall curve for model evaluation
    
    Args:
        metrics (dict): Dictionary of model metrics from calculate_model_metrics
        ax (matplotlib.axes.Axes, optional): Axes to plot on. If None, creates new figure.
    
    Returns:
        matplotlib.axes.Axes: The axes with the PR curve plotted
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot Precision-Recall curve
    ax.plot(metrics['recall'], metrics['precision'], 'g-', 
            label=f'PR Curve (AUC = {metrics["pr_auc"]:.3f})')
    
    # Plot baseline
    no_skill = sum(metrics['y_true']) / len(metrics['y_true'])
    ax.plot([0, 1], [no_skill, no_skill], 'k--', label=f'Baseline ({no_skill:.3f})')
    
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    
    return ax

def plot_score_distributions(metrics, ax=None):
    """
    Plot score distributions for won and lost leads
    
    Args:
        metrics (dict): Dictionary of model metrics from calculate_model_metrics
        ax (matplotlib.axes.Axes, optional): Axes to plot on. If None, creates new figure.
    
    Returns:
        matplotlib.axes.Axes: The axes with the score distributions plotted
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    # Get the scores
    won_scores = metrics['won_scores']
    lost_scores = metrics['lost_scores']
    
    # Plot histograms
    bins = np.linspace(0, 1, 20)
    
    if len(won_scores) > 0:
        ax.hist(won_scores, bins=list(bins), alpha=0.5, color='green', label='Won Leads')
    if len(lost_scores) > 0:
        ax.hist(lost_scores, bins=list(bins), alpha=0.5, color='red', label='Lost Leads')
    
    # Plot thresholds
    best_threshold = metrics['best_threshold']
    
    # If we have F1 threshold, show it
    if 'f1_threshold' in metrics:
        f1_threshold = metrics['f1_threshold']
        ax.axvline(x=f1_threshold, color='blue', linestyle='--', 
                  label=f'Hot threshold (F1): {f1_threshold:.3f}')
        
        # Show Youden's J as comparison
        if 'j_threshold' in metrics:
            j_threshold = metrics['j_threshold']
            ax.axvline(x=j_threshold, color='purple', linestyle=':', 
                      label=f'Alt (Youden\'s J): {j_threshold:.3f}')
            
        # Calculate warm/cool thresholds based on F1
        second_threshold = f1_threshold / 2
        third_threshold = f1_threshold / 4
    else:
        # Fall back to previous approach
        ax.axvline(x=best_threshold, color='blue', linestyle='--', 
                  label=f'Hot threshold: {best_threshold:.3f}')
        
        second_threshold = best_threshold / 2
        third_threshold = best_threshold / 4
    
    # Add warm threshold
    ax.axvline(x=second_threshold, color='orange', linestyle='--', 
              label=f'Warm threshold: {second_threshold:.3f}')
    
    # Add cool threshold
    ax.axvline(x=third_threshold, color='green', linestyle='--', 
              label=f'Cool threshold: {third_threshold:.3f}')
              
    # Show F1 score if available
    if 'max_f1_score' in metrics:
        ax.text(0.05, 0.95, f"Best F1 Score: {metrics['max_f1_score']:.3f}", 
               transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xlabel('Model Score')
    ax.set_ylabel('Number of Leads')
    ax.set_title('Score Distribution: Won vs. Lost Leads')
    ax.legend()
    
    return ax

def get_custom_threshold_metrics(y_true, y_pred_proba, threshold):
    """
    Calculate metrics for a custom threshold value
    
    Args:
        y_true (array-like): True binary labels
        y_pred_proba (array-like): Predicted probabilities for the positive class
        threshold (float): Custom threshold for positive class
    
    Returns:
        dict: Dictionary of metrics at the custom threshold
    """
    y_pred = (y_pred_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    
    # Extract values from confusion matrix
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return {
        'confusion_matrix': cm,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'specificity': specificity,
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn
    }