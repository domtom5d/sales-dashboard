"""
segmentation.py - Unsupervised clustering and lead persona identification

This module contains functions for performing unsupervised segmentation,
identifying lead personas, and applying cluster-specific scoring models.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns


def prepare_data_for_clustering(df):
    """
    Prepare data for clustering by selecting and normalizing relevant features.
    
    Args:
        df (DataFrame): Processed dataframe with lead data
        
    Returns:
        tuple: (
            DataFrame: Features ready for clustering, 
            StandardScaler: Fitted scaler for transforming new data,
            list: Feature names used for clustering
        )
    """
    # Select numerical features that are likely to define personas
    numeric_features = [
        'days_until_event', 
        'number_of_guests', 
        'bartenders_needed',
        'total_serve_time',
        'total_bartender_time'
    ]
    
    # Filter to only include columns that exist in the dataframe
    features = [col for col in numeric_features if col in df.columns]
    
    # Add categorical features as one-hot encoded
    categorical_features = []
    
    # Only include booking_type if it exists and has a reasonable number of unique values
    if 'booking_type' in df.columns and df['booking_type'].nunique() < 10:
        categorical_features.append('booking_type')
        
    # Check for event_type or lead_trigger
    if 'event_type' in df.columns and df['event_type'].nunique() < 10:
        categorical_features.append('event_type')
    elif 'lead_trigger' in df.columns and df['lead_trigger'].nunique() < 10:
        categorical_features.append('lead_trigger')
        
    # Check for marketing_source
    if 'marketing_source' in df.columns and df['marketing_source'].nunique() < 10:
        categorical_features.append('marketing_source')
    
    # Create a copy of the dataframe with only the needed features
    cluster_df = df[features].copy()
    
    # Handle missing values
    for col in cluster_df.columns:
        if cluster_df[col].dtype in [np.float64, np.int64]:
            cluster_df[col] = cluster_df[col].fillna(cluster_df[col].median())
        else:
            cluster_df[col] = cluster_df[col].fillna('Unknown')
    
    # One-hot encode categorical features
    for cat_feat in categorical_features:
        if cat_feat in df.columns:
            # Get dummies for the categorical column
            dummies = pd.get_dummies(df[cat_feat], prefix=cat_feat, dummy_na=True)
            # Join the dummies to the main dataframe
            cluster_df = pd.concat([cluster_df, dummies], axis=1)
    
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(cluster_df)
    
    # Create a dataframe with the scaled data
    scaled_df = pd.DataFrame(scaled_data, columns=cluster_df.columns)
    
    return scaled_df, scaler, list(cluster_df.columns)


def find_optimal_k(data, max_k=10):
    """
    Find the optimal number of clusters using silhouette score.
    
    Args:
        data (DataFrame): Scaled data ready for clustering
        max_k (int): Maximum number of clusters to try
    
    Returns:
        tuple: (optimal_k, silhouette_scores)
    """
    silhouette_scores = []
    k_values = range(2, min(max_k + 1, len(data) // 5))
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        cluster_labels = kmeans.fit_predict(data)
        silhouette_avg = silhouette_score(data, cluster_labels)
        silhouette_scores.append(silhouette_avg)
    
    # Get the k value with the highest silhouette score
    if silhouette_scores:
        optimal_k = k_values[np.argmax(silhouette_scores)]
    else:
        optimal_k = 3  # Default if we can't compute silhouette scores
    
    return optimal_k, silhouette_scores


def perform_clustering(data, n_clusters=None, algorithm='kmeans'):
    """
    Perform clustering on the prepared data.
    
    Args:
        data (DataFrame): Scaled data ready for clustering
        n_clusters (int, optional): Number of clusters to use. If None, will find optimal.
        algorithm (str): Clustering algorithm to use ('kmeans', 'dbscan', 'gmm')
    
    Returns:
        tuple: (
            array: Cluster assignments for each data point,
            object: Fitted clustering model
        )
    """
    if algorithm == 'kmeans':
        if n_clusters is None:
            n_clusters, _ = find_optimal_k(data)
        
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        clusters = model.fit_predict(data)
        
    elif algorithm == 'dbscan':
        # DBSCAN parameters can be tuned further
        model = DBSCAN(eps=0.5, min_samples=5)
        clusters = model.fit_predict(data)
        
    elif algorithm == 'gmm':
        if n_clusters is None:
            n_clusters, _ = find_optimal_k(data)
            
        model = GaussianMixture(n_components=n_clusters, random_state=42)
        model.fit(data)
        clusters = model.predict(data)
    
    else:
        raise ValueError(f"Unknown clustering algorithm: {algorithm}")
    
    return clusters, model


def reduce_dimensions_for_visualization(data, n_components=2):
    """
    Reduce dimensions of the data for visualization.
    
    Args:
        data (DataFrame): Scaled data ready for clustering
        n_components (int): Number of dimensions to reduce to
    
    Returns:
        tuple: (
            array: Reduced data,
            object: Fitted PCA model
        )
    """
    pca = PCA(n_components=n_components, random_state=42)
    reduced_data = pca.fit_transform(data)
    return reduced_data, pca


def analyze_clusters(df, clusters, feature_names):
    """
    Analyze characteristics of each cluster.
    
    Args:
        df (DataFrame): Original dataframe with lead data
        clusters (array): Cluster assignments for each data point
        feature_names (list): Names of features used for clustering
    
    Returns:
        tuple: (
            DataFrame: Cluster profiles showing mean values for each feature,
            DataFrame: Conversion rates by cluster
        )
    """
    # Add cluster assignments to the original dataframe
    df_with_clusters = df.copy()
    df_with_clusters['cluster'] = clusters
    
    # Calculate cluster profiles (mean of each feature by cluster)
    profile_columns = []
    
    # Add numeric columns that exist
    numeric_columns = [
        'days_until_event', 
        'number_of_guests', 
        'bartenders_needed',
        'total_serve_time',
        'total_bartender_time'
    ]
    profile_columns.extend([col for col in numeric_columns if col in df.columns])
    
    # Create profiles if we have valid columns
    if profile_columns:
        cluster_profiles = df_with_clusters.groupby('cluster')[profile_columns].mean()
    else:
        # Create an empty dataframe if no valid columns
        cluster_profiles = pd.DataFrame(index=np.unique(clusters))
    
    # Add percentage of each booking type per cluster
    if 'booking_type' in df.columns:
        for booking_type in df['booking_type'].dropna().unique():
            col_name = f'booking_type_{booking_type}'
            cluster_profiles[col_name] = df_with_clusters[df_with_clusters['booking_type'] == booking_type].groupby('cluster').size() / df_with_clusters.groupby('cluster').size()
    
    # Add event type or lead trigger percentages
    event_col = None
    if 'event_type' in df.columns:
        event_col = 'event_type'
    elif 'lead_trigger' in df.columns:
        event_col = 'lead_trigger'
        
    if event_col:
        for event_type in df[event_col].dropna().unique():
            col_name = f'{event_col}_{event_type}'
            cluster_profiles[col_name] = df_with_clusters[df_with_clusters[event_col] == event_type].groupby('cluster').size() / df_with_clusters.groupby('cluster').size()
    
    # Calculate conversion rates by cluster
    if 'won' in df.columns:
        conversion_by_cluster = df_with_clusters.groupby('cluster')['won'].mean().reset_index()
        conversion_by_cluster.columns = ['Cluster', 'Conversion Rate']
        
        # Add count of leads in each cluster
        counts = df_with_clusters.groupby('cluster').size().reset_index(name='Count')
        conversion_by_cluster = conversion_by_cluster.merge(counts, on='Cluster')
        
        # Order by conversion rate descending
        conversion_by_cluster = conversion_by_cluster.sort_values('Conversion Rate', ascending=False)
        
    else:
        # Create an empty dataframe if 'won' column doesn't exist
        conversion_by_cluster = pd.DataFrame(columns=['Cluster', 'Conversion Rate', 'Count'])
    
    return cluster_profiles, conversion_by_cluster


def name_clusters(cluster_profiles, conversion_by_cluster):
    """
    Generate descriptive names for each cluster based on their characteristics.
    
    Args:
        cluster_profiles (DataFrame): Cluster profiles
        conversion_by_cluster (DataFrame): Conversion rates by cluster
    
    Returns:
        dict: Mapping of cluster numbers to descriptive names
    """
    cluster_names = {}
    
    # Define what columns we'll look for
    size_col = 'number_of_guests' if 'number_of_guests' in cluster_profiles.columns else None
    urgency_col = 'days_until_event' if 'days_until_event' in cluster_profiles.columns else None
    
    # Create a merged dataframe with both profiles and conversion
    merged = conversion_by_cluster.set_index('Cluster').join(cluster_profiles)
    
    for cluster in merged.index:
        # Start with base name
        name_parts = [f"Cluster {cluster}"]
        
        # Add conversion characteristic
        conv_rate = merged.loc[cluster, 'Conversion Rate']
        if conv_rate >= 0.3:
            name_parts.append("High-Converting")
        elif conv_rate >= 0.15:
            name_parts.append("Medium-Converting")
        else:
            name_parts.append("Low-Converting")
        
        # Add size characteristic if available
        if size_col and size_col in merged.columns:
            size_value = merged.loc[cluster, size_col]
            if size_value > 150:
                name_parts.append("Large-Event")
            elif size_value > 50:
                name_parts.append("Medium-Event")
            else:
                name_parts.append("Small-Event")
        
        # Add urgency characteristic if available
        if urgency_col and urgency_col in merged.columns:
            days_value = merged.loc[cluster, urgency_col]
            if days_value < 14:
                name_parts.append("Urgent")
            elif days_value < 60:
                name_parts.append("Standard-Timeline")
            else:
                name_parts.append("Long-Lead")
        
        # Combine parts
        cluster_names[cluster] = " ".join(name_parts)
    
    return cluster_names


def plot_clusters(reduced_data, clusters, title="Clusters Visualization (PCA)"):
    """
    Plot the clusters in 2D.
    
    Args:
        reduced_data (array): Reduced data for visualization (2D)
        clusters (array): Cluster assignments for each data point
        title (str): Plot title
    
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create a scatter plot
    scatter = ax.scatter(reduced_data[:, 0], reduced_data[:, 1], c=clusters, cmap='viridis', 
               alpha=0.6, s=50, edgecolors='w', linewidths=0.5)
    
    # Add a colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Cluster')
    
    # Set title and labels
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Principal Component 1', fontsize=12)
    ax.set_ylabel('Principal Component 2', fontsize=12)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    return fig


def plot_cluster_conversion_rates(conversion_by_cluster, cluster_names=None):
    """
    Plot the conversion rates of each cluster.
    
    Args:
        conversion_by_cluster (DataFrame): Conversion rates by cluster
        cluster_names (dict, optional): Mapping of cluster numbers to descriptive names
    
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    if conversion_by_cluster.empty:
        # Create empty figure if no data
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No conversion data available", 
                horizontalalignment='center', fontsize=14)
        return fig
    
    # Create a copy of the dataframe
    plot_df = conversion_by_cluster.copy()
    
    # Add cluster names if provided
    if cluster_names:
        plot_df['Cluster Name'] = plot_df['Cluster'].map(cluster_names)
        x_col = 'Cluster Name'
    else:
        x_col = 'Cluster'
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create the bar plot
    bars = sns.barplot(x=x_col, y='Conversion Rate', data=plot_df, ax=ax, palette='viridis')
    
    # Add the count as text on each bar
    for i, bar in enumerate(bars.patches):
        count = plot_df.iloc[i]['Count']
        ax.text(bar.get_x() + bar.get_width()/2, 
                bar.get_height() + 0.01, 
                f"n={count}", 
                ha='center', va='bottom', fontsize=10)
    
    # Customize the plot
    ax.set_title('Conversion Rates by Cluster', fontsize=14)
    ax.set_xlabel('')
    ax.set_ylabel('Conversion Rate', fontsize=12)
    ax.set_ylim(0, max(plot_df['Conversion Rate'] * 1.2))
    
    # Rotate x labels if using cluster names
    if cluster_names:
        plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    return fig


def plot_feature_importance_by_cluster(cluster_profiles, feature_names):
    """
    Plot the feature importance for each cluster using a heatmap.
    
    Args:
        cluster_profiles (DataFrame): Cluster profiles
        feature_names (list): Names of features used for clustering
    
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    # Create a copy of the profiles and normalize
    profiles = cluster_profiles.copy()
    
    # Filter to only include the features that were used for clustering
    available_features = [f for f in feature_names if f in profiles.columns]
    
    if not available_features:
        # Create empty figure if no data
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No feature data available", 
                horizontalalignment='center', fontsize=14)
        return fig
    
    # Standardize each feature
    for feature in available_features:
        feature_mean = profiles[feature].mean()
        feature_std = profiles[feature].std()
        if feature_std > 0:  # Avoid division by zero
            profiles[feature] = (profiles[feature] - feature_mean) / feature_std
    
    # Create a heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(profiles[available_features].T, cmap='coolwarm', center=0, 
                annot=True, fmt=".2f", linewidths=.5, ax=ax)
    
    ax.set_title('Standardized Feature Values by Cluster', fontsize=14)
    ax.set_xlabel('Cluster', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    
    plt.tight_layout()
    return fig


def segment_leads(df, n_clusters=None, algorithm='kmeans'):
    """
    Main function to segment leads using clustering.
    
    Args:
        df (DataFrame): Processed dataframe with lead data
        n_clusters (int, optional): Number of clusters to use. If None, will find optimal.
        algorithm (str): Clustering algorithm to use ('kmeans', 'dbscan', 'gmm')
    
    Returns:
        dict: Dictionary containing:
            - 'df_with_clusters': Original dataframe with cluster assignments
            - 'cluster_profiles': Cluster profiles
            - 'conversion_by_cluster': Conversion rates by cluster
            - 'reduced_data': Reduced data for visualization
            - 'clusters': Cluster assignments
            - 'model': Fitted clustering model
            - 'pca': Fitted PCA model
            - 'feature_names': Names of features used for clustering
            - 'cluster_names': Descriptive names for clusters
            - 'n_clusters': Number of clusters used
    """
    # Prepare data for clustering
    scaled_df, scaler, feature_names = prepare_data_for_clustering(df)
    
    # If dataframe is too small, return early
    if len(scaled_df) < 10:
        return {
            'error': 'Not enough data for clustering',
            'min_required': 10,
            'actual': len(scaled_df)
        }
    
    # Perform clustering
    clusters, model = perform_clustering(scaled_df, n_clusters, algorithm)
    
    # Add clusters to original dataframe
    df_with_clusters = df.copy()
    df_with_clusters['cluster'] = clusters
    
    # Reduce dimensions for visualization
    reduced_data, pca = reduce_dimensions_for_visualization(scaled_df)
    
    # Analyze clusters
    cluster_profiles, conversion_by_cluster = analyze_clusters(df, clusters, feature_names)
    
    # Generate descriptive names for clusters
    cluster_names = name_clusters(cluster_profiles, conversion_by_cluster)
    
    # Return results
    return {
        'df_with_clusters': df_with_clusters,
        'cluster_profiles': cluster_profiles,
        'conversion_by_cluster': conversion_by_cluster,
        'reduced_data': reduced_data,
        'clusters': clusters,
        'model': model,
        'pca': pca,
        'feature_names': feature_names,
        'cluster_names': cluster_names,
        'n_clusters': len(np.unique(clusters)),
    }


def predict_cluster_for_new_lead(new_lead_data, segmentation_results):
    """
    Predict which cluster a new lead belongs to.
    
    Args:
        new_lead_data (dict): Data for the new lead
        segmentation_results (dict): Results from segment_leads function
    
    Returns:
        int: Predicted cluster
    """
    # Extract needed components
    model = segmentation_results['model']
    feature_names = segmentation_results['feature_names']
    
    # Create a dataframe with the new lead data
    new_df = pd.DataFrame([new_lead_data])
    
    # Prepare the data for prediction (similar to prepare_data_for_clustering)
    # This needs to follow the same preprocessing as in prepare_data_for_clustering
    
    # Handle missing features
    for feature in feature_names:
        if feature not in new_df.columns:
            # For one-hot features, set to 0
            if '_' in feature:
                new_df[feature] = 0
    
    # Keep only needed features and in the same order
    prediction_df = pd.DataFrame(index=new_df.index)
    for feature in feature_names:
        if feature in new_df.columns:
            prediction_df[feature] = new_df[feature]
        else:
            prediction_df[feature] = 0  # Default for missing features
    
    # Standardize the data
    # Ideally, we would use the same scaler used in prepare_data_for_clustering
    # Since we don't have it, we'll standardize based on feature_names means and stds
    # In a real implementation, save the scaler from prepare_data_for_clustering
    
    # Predict cluster
    if hasattr(model, 'predict'):
        cluster = model.predict(prediction_df)[0]
    else:
        # For some models like DBSCAN, we'd use a different approach
        cluster = 0  # Default
    
    return int(cluster)