"""
lead_personas_tab.py - Lead Personas Tab Module

This module provides the implementation for the Lead Personas tab
in the Sales Conversion Analytics Dashboard.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def render_lead_personas_tab(df):
    """
    Render the Lead Personas tab with clustering analysis and visualizations
    
    Args:
        df (DataFrame): Processed dataframe with conversion data
    """
    st.markdown("## 🧠 Lead Personas")
    st.markdown("Automatically segment your leads into distinct groups based on their characteristics. This helps identify different types of customers and their unique behaviors.")
    
    if df is None or len(df) == 0:
        st.warning("No data available for analysis.")
        return
    
    # Add a Generate Personas button
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("### Generate Lead Personas")
        st.markdown("Click the button to identify distinct customer segments in your data.")
    
    with col2:
        generate_button = st.button(
            "Generate Personas", 
            help="Click to create lead personas using K-Means clustering on your data"
        )
    
    # Generate or retrieve personas
    if generate_button or 'personas_df' in st.session_state:
        # Show the clustering analysis
        create_lead_personas(df)
    else:
        st.info("Click the 'Generate Personas' button to identify customer segments in your data.")
        st.markdown("""
        ### What are Lead Personas?
        
        Lead personas are distinct groups of customers with similar characteristics. Identifying these segments helps you:
        
        - Tailor your sales approach to different customer types
        - Focus resources on the most promising segments
        - Understand the unique needs of different customer groups
        - Develop targeted marketing messages
        
        The clustering algorithm will analyze factors such as:
        - Event size (number of guests)
        - Price sensitivity
        - Lead time
        - Customer location
        - Other key attributes in your data
        """)

def create_lead_personas(df):
    """
    Create lead personas by clustering the data
    
    Args:
        df (DataFrame): The processed data with features
    """
    # Select only numeric columns for clustering
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Filter out outcome/target columns and ensure we have the right features
    exclude_cols = ['outcome', 'won', 'lost', 'box_key', 'latitude', 'longitude']
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    # Add minimum number of features for meaningful clustering
    required_features = ['number_of_guests', 'price_per_guest', 'days_until_event', 'days_since_inquiry']
    final_features = []
    
    for feature in required_features:
        if feature in df.columns:
            if df[feature].notna().sum() >= (len(df) * 0.5):  # Require at least 50% non-NA values
                final_features.append(feature)
    
    # Add any remaining useful numeric features
    additional_features = ['bartenders_needed', 'actual_deal_value', 'is_corporate']
    for feature in additional_features:
        if feature in df.columns and df[feature].notna().sum() >= (len(df) * 0.5):
            final_features.append(feature)
    
    if len(final_features) < 2:
        st.warning("Not enough features available for clustering. Need at least 2 numeric columns with sufficient data.")
        return
    
    # Create a copy with only the selected features
    cluster_data = df[final_features].copy()
    
    # Handle missing values
    cluster_data = cluster_data.fillna(cluster_data.median())
    
    # Scale the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(cluster_data)
    
    # Determine optimal number of clusters
    with st.expander("Cluster Analysis Details", expanded=False):
        st.markdown("### Choosing the Number of Clusters")
        
        # Calculate inertia for different numbers of clusters
        inertia = []
        k_range = range(2, min(8, len(df) // 10 + 1))  # Up to 8 clusters, but no more than 1/10 of data points
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
            kmeans.fit(scaled_data)
            inertia.append(kmeans.inertia_)
        
        # Plot the elbow curve
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(list(k_range), inertia, marker='o')
        ax.set_xlabel('Number of Clusters')
        ax.set_ylabel('Inertia (Within-Cluster Sum of Squares)')
        ax.set_title('Elbow Method for Optimal Number of Clusters')
        st.pyplot(fig)
        
        # Determine optimal k using the elbow method
        optimal_k = 3  # Default
        if len(inertia) > 1:
            # Calculate the rate of change in inertia
            inertia_changes = [inertia[i] - inertia[i+1] for i in range(len(inertia)-1)]
            
            # Find where the rate of change starts to flatten
            for i, change in enumerate(inertia_changes):
                if i > 0 and change < inertia_changes[0] * 0.3:  # If change is less than 30% of the first change
                    optimal_k = k_range[i]
                    break
    
    # Create a slider for selecting the number of clusters
    num_clusters = st.slider(
        "Number of Lead Personas", 
        min_value=2, 
        max_value=min(8, len(df) // 10 + 1), 
        value=optimal_k,
        help="Adjust the number of personas to identify in your data"
    )
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')
    cluster_labels = kmeans.fit_predict(scaled_data)
    
    # Add cluster labels back to the original data
    df_with_clusters = df.copy()
    df_with_clusters['lead_persona'] = cluster_labels
    
    # Save results to session state
    st.session_state.personas_df = df_with_clusters
    st.session_state.personas_kmeans = kmeans
    st.session_state.personas_features = final_features
    st.session_state.personas_scaler = scaler
    
    # Display cluster information
    show_cluster_profiles(df_with_clusters, final_features, kmeans, scaler)
    
    # Visualize the clusters
    visualize_clusters(df_with_clusters, scaled_data, final_features, cluster_labels, kmeans)

def show_cluster_profiles(df, feature_cols, kmeans, scaler):
    """
    Show the profile of each cluster
    
    Args:
        df (DataFrame): Dataframe with cluster labels
        feature_cols (list): List of feature columns used for clustering
        kmeans (KMeans): Fitted KMeans model
        scaler (StandardScaler): Fitted scaler
    """
    st.markdown("### Lead Persona Profiles")
    
    # Get cluster centers and convert back to original scale
    centers = scaler.inverse_transform(kmeans.cluster_centers_)
    
    # Create a dataframe of the cluster centers
    centers_df = pd.DataFrame(centers, columns=feature_cols)
    
    # Calculate additional metrics for each cluster
    cluster_stats = []
    
    for cluster_id in range(len(centers)):
        cluster_df = df[df['lead_persona'] == cluster_id]
        
        # Calculate metrics
        stats = {
            'Persona': f"Persona {cluster_id + 1}",
            'Count': len(cluster_df),
            'Percentage': len(cluster_df) / len(df) * 100,
            'Conversion Rate': cluster_df['outcome'].mean() * 100 if 'outcome' in cluster_df.columns else 0
        }
        
        # Add feature means for this cluster
        for i, feature in enumerate(feature_cols):
            stats[feature] = centers_df.iloc[cluster_id, i]
        
        cluster_stats.append(stats)
    
    # Convert to dataframe
    stats_df = pd.DataFrame(cluster_stats)
    
    # Determine the persona names based on characteristics
    persona_names = []
    for idx, row in stats_df.iterrows():
        name = determine_persona_name(row, feature_cols)
        persona_names.append(name)
    
    stats_df['Persona Name'] = persona_names
    
    # Define which columns to display and their order
    display_cols = ['Persona', 'Persona Name', 'Count', 'Percentage', 'Conversion Rate'] + feature_cols
    display_cols = [col for col in display_cols if col in stats_df.columns]
    
    # Format the dataframe for display
    formatted_df = stats_df[display_cols].copy()
    for col in ['Percentage', 'Conversion Rate']:
        if col in formatted_df.columns:
            formatted_df[col] = formatted_df[col].round(1).astype(str) + '%'
    
    for col in feature_cols:
        if col in formatted_df.columns:
            formatted_df[col] = formatted_df[col].round(1)
    
    st.table(formatted_df)
    
    # Show individual persona details
    for cluster_id in range(len(centers)):
        cluster_df = df[df['lead_persona'] == cluster_id]
        persona_name = persona_names[cluster_id]
        
        with st.expander(f"Persona {cluster_id + 1}: {persona_name} ({len(cluster_df)} leads)", expanded=False):
            # Conversion metrics
            if 'outcome' in cluster_df.columns:
                conversion_rate = cluster_df['outcome'].mean() * 100
                st.metric("Conversion Rate", f"{conversion_rate:.1f}%")
            
            # Key characteristics
            st.markdown("#### Key Characteristics")
            for feature in feature_cols:
                if feature in cluster_df.columns:
                    mean_value = cluster_df[feature].mean()
                    overall_mean = df[feature].mean()
                    difference = mean_value - overall_mean
                    
                    # Format based on the feature type
                    if feature in ['price_per_guest', 'actual_deal_value']:
                        formatted_value = f"${mean_value:.2f}"
                        formatted_diff = f"${difference:.2f} vs. avg"
                    elif feature in ['number_of_guests', 'bartenders_needed']:
                        formatted_value = f"{mean_value:.1f}"
                        formatted_diff = f"{difference:.1f} vs. avg"
                    elif feature in ['days_until_event', 'days_since_inquiry']:
                        formatted_value = f"{mean_value:.1f} days"
                        formatted_diff = f"{difference:.1f} days vs. avg"
                    else:
                        formatted_value = f"{mean_value:.2f}"
                        formatted_diff = f"{difference:.2f} vs. avg"
                    
                    st.metric(
                        feature.replace('_', ' ').title(), 
                        formatted_value,
                        formatted_diff,
                        delta_color="normal"
                    )
            
            # Categorical breakdowns
            categorical_cols = ['clean_booking_type', 'state', 'event_season', 'weekday']
            for col in categorical_cols:
                if col in cluster_df.columns and cluster_df[col].notna().sum() > 0:
                    st.markdown(f"#### {col.replace('_', ' ').title()} Distribution")
                    count_series = cluster_df[col].value_counts().head(5)
                    
                    # Calculate percentages
                    percentages = count_series / count_series.sum() * 100
                    
                    # Create a dataframe for display
                    cat_df = pd.DataFrame({
                        col.title(): count_series.index,
                        'Count': count_series.values,
                        'Percentage': percentages.values
                    })
                    cat_df['Percentage'] = cat_df['Percentage'].round(1).astype(str) + '%'
                    
                    st.table(cat_df)

def determine_persona_name(row, feature_cols):
    """
    Determine a descriptive name for a persona based on its characteristics
    
    Args:
        row (Series): Row from the cluster stats dataframe
        feature_cols (list): Feature columns used for clustering
        
    Returns:
        str: Descriptive name for the persona
    """
    attributes = []
    
    # Check for high/low values in key dimensions
    if 'number_of_guests' in feature_cols:
        guests = row['number_of_guests']
        if guests > 150:
            attributes.append("Large Event")
        elif guests < 50:
            attributes.append("Small Event")
    
    if 'price_per_guest' in feature_cols:
        price = row['price_per_guest']
        if price > 100:
            attributes.append("Premium")
        elif price < 50:
            attributes.append("Budget")
    
    if 'days_until_event' in feature_cols:
        days = row['days_until_event']
        if days > 180:
            attributes.append("Long Lead")
        elif days < 30:
            attributes.append("Short Notice")
    
    if 'is_corporate' in feature_cols and row['is_corporate'] > 0.5:
        attributes.append("Corporate")
    
    # Generate name based on conversion rate
    if 'Conversion Rate' in row:
        conv_rate = float(row['Conversion Rate'].replace('%', ''))
        if conv_rate > 75:
            attributes.append("High-Converting")
        elif conv_rate < 25:
            attributes.append("Low-Converting")
    
    # Combine attributes or use default
    if attributes:
        return " / ".join(attributes)
    else:
        return f"Balanced Persona"

def visualize_clusters(df, scaled_data, feature_cols, cluster_labels, kmeans):
    """
    Visualize the clusters using PCA or feature plots
    
    Args:
        df (DataFrame): Dataframe with cluster labels
        scaled_data (array): Scaled feature data
        feature_cols (list): Feature columns used for clustering
        cluster_labels (array): Cluster assignments
        kmeans (KMeans): Fitted KMeans model
    """
    st.markdown("### Cluster Visualization")
    
    # Create tabs for different visualizations
    viz_tab1, viz_tab2 = st.tabs(["PCA Visualization", "Feature Comparisons"])
    
    with viz_tab1:
        # Use PCA to reduce dimensions for visualization
        if len(feature_cols) >= 2:
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(scaled_data)
            
            # Add results to a DataFrame for plotting
            pca_df = pd.DataFrame({
                'PCA1': pca_result[:, 0],
                'PCA2': pca_result[:, 1],
                'Cluster': cluster_labels
            })
            
            # Get cluster centers in PCA space
            centers_pca = pca.transform(kmeans.cluster_centers_)
            
            # Plot
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot each cluster with a different color
            for cluster_id in range(kmeans.n_clusters):
                cluster_points = pca_df[pca_df['Cluster'] == cluster_id]
                ax.scatter(
                    cluster_points['PCA1'], 
                    cluster_points['PCA2'],
                    alpha=0.7,
                    label=f'Persona {cluster_id + 1}'
                )
            
            # Plot cluster centers
            ax.scatter(
                centers_pca[:, 0], 
                centers_pca[:, 1], 
                s=100, 
                c='black', 
                marker='X', 
                label='Persona Centers'
            )
            
            # Label axes and add legend
            ax.set_xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
            ax.set_ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
            ax.set_title('Lead Personas - PCA Visualization')
            ax.legend()
            
            st.pyplot(fig)
            
            # Explain PCA components
            st.markdown("#### PCA Component Interpretation")
            
            # Get feature importances for the PCA components
            loadings = pd.DataFrame(
                data=pca.components_.T, 
                index=feature_cols
            )
            loadings.columns = ['PC1', 'PC2']
            
            # Get the top features for each component
            pc1_importance = pd.Series({col: abs(loadings.loc[col, 'PC1']) for col in feature_cols})
            pc2_importance = pd.Series({col: abs(loadings.loc[col, 'PC2']) for col in feature_cols})
            
            # Sort by importance
            pc1_loadings = pc1_importance.sort_values(ascending=False)
            pc2_loadings = pc2_importance.sort_values(ascending=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Principal Component 1 - Top Features:**")
                for feature, value in pc1_loadings.head(3).items():
                    direction = "+" if loadings.loc[feature, 'PC1'] > 0 else "-"
                    feature_str = feature if isinstance(feature, str) else str(feature)
                    st.markdown(f"- {feature_str.replace('_', ' ').title()}: {direction} ({value:.3f})")
            
            with col2:
                st.markdown("**Principal Component 2 - Top Features:**")
                for feature, value in pc2_loadings.head(3).items():
                    direction = "+" if loadings.loc[feature, 'PC2'] > 0 else "-"
                    feature_str = feature if isinstance(feature, str) else str(feature)
                    st.markdown(f"- {feature_str.replace('_', ' ').title()}: {direction} ({value:.3f})")
    
    with viz_tab2:
        if len(feature_cols) >= 2:
            # Allow user to select features to compare
            if len(feature_cols) > 2:
                col1, col2 = st.columns(2)
                with col1:
                    x_feature = st.selectbox("X-axis feature", options=feature_cols, index=0)
                with col2:
                    remaining_features = [f for f in feature_cols if f != x_feature]
                    y_feature = st.selectbox("Y-axis feature", options=remaining_features, index=0)
            else:
                x_feature, y_feature = feature_cols[0], feature_cols[1]
            
            # Create a scatter plot
            fig, ax = plt.subplots(figsize=(10, 6))
            
            for cluster_id in range(kmeans.n_clusters):
                cluster_points = df[df['lead_persona'] == cluster_id]
                ax.scatter(
                    cluster_points[x_feature], 
                    cluster_points[y_feature],
                    alpha=0.7,
                    label=f'Persona {cluster_id + 1}'
                )
            
            # Get the cluster centers and convert back to original scale
            centers = kmeans.cluster_centers_
            
            # Get the scaler that was used to scale the data
            if 'personas_scaler' in st.session_state:
                current_scaler = st.session_state.personas_scaler
            else:
                # Create a new scaler as fallback (should not happen in normal flow)
                current_scaler = StandardScaler()
            centers_orig = current_scaler.inverse_transform(centers)
            
            # Get the index positions of the selected features
            x_idx = feature_cols.index(x_feature)
            y_idx = feature_cols.index(y_feature)
            
            # Plot the cluster centers
            ax.scatter(
                centers_orig[:, x_idx], 
                centers_orig[:, y_idx], 
                s=100, 
                c='black', 
                marker='X', 
                label='Persona Centers'
            )
            
            # Set labels
            ax.set_xlabel(x_feature.replace('_', ' ').title())
            ax.set_ylabel(y_feature.replace('_', ' ').title())
            ax.set_title(f'{x_feature.title()} vs {y_feature.title()} by Lead Persona')
            ax.legend()
            
            st.pyplot(fig)
            
            # Show correlation of features with conversion
            if 'outcome' in df.columns:
                st.markdown("#### Feature Correlation with Conversion Rate")
                
                # Calculate correlation of features with outcome
                corr_with_outcome = df[feature_cols + ['outcome']].corr()['outcome'].drop('outcome')
                
                # Sort and plot
                corr_sorted = corr_with_outcome.sort_values(ascending=False)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                corr_sorted.plot(kind='barh', ax=ax)
                ax.set_xlabel('Correlation with Conversion')
                ax.set_title('Features Correlation with Lead Conversion')
                
                st.pyplot(fig)