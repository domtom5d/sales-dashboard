# Sales Conversion Analytics Dashboard

## Overview
This is a Streamlit-powered sales conversion analytics dashboard that transforms raw Streak export data into actionable insights. The dashboard analyzes sales conversion patterns, visualizes key metrics, and provides AI-enhanced insights.

## Key Components

### Main App (app.py)
- Entry point for the Streamlit application
- Handles data loading and normalization
- Defines the multi-tab interface (Conversion Analysis, Lead Scoring, etc.)
- Implements the main dashboard UI components
- Currently displays all available data without filtering (filters have been removed)

### Conversion Analysis (conversion_analysis.py)
- Contains functions for analyzing conversion data
- Implements data visualization for conversion metrics
- Provides trend analysis and category breakdown
- Defines data processing and normalization functions

### Database Interface (database.py)
- Defines database schema using SQLAlchemy
- Implements Lead and Operation models
- Provides data import and retrieval functions
- Handles phone matching and other data processing operations

## Current Implementation
- The application loads all data without filtering, displaying the complete dataset
- Filter controls have been removed from the UI to simplify the dashboard
- Key performance indicators are displayed in a row of metrics cards
- Conversion trend charts show patterns over time
- Category breakdowns highlight the best and worst performing segments

## Technology Stack
- Streamlit web framework
- Python data analysis libraries (pandas, numpy)
- Data visualization (matplotlib, seaborn, plotly)
- SQLite database with SQLAlchemy ORM
- Machine learning for lead scoring

## Data Flow
1. Data is loaded from CSV files or database
2. Processing and normalization is applied
3. Complete dataset is displayed in the dashboard
4. Visualizations are generated to show key insights
5. AI-powered insights are available in dedicated tabs