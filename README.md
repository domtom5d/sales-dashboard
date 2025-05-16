# Sales Conversion Analytics Dashboard

A Streamlit-powered sales conversion analytics dashboard that transforms raw Streak export data into actionable insights through advanced data processing and AI-enhanced visualization techniques.

## Features

- **Conversion Analysis**: Analyze conversion rates across different dimensions including booking types, referral sources, and time periods
- **Deal Value Insights**: Visualize revenue metrics and pricing patterns by different categories
- **AI-Powered Insights**: Generate intelligent recommendations for improving conversion rates, optimizing pricing, and targeting leads (requires API key)
- **Interactive Visualizations**: Explore data through interactive charts and graphs
- **Data Quality Analysis**: Identify and address data quality issues

## Setup

1. Clone this repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

### Setting Up AI Features

To enable the AI-powered insights features:

1. Add an environment variable named `AG_SECRET` with your API key
2. Refresh the app to see the AI-powered recommendations

## Data

The dashboard works with Streak CRM exports. You can either:
- Upload CSV files directly through the interface
- Import data into the SQLite database

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.