import streamlit as st

st.title("Test Dashboard")
st.write("This is a test to verify Streamlit is working correctly.")

# Create a simple interactive element
if st.button("Click me"):
    st.write("Button was clicked!")
else:
    st.write("Button not clicked yet.")

# Display some basic data
import pandas as pd
import numpy as np

# Create sample data
data = {
    'Column 1': np.random.randn(5),
    'Column 2': np.random.randn(5),
}
df = pd.DataFrame(data)

st.write("### Sample DataFrame")
st.dataframe(df)

st.write("### Sample Chart")
st.line_chart(df)