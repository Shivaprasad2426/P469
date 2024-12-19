import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# Function to load data
@st.cache
def load_data(file_path):
    return pd.read_csv(file_path, parse_dates=['Date'])

# Title and Header
st.title("Stock Market Dashboard")
st.header("Interactive Analysis and Predictions of Stock Data")

# Upload File
uploaded_file = st.file_uploader("Upload a CSV File (with 'Date' and price columns)", type=["csv"])

if uploaded_file:
    # Load and Display Data
    data = load_data(uploaded_file)

    # Convert 'Date' column to datetime (if not already)
    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date'])
        data.sort_values('Date', inplace=True)

    if st.checkbox("Show Raw Data"):
        st.subheader("Raw Data")
        st.write(data.head())

    # Date Filtering
    st.subheader("Filter by Date Range")
    if 'Date' in data.columns:
        start_date = st.date_input("Start Date", value=data['Date'].min().date())
        end_date = st.date_input("End Date", value=data['Date'].max().date())

        # Filter data by date range
        mask = (data['Date'] >= pd.Timestamp(start_date)) & (data['Date'] <= pd.Timestamp(end_date))
        filtered_data = data.loc[mask]
        st.write(f"Filtered Data ({len(filtered_data)} rows):")
        st.write(filtered_data)

    # Visualization Options
    st.subheader("Visualization Options")
    column_to_plot = st.selectbox("Select Column to Plot", data.columns)

    # Line Chart
    st.subheader("Line Chart")
    line_fig = px.line(filtered_data, x='Date', y=column_to_plot, title=f"{column_to_plot} Over Time")
    st.plotly_chart(line_fig)

    # Predictions
    st.subheader("Predict Future Stock Prices")
    if column_to_plot != 'Date' and pd.api.types.is_numeric_dtype(data[column_to_plot]):
        # Prepare data for prediction
        data['Days'] = (data['Date'] - data['Date'].min()).dt.days
        X = data[['Days']]
        y = data[column_to_plot]

        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train Model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        st.write(f"Model RMSE: {rmse:.2f}")

        # Future Predictions
        future_days = st.slider("Select Number of Days to Predict", min_value=1, max_value=365, value=30)
        future_dates = pd.date_range(data['Date'].max() + pd.Timedelta(days=1), periods=future_days)
        future_days_since_start = (future_dates - data['Date'].min()).days
        future_predictions = model.predict(future_days_since_start.values.reshape(-1, 1))

        # Plot Predictions
        future_data = pd.DataFrame({'Date': future_dates, f"Predicted {column_to_plot}": future_predictions})
        pred_fig = px.line(future_data, x='Date', y=f"Predicted {column_to_plot}", title="Future Predictions")
        st.plotly_chart(pred_fig)

    else:
        st.write("Please select a numeric column for predictions.")
else:
    st.write("Please upload a CSV file to get started.")
