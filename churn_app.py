import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Set page configuration to wide
st.set_page_config(page_title="Little Rider Churn Prediction App", page_icon="🚕", layout="wide")

# Streamlit app title
st.title("Little 🚕 Rider Churn Prediction App")
st.write("Please ensure the uploaded Excel file contains the required columns: rider_, last_seen_dated, registered_on, Trips_, DriverCancellations_, RiderCancellations_, Timeouts_, Total_requests, no_drivers_found, FulfillmentRate.")

# Create tabs for file upload and manual input
tab1, tab2 = st.tabs(["Upload Excel File", "Manual Input"])

# Tab 1: File Upload
with tab1:
    st.write("Upload an Excel file with rider data to predict churn probabilities.")
    uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx"], key="file_uploader")

    if uploaded_file is not None:
        try:
            # Load the data
            data = pd.read_excel(uploaded_file)
            st.write("Uploaded Data Preview:")
            st.dataframe(data.head())

            # Load the saved preprocessing objects
            to_drop = joblib.load('C:/Users/philip.otieno/Desktop/ML - AI - Projects/dropped_columns.pkl')
            poly = joblib.load('C:/Users/philip.otieno/Desktop/ML - AI - Projects/poly_features.pkl')
            selector = joblib.load('C:/Users/philip.otieno/Desktop/ML - AI - Projects/selector.pkl')
            imputer = joblib.load('C:/Users/philip.otieno/Desktop/ML - AI - Projects/imputer.pkl')
            scaler = joblib.load('C:/Users/philip.otieno/Desktop/ML - AI - Projects/scaler.pkl')

            # Determine which model to load (prefer LightGBM if available)
            try:
                model = joblib.load('C:/Users/philip.otieno/Desktop/ML - AI - Projects/churn_model_lgbm.pkl')
                st.write("Using LightGBM model for predictions.")
            except FileNotFoundError:
                model = joblib.load('C:/Users/philip.otieno/Desktop/ML - AI - Projects/churn_model_rf.pkl')
                st.write("Using Random Forest model for predictions.")

            # Apply the same feature engineering as during training
            data['last_seen_dated'] = pd.to_datetime(data['last_seen_dated'])
            data['registered_on'] = pd.to_datetime(data['registered_on'])
            data['tenure_days'] = (data['last_seen_dated'] - data['registered_on']).dt.days
            data['has_trips'] = (data['Trips_'] > 0).astype(int)
            data['Cancellations_per_Trip'] = (data['DriverCancellations_'] + data['RiderCancellations_']) / data['Trips_'].replace(0, np.nan)
            data['Cancellations_per_Trip'] = data['Cancellations_per_Trip'].fillna(0)
            data['Timeouts_per_Request'] = data['Timeouts_'] / data['Total_requests'].replace(0, np.nan)
            data['Timeouts_per_Request'] = data['Timeouts_per_Request'].fillna(0)
            data['NoDrivers_per_Request'] = data['no_drivers_found'] / data['Total_requests'].replace(0, np.nan)
            data['NoDrivers_per_Request'] = data['NoDrivers_per_Request'].fillna(0)
            data['Trip_Frequency'] = data['Total_requests'] / (data['tenure_days'].replace(0, np.nan) / 30)
            data['Trip_Frequency'] = data['Trip_Frequency'].fillna(0)
            data.replace([np.inf, -np.inf], np.nan, inplace=True)

            # Select the initial features
            initial_features = [
                'no_drivers_found', 'Timeouts_', 'FulfillmentRate', 'DriverCancellations_',
                'has_trips', 'Timeouts_per_Request', 'tenure_days', 'Trip_Frequency'
            ]
            X = data[initial_features].copy()

            # Ensure no infinite values
            X.replace([np.inf, -np.inf], np.nan, inplace=True)
            X = X.fillna(X.median())

            # Drop highly correlated features
            X_reduced = X.drop(columns=to_drop) if to_drop else X

            # Apply polynomial features
            X_poly = poly.transform(X_reduced)

            # Apply feature selection
            X_selected = selector.transform(X_poly)

            # Impute and scale
            X_imputed = imputer.transform(X_selected)
            X_scaled = scaler.transform(X_imputed)

            # Predict churn probabilities
            churn_probabilities = model.predict_proba(X_scaled)[:, 1]

            # Add predictions to the data
            data['churn_probability'] = churn_probabilities

            # Display the results
            st.write("Predictions (Top 5):")
            st.dataframe(data[['rider_', 'churn_probability']].head())

            # Save the results to a new Excel file
            output_file = "churn_predictions.xlsx"
            data.to_excel(output_file, index=False)

            # Provide a download link
            with open(output_file, "rb") as file:
                st.download_button(
                    label="Download Predictions as Excel",
                    data=file,
                    file_name=output_file,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.write("Please ensure the uploaded Excel file contains the required columns: rider_, last_seen_dated, registered_on, Trips_, DriverCancellations_, RiderCancellations_, Timeouts_, Total_requests, no_drivers_found, FulfillmentRate.")

# Tab 2: Manual Input
with tab2:
    st.write("Enter rider data manually to predict churn probability.")

    # Create a form for manual input
    with st.form(key="manual_input_form"):
        rider_id = st.text_input("Rider ID (e.g., 254729773325)", value="123456789")
        total_requests = st.number_input("Total Requests", min_value=0, value=100, step=1)
        trips = st.number_input("Trips", min_value=0, value=100, step=1)
        driver_cancellations = st.number_input("Driver Cancellations", min_value=0, value=0, step=1)
        rider_cancellations = st.number_input("Rider Cancellations", min_value=0, value=0, step=1)
        timeouts = st.number_input("Timeouts", min_value=0, value=0, step=1)
        no_drivers_found = st.number_input("No Drivers Found", min_value=0, value=0, step=1)
        fulfillment_rate = st.number_input("Fulfillment Rate (%)", min_value=0.0, max_value=100.0, value=100.0, step=0.1)
        last_seen_dated = st.date_input("Last Seen Date", value=datetime(2025, 2, 23))
        registered_on = st.date_input("Registered On Date", value=datetime(2024, 6, 23))

        submit_button = st.form_submit_button(label="Predict Churn Probability")

    if submit_button:
        try:
            # Load the saved preprocessing objects
            to_drop = joblib.load('C:/Users/philip.otieno/Desktop/ML - AI - Projects/dropped_columns.pkl')
            poly = joblib.load('C:/Users/philip.otieno/Desktop/ML - AI - Projects/poly_features.pkl')
            selector = joblib.load('C:/Users/philip.otieno/Desktop/ML - AI - Projects/selector.pkl')
            imputer = joblib.load('C:/Users/philip.otieno/Desktop/ML - AI - Projects/imputer.pkl')
            scaler = joblib.load('C:/Users/philip.otieno/Desktop/ML - AI - Projects/scaler.pkl')

            # Determine which model to load (prefer LightGBM if available)
            try:
                model = joblib.load('C:/Users/philip.otieno/Desktop/ML - AI - Projects/churn_model_lgbm.pkl')
                st.write("Using LightGBM model for predictions.")
            except FileNotFoundError:
                model = joblib.load('C:/Users/philip.otieno/Desktop/ML - AI - Projects/churn_model_rf.pkl')
                st.write("Using Random Forest model for predictions.")

            # Create a DataFrame with the manual input
            data = pd.DataFrame({
                'rider_': [rider_id],
                'last_seen_dated': [pd.to_datetime(last_seen_dated)],
                'registered_on': [pd.to_datetime(registered_on)],
                'Total_requests': [total_requests],
                'Trips_': [trips],
                'DriverCancellations_': [driver_cancellations],
                'RiderCancellations_': [rider_cancellations],
                'Timeouts_': [timeouts],
                'no_drivers_found': [no_drivers_found],
                'FulfillmentRate': [fulfillment_rate]
            })

            # Apply the same feature engineering as during training
            data['tenure_days'] = (data['last_seen_dated'] - data['registered_on']).dt.days
            data['has_trips'] = (data['Trips_'] > 0).astype(int)
            data['Cancellations_per_Trip'] = (data['DriverCancellations_'] + data['RiderCancellations_']) / data['Trips_'].replace(0, np.nan)
            data['Cancellations_per_Trip'] = data['Cancellations_per_Trip'].fillna(0)
            data['Timeouts_per_Request'] = data['Timeouts_'] / data['Total_requests'].replace(0, np.nan)
            data['Timeouts_per_Request'] = data['Timeouts_per_Request'].fillna(0)
            data['NoDrivers_per_Request'] = data['no_drivers_found'] / data['Total_requests'].replace(0, np.nan)
            data['NoDrivers_per_Request'] = data['NoDrivers_per_Request'].fillna(0)
            data['Trip_Frequency'] = data['Total_requests'] / (data['tenure_days'].replace(0, np.nan) / 30)
            data['Trip_Frequency'] = data['Trip_Frequency'].fillna(0)
            data.replace([np.inf, -np.inf], np.nan, inplace=True)

            # Select the initial features
            initial_features = [
                'no_drivers_found', 'Timeouts_', 'FulfillmentRate', 'DriverCancellations_',
                'has_trips', 'Timeouts_per_Request', 'tenure_days', 'Trip_Frequency'
            ]
            X = data[initial_features].copy()

            # Ensure no infinite values
            X.replace([np.inf, -np.inf], np.nan, inplace=True)
            X = X.fillna(X.median())

            # Drop highly correlated features
            X_reduced = X.drop(columns=to_drop) if to_drop else X

            # Apply polynomial features
            X_poly = poly.transform(X_reduced)

            # Apply feature selection
            X_selected = selector.transform(X_poly)

            # Impute and scale
            X_imputed = imputer.transform(X_selected)
            X_scaled = scaler.transform(X_imputed)

            # Predict churn probability
            churn_probability = model.predict_proba(X_scaled)[0, 1]

            # Display the result
            st.write(f"**Churn Probability for Rider {rider_id}:** {churn_probability:.2%}")
            st.write("**Interpretation:**")
            if churn_probability > 0.8:
                st.write("This rider is at **high risk** of churning. Consider taking immediate retention actions, such as offering incentives or improving their experience.")
            elif churn_probability > 0.5:
                st.write("This rider is at **moderate risk** of churning. Monitor their activity and consider proactive engagement to prevent churn.")
            else:
                st.write("This rider is at **low risk** of churning. Continue providing a good experience to maintain their loyalty.")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.write("Please ensure all inputs are valid and the model files are available.")