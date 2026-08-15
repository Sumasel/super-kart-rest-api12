import streamlit as st
import requests
import json
import pandas as pd

st.set_page_config(layout="wide")

st.title("SuperKart Sales Forecasting App")
st.write("Enter product and store details to predict sales revenue.")

# Input fields for product details
st.header("Product Details")

product_id = st.text_input("Product ID", "FDW03")
product_weight = st.number_input("Product Weight", min_value=0.1, max_value=25.0, value=10.0, step=0.1)
product_sugar_content = st.selectbox("Product Sugar Content", ['Low Sugar', 'Regular', 'No Sugar'])
product_allocated_area = st.number_input("Product Allocated Area", min_value=0.0, max_value=0.3, value=0.05, step=0.001)
product_type = st.selectbox("Product Type", ['Dairy', 'Soft Drinks', 'Meat', 'Fruits and Vegetables', 'Household', 'Baking Goods', 'Snack Foods', 'Frozen Foods', 'Breakfast', 'Health and Hygiene', 'Hard Drinks', 'Canned', 'Breads', 'Starchy Foods', 'Others', 'Seafood'])
product_mrp = st.number_input("Product MRP", min_value=10.0, max_value=300.0, value=150.0, step=0.1)

# Input fields for store details
st.header("Store Details")

store_id = st.selectbox("Store ID", ['OUT001', 'OUT002', 'OUT003', 'OUT004'])
store_establishment_year = st.number_input("Store Establishment Year", min_value=1985, max_value=2024, value=2000, step=1)
store_size = st.selectbox("Store Size", ['Small', 'Medium', 'High'])
store_location_city_type = st.selectbox("Store Location City Type", ['Tier 1', 'Tier 2', 'Tier 3'])
store_type = st.selectbox("Store Type", ['Supermarket Type1', 'Supermarket Type2', 'Departmental Store', 'Food Mart'])

# Prepare the data as a dictionary (matching the structure expected by the Flask API)
# This is moved outside the if block so it's always defined.
input_data = {
    "Product_Id": [product_id],
    "Product_Weight": [product_weight],
    "Product_Sugar_Content": [product_sugar_content],
    "Product_Allocated_Area": [product_allocated_area],
    "Product_Type": [product_type],
    "Product_MRP": [product_mrp],
    "Store_Id": [store_id],
    "Store_Establishment_Year": [store_establishment_year],
    "Store_Size": [store_size],
    "Store_Location_City_Type": [store_location_city_type],
    "Store_Type": [store_type]
}

# Prediction button
if st.button("Predict Sales"):
    # Convert to JSON
    json_data = json.dumps(input_data)

  
   
    # IMPORTANT: Ensure the URL ends with '/predict'
    backend_url = "http://backend:7860/predict" # <--- CORRECTED BACKEND SPACE URL

    try:
        response = requests.post(backend_url, data=json_data, headers={'Content-Type': 'application/json'})
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
        prediction = response.json().get("prediction")

        if prediction:
            st.success(f"Predicted Sales Revenue: ${prediction[0]:,.2f}")
        else:
            st.error("Prediction failed. Please check the input and try again.")

    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to the backend API: {e}")
        st.error("Please ensure the backend URL is correct and the backend space is running.")
    except json.JSONDecodeError:
        st.error("Failed to decode JSON response from the backend. Please check backend logs.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

# Optional: Display the raw input data for debugging
st.sidebar.subheader("Raw Input Data")
st.sidebar.json(input_data)


# Batch Prediction
st.subheader("Batch Prediction")

uploaded_file = st.file_uploader(
    "Upload a CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    if st.button("Predict for Batch", type="primary"):

        response = requests.post(
            f"http://backend:7860/predictbatch",
            files={"file": uploaded_file}
        )

        if response.status_code == 200:
            results = response.json()

            st.success("Predictions completed successfully!")

            try:
                if isinstance(results, list):
                    df = pd.DataFrame(results)
                elif isinstance(results, dict):
                    # Check if all values are scalars
                    if all(not isinstance(v, (list, dict)) for v in results.values()):
                        df = pd.DataFrame([results])
                    else:
                        df = pd.DataFrame(results)
                else:
                    df = pd.DataFrame({"Result": [results]})

                st.dataframe(df, use_container_width=True)

            except Exception as e:
                st.error(f"Unable to display results as a table: {e}")
                st.json(results)

        else:
            st.error("Unable to connect to the prediction API.")





