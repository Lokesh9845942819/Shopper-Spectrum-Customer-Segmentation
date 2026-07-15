




import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Shopper Spectrum",
    page_icon="🛒",
    layout="wide"
)

st.title("🛒 Shopper Spectrum")
st.subheader("Customer Segmentation & Product Recommendation System")

st.write("""
This application predicts customer segments using the K-Means model
and recommends products based on purchase similarity.
""")

# -----------------------------
# Load Models
# -----------------------------
kmeans = joblib.load("../models/kmeans_model.pkl")
scaler = joblib.load("../models/scaler.pkl")

st.success("✅ Models Loaded Successfully")







# Load dataset
df = pd.read_csv("../dataset/online_retail.csv")

# Keep only valid rows
df = df.dropna(subset=["CustomerID", "Description"])
df = df[df["Quantity"] > 0]

# Create Customer-Product Matrix
customer_product = df.pivot_table(
    index="CustomerID",
    columns="Description",
    values="Quantity",
    aggfunc="sum",
    fill_value=0
)

# Calculate cosine similarity between products
product_similarity = cosine_similarity(customer_product.T)

similarity_df = pd.DataFrame(
    product_similarity,
    index=customer_product.columns,
    columns=customer_product.columns
)

# Recommendation function
def recommend_products(product_name, top_n=5):
    if product_name not in similarity_df.index:
        return []

    similar_products = similarity_df[product_name].sort_values(ascending=False)
    return similar_products.iloc[1:top_n+1].index.tolist()








st.header("Customer Details")

recency = st.number_input("Recency (Days)", min_value=0, value=30)
frequency = st.number_input("Frequency (Purchases)", min_value=1, value=5)
monetary = st.number_input("Monetary (Amount)", min_value=0.0, value=1000.0)


if st.button("Predict Customer Segment"):

    customer = [[recency, frequency, monetary]]

    customer_scaled = scaler.transform(customer)

    Prediction = int(kmeans.predict(customer_scaled)[0])

    cluster_info = {
    0: ("Regular Customer", "🟢 Buys occasionally with average spending."),
    1: ("Inactive Customer", "🔴 Hasn't purchased recently. Consider promotional offers."),
    2: ("High Value Customer", "⭐ Loyal customer with frequent purchases and high spending.")
    }

    name, desc = cluster_info[Prediction]

    st.success(f"Predicted Customer Segment: {name}")
    st.info(desc)
st.header("Product Recommendation")




products = sorted(similarity_df.index.tolist())




selected_product = st.selectbox(
    "Select a Product",
    products
)



if st.button("Recommend Products"):

    st.subheader("Recommended Products")
    st.success(f"Top 5 products similar to '{selected_product}'")

    recommendations = recommend_products(selected_product)

    for item in recommendations:
        if item != selected_product:
            st.write("✅", item)