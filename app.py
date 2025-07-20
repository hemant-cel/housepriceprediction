import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Page configuration
st.set_page_config(page_title="House Price Predictor", layout="centered")

st.title("🏠 House Price Prediction App")
st.write("Enter the house details below to predict the price.")

# Load dataset
df = pd.read_csv(r"C:\Users\heman\OneDrive\Desktop\housepriseprediction\Housing.csv")

# Select features and target
X = df[['area', 'bedrooms', 'bathrooms']]
y = df['price']

# Train the model
model = LinearRegression()
model.fit(X, y)

# Sidebar for input
st.sidebar.header("Enter House Features")
area = st.sidebar.number_input("Area (in sq ft)", min_value=500, max_value=10000, value=2000)
bedrooms = st.sidebar.slider("Number of Bedrooms", 1, 10, 3)
bathrooms = st.sidebar.slider("Number of Bathrooms", 1, 10, 2)

# Prediction
input_data = pd.DataFrame([[area, bedrooms, bathrooms]], columns=['area', 'bedrooms', 'bathrooms'])
predicted_price = model.predict(input_data)[0]

st.subheader("📈 Predicted House Price:")
st.success(f"₹ {predicted_price:,.2f}")

# Visualize actual vs predicted
st.subheader("🔍 Actual vs Predicted Price (on test set)")

# Splitting data for visualization
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_pred = model.predict(X_test)

fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, color='skyblue', edgecolors='black')
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
ax.set_xlabel('Actual Price')
ax.set_ylabel('Predicted Price')
ax.set_title('Actual vs Predicted House Prices')
st.pyplot(fig)
