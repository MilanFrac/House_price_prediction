import pandas as pd
import glob
import os
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
import numpy as np

st.set_page_config(page_title="Apartment Prices in Poland", layout="wide")
st.title("Interactive Analysis of the Polish Apartment Market")

folder_path = "C:\archive"
file_pattern = os.path.join(folder_path, 'apartments_pl_*.csv')

df_list = []
for file in glob.glob(file_pattern):
    df_temp = pd.read_csv(file)
    df_temp["source_file"] = os.path.basename(file)
    df_list.append(df_temp)

df = pd.concat(df_list, ignore_index=True)


df['year'] = df['source_file'].str.extract(r'_(\d{4})_')[0]
df['month'] = df['source_file'].str.extract(r'_(\d{4})_(\d{2})')[1]
df['year_month'] = df['year'] + '-' + df['month']


st.sidebar.header("Filters")

if 'city' not in df.columns:
    st.error("Missing 'city' column in the data.")
    st.stop()

all_cities = sorted(df['city'].dropna().unique())
selected_cities = st.sidebar.multiselect("Cities", all_cities, default=all_cities[:5])

years = sorted(df['year'].dropna().unique())
selected_years = st.sidebar.multiselect("Year", years, default=years)

selected_rooms = None
if 'rooms' in df.columns:
    available_rooms = sorted(df['rooms'].dropna().unique())
    selected_rooms = st.sidebar.multiselect("Numbers of Rooms", available_rooms)

selected_area = None
if 'area' in df.columns:
    min_area, max_area = int(df['price'].min()), int(df['price'].max())
    selected_area = st.sidebar.slider("Area (m^2)", min_area, max_area, (min_area, max_area))

selected_price = None
if 'price' in df.columns:
    min_price, max_price = int(df['price'].min()), int(df["price"].max())
    selected_price = st.sidebar.slider("Price (PLN)",min_price, max_price, (min_price, max_price))

filtered_df = df[
    df['city'].isin(selected_cities) &
    df['year'].isin(selected_years)
]    

if selected_rooms:
    filtered_df = filtered_df[filtered_df['rooms'].isin(selected_rooms)]
if selected_area:
    filtered_df = filtered_df[(filtered_df['area'] >= selected_area[0]) & (filtered_df['area'] <= selected_area[1])]
if selected_price:
    filtered_df = filtered_df[(filtered_df['price'] >= selected_price[0]) & (filtered_df['price'] <= selected_price[1])]

if 'area' in filtered_df.columns and 'price' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['area'] > 0]
    filtered_df['price_per_m2'] = filtered_df['price'] / filtered_df['area']

#predykcja
st.subheader("Predict Apartment Price")

features = ['squareMeters', 'rooms', 'floor', 'floorCount', 'buildYear',
    'centreDistance', 'poiCount', 'schoolDistance', 'clinicDistance',
    'postOfficeDistance', 'kindergartenDistance', 'restaurantDistance',
    'collegeDistance', 'pharmacyDistance']

model_df = filtered_df.dropna(subset = features + ['price'])

if not model_df.empty:
    X = model_df[features]
    y = model_df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.markdown(f"**Model Accuracy (R^2):** `{r2_score(y_test, y_pred):.2f}`")

    with st.expander("Enter Apartment Features", expanded=False):
        col1, col2 = st.columns(2)
        input_data = {}
        with col1:
            for feature in features[:len(features)//2]:
                min_val = int(df[feature].min())
                max_val = int(df[feature].max())
                default_val = int(df[feature].mean())
                input_data[feature] = st.number_input(f"{feature.replace('_', ' ').title()}:", min_value=min_val, max_value=max_val, value=default_val)
        with col2:
            for feature in features[len(features)//2:]:
                min_val = int(df[feature].min())
                max_val = int(df[feature].max())
                default_val = int(df[feature].mean())
                input_data[feature] = st.number_input(f"{feature.replace('_', ' ').title()}:", min_value=min_val, max_value=max_val, value=default_val)

        if st.button("Predict Price"):
            input_df = pd.DataFrame([input_data])
            predicted_price = model.predict(input_df)[0]
            st.success(f" Estimated Apartament Price: {predicted_price:,.0f} PLN")

        
st.markdown("### Feature Importance")
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
sorted_features = [features[i] for i in indices]
sorted_importances = importances[indices]
fig_imp, ax = plt.subplots(figsize=(10,5))
sns.barplot(x=sorted_importances, y=sorted_features, ax=ax)
ax.set_title("Feature Importance")
st.pyplot(fig_imp)

st.subheader("Average Price per City")
city_avg = filtered_df.groupby("city")["price"].mean().sort_values(ascending= False)
fig1, ax1 = plt.subplots(figsize=(10,5))
sns.barplot(x=city_avg.values, y=city_avg.index, ax=ax1)
ax1.set_xlabel("Average Price (PLN)")
ax1.set_title("Average Price by City")
st.pyplot(fig1)

st.subheader("Apartment Count by Number of POIs nearby")
if 'poiCount' in filtered_df.columns:
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    sns.histplot(filtered_df['poiCount'], bins=30, kde=True, ax=ax2)
    ax2.set_title("Distribution of POI Count")
    ax2.set_xlabel("Number of Points of Interest")
    ax2.set_ylabel("Number of Apartments")
    st.pyplot(fig2)

st.subheader("Price vs Selected Features")
selected_feature = st.selectbox("Select featire to analyze:", features)
if selected_feature in filtered_df.columns:
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    sns.scatterplot(data=filtered_df, x=selected_feature, ax=ax3, alpha= 0.6)
    sns.regplot(data=filtered_df, x=selected_feature, y="price", scatter=False, ax=ax3, color="red")
    ax3.set_title(f"Price vs {selected_feature}")
    st.pyplot(fig3)

st.subheader("Price Distribution by Number of Rooms")
fig4, ax4 = plt.subplots(figsize=(10, 5))
sns.boxplot(data=filtered_df, x='rooms', y='price', ax=ax4)
ax4.set_title("Price Distribution by Number of Rooms")
st.pyplot(fig4)

st.subheader("Average Price per Floor")
if 'floor' in filtered_df.columns:
    floor_avg = filtered_df.groupby('floor')['price'].mean().sort_index()
    fig5, ax5 = plt.subplots(figsize=(10, 5))
    sns.lineplot(x=floor_avg.index, y=floor_avg.values, marker='o',ax=ax5)
    ax5.set_title("Average Price per Floor")
    ax5.set_xlabel("Floor")
    ax5.set_ylabel("Average Price (PLN)")
    st.pyplot(fig5)
else:
    st.warning("Not enough data to train the model.")
