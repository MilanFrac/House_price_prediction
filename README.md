# 🏘️ Apartment Prices in Poland – Streamlit Dashboard

This is an interactive data visualization and machine learning dashboard built using **Streamlit**, analyzing apartment prices in Poland. The application provides filters, price statistics, and a regression model to predict real estate prices based on apartment features.

---

## 📊 Features

- 📁 Load and combine multiple CSV datasets from a folder
- 📍 Filter apartments by:
  - City
  - Year
  - Room count
  - Area
  - Price
- 🧠 Train an XGBoost regression model on selected features
- 🔮 Predict apartment price based on user input
- 📈 Display charts:
  - Feature importance
  - Average price per city
  - Price distribution by POI count
  - Price vs selected features (with regression line)
  - Boxplot: Price vs number of rooms
  - Line chart: Price per floor

---

## 🛠️ Tech Stack

- [Streamlit](https://streamlit.io/)
- [Pandas](https://pandas.pydata.org/)
- [Matplotlib](https://matplotlib.org/)
- [Seaborn](https://seaborn.pydata.org/)
- [XGBoost](https://xgboost.readthedocs.io/)
- [Scikit-learn](https://scikit-learn.org/)

---

## 📂 Project Structure

```
📁 your_project/
 ├── main.py           # Main Streamlit app
 ├── /archive/         # Folder with CSV files e.g., apartments_pl_2020_01.csv
 └── README.md
```

> ⚠️ The data folder should contain files named like: `apartments_pl_2022_06.csv`, etc.

---

## 🚀 How to Run

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/apartment-prices-pl.git
cd apartment-prices-pl
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

**Example `requirements.txt`:**

```
streamlit
pandas
matplotlib
seaborn
xgboost
scikit-learn
```

### 3. Run the app

```bash
streamlit run main.py
```

---

## 🤖 Model

**Model:** XGBoost Regressor  
**Target:** Apartment price

**Features:**

- `squareMeters`, `rooms`, `floor`, `floorCount`, `buildYear`
- `centreDistance`, `poiCount`, `schoolDistance`, `clinicDistance`
- `postOfficeDistance`, `kindergartenDistance`, `restaurantDistance`
- `collegeDistance`, `pharmacyDistance`

---

## 💡 Example Use Cases

- Estimate how much a flat might cost in **Gdańsk** or **Warsaw** based on size, rooms, and amenities.
- Discover which features most affect apartment pricing.
- Compare price evolution across years and cities.

---

## ❗ Notes

- The model requires **sufficient data**. Filtering too narrowly may prevent model training.
- Data cleaning is **minimal** – ensure input files are consistent.

---

## 🧾 License

MIT License
