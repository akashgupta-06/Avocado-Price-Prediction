import pickle
import streamlit as st
import pandas as pd

# Load models and training column structure
model_names = ['lgbm_model', 'xgboost_model', 'Random_model']
models = {name: pickle.load(open(f'{name}.pkl', 'rb')) for name in model_names}
training_columns = pickle.load(open("columns.pkl", "rb"))  # 👈 Load saved columns

results_df = pd.read_csv("model_evaluation_results.csv")

st.title("🥑 Avocado Price Prediction App")
st.sidebar.header("⚙️ Model Selection")
model_name = st.sidebar.selectbox("Choose a model", model_names)

# Numerical inputs
Volume = st.number_input("Total Volume", min_value=0.0, format="%.2f")
Small_bags = st.number_input("Small Bags", min_value=0.0, format="%.2f")
Large_bags = st.number_input("Large Bags", min_value=0.0, format="%.2f")
XLarge_bags = st.number_input("XLarge Bags", min_value=0.0, format="%.2f")

# Categorical options
type_options = ["conventional", "organic"]
year_options = ["2015", "2016", "2017", "2018"]
region_options = ['Albany', 'Atlanta', 'BaltimoreWashington', 'Boise', 'Boston',
       'BuffaloRochester', 'California', 'Charlotte', 'Chicago',
       'CincinnatiDayton', 'Columbus', 'DallasFtWorth', 'Denver',
       'Detroit', 'GrandRapids', 'GreatLakes', 'HarrisburgScranton',
       'HartfordSpringfield', 'Houston', 'Indianapolis', 'Jacksonville',
       'LasVegas', 'LosAngeles', 'Louisville', 'MiamiFtLauderdale',
       'Nashville', 'NewOrleansMobile', 'NewYork', 'NorthernNewEngland',
       'Orlando', 'Philadelphia', 'PhoenixTucson', 'Pittsburgh', 'Plains',
       'Portland', 'RaleighGreensboro', 'RichmondNorfolk', 'Roanoke',
       'Sacramento', 'SanDiego', 'SanFrancisco', 'Seattle',
       'SouthCarolina', 'SouthCentral', 'Spokane', 'StLouis', 'Syracuse',
       'Tampa', 'TotalUS', 'WestTexNewMexico']

type_selected = st.selectbox("Type", type_options)
year_selected = st.selectbox("Year", year_options)
region_selected = st.selectbox("Region", region_options)

# Prediction
if st.button("🔮 Predict"):
    # Build raw input DataFrame
    raw_input = {
        "Total Volume": Volume,
        "Small Bags": Small_bags,
        "Large Bags": Large_bags,
        "XLarge Bags": XLarge_bags,
        "type": type_selected,
        "year": year_selected,
        "region": region_selected
    }

    input_df = pd.DataFrame([raw_input])

    # 🪄 Encode using get_dummies like in training
    input_encoded = pd.get_dummies(input_df)

    # 🪄 Reindex to match training columns
    input_encoded = input_encoded.reindex(columns=training_columns, fill_value=0)

    # Predict
    model = models[model_name]
    prediction = model.predict(input_encoded)[0]

    st.success(f"🥑 Predicted Price with {model_name}: ${prediction:,.2f}")

# Show evaluation results
st.header("📊 Model Evaluation Results")
st.dataframe(results_df)

