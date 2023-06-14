import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title='Daegu Apartment Predict Price'
)

# Load the XGBoost model from the .sav file
model = joblib.load('xgboost_model_1.sav')

# Define the Streamlit app
def main():
    st.title("Apartment Daegu Price Prediction")
    st.write("Enter the apartment features to predict the price")

    # Create input fields for the features
    hallway_type = st.selectbox("Hallway Type", ['terraced', 'mixed', 'corridor'])
    time_to_subway = st.selectbox("Time to Subway", ['0-5min', '5min~10min', '10min~15min', '15min~20min', 'no_bus_stop_nearby'])
    subway_station = st.selectbox("Subway Station", ['Kyungbuk_uni_hospital', 'Chil-sung-market', 'Bangoge', 'Sin-nam', 'Banwoldang', 'no_subway_nearby', 'Myung-duk', 'Daegu'])
    facilities_etc = st.number_input("N_FacilitiesNearBy(ETC)")
    facilities_office = st.number_input("N_FacilitiesNearBy(PublicOffice)")
    school_nearby = st.number_input("N_SchoolNearBy(University)")
    parking_lot = st.number_input("N_Parkinglot(Basement)")
    year_built = st.number_input("Year Built")
    facilities_in_apt = st.number_input("N_FacilitiesInApt")
    size_sqf = st.number_input("Size(sqf)")

    # Map the time_to_subway to ordinal values
    time_mapping = {'0-5min': 4, '5min~10min': 3, '10min~15min': 2, '15min~20min': 1, 'no_bus_stop_nearby': 0}
    time_to_subway = time_mapping[time_to_subway]

    # Create a DataFrame with the input values
    data = {
        "HallwayType": [hallway_type],
        "TimeToSubway": [time_to_subway],
        "SubwayStation": [subway_station],
        "N_FacilitiesNearBy(ETC)": [facilities_etc],
        "N_FacilitiesNearBy(PublicOffice)": [facilities_office],
        "N_SchoolNearBy(University)": [school_nearby],
        "N_Parkinglot(Basement)": [parking_lot],
        "YearBuilt": [year_built],
        "N_FacilitiesInApt": [facilities_in_apt],
        "Size(sqf)": [size_sqf]
    }
    input_df = pd.DataFrame(data)

    # Make predictions using the XGBoost model
    prediction = model.predict(input_df)

    # Display the predicted price
    st.write(f"Predicted Price: ${round(prediction[0])}")
    st.write (f"with upper base: ${round(prediction[0])+38515}")
    st.write (f"with lower base: ${round(prediction[0])-38515}" )

# Run the app
if __name__ == "__main__":
    main()
