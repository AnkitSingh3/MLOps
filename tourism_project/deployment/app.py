import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="AnkitImpetus/mlops-hf-space-model", filename="best_tourism_package_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Tourism Package Prediction
st.title("Tourism Package Prediction App")
st.write("""
This application predicts potential buyers, and enhances decision-making for marketing strategies.
Please enter below parameters to get a prediction.
""")

# Collect user input
Age = st.number_input("Age (customer age)", min_value=18, max_value=61, value=35)
TypeofContact = st.selectbox("Type Of Contact", ["Self Enquiry", "Company Invited"])
CityTier = st.selectbox("City Tier (customer city classification tier)", [1, 2, 3])
DurationOfPitch = st.number_input("Duration Of Pitch (duration of sales pitch in minutes)", min_value=5.0, max_value=127.0,value=15.0)
Occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
Gender = st.selectbox("Gender", ["Male", "Female"])
NumberOfPersonVisiting = st.number_input("Number Of Persons Visiting (number of people visiting)", min_value=1, max_value=5,value=2)
NumberOfFollowups = st.selectbox("Number Of Followups",[1, 2, 3, 4, 5, 6])
ProductPitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
PreferredPropertyStar = st.selectbox("Preferred Property Star Rating", [3, 4, 5])
MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
NumberOfTrips = st.number_input("Number Of Trips (number of trips taken)", min_value=1, max_value=22,value=2)
Passport = st.selectbox("Passport Available?", ["Yes", "No"])
PitchSatisfactionScore = st.selectbox("Pitch Satisfaction Score", [1, 2, 3, 4, 5])
OwnCar = st.selectbox("Owns a Car?", ["Yes", "No"])
NumberOfChildrenVisiting = st.number_input("Number Of Children Visiting", min_value=0, value=0)
Designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
MonthlyIncome = st.number_input("Monthly Income", min_value=0.0, value=50000.0)


# Create input dataframe
input_data = pd.DataFrame([{
    'Age': Age,
    'CityTier': CityTier,
    'DurationOfPitch': DurationOfPitch,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'NumberOfFollowups': NumberOfFollowups,
    'PreferredPropertyStar': PreferredPropertyStar,
    'NumberOfTrips': NumberOfTrips,
    'Passport': 1 if Passport == "Yes" else 0,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'OwnCar': 1 if OwnCar == "Yes" else 0,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'MonthlyIncome': MonthlyIncome,
    'TypeofContact': TypeofContact,
    'Occupation': Occupation,
    'Gender': Gender,
    'ProductPitched': ProductPitched,
    'MaritalStatus': MaritalStatus,
    'Designation': Designation
}])

# Classification threshold
classification_threshold = 0.50

if st.button("Predict"):
    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = int(prediction_proba >= classification_threshold)

    result = "Purchase the package" if prediction == 1 else "Not purchase the package"

    st.subheader("Prediction Result")
    st.write(f"Probability of Purchase: **{prediction_proba:.2%}**")
    st.write(f"Prediction: **{result}**")
