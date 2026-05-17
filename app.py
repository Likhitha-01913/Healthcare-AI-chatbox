import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# --------------------------------------------
# Training Dataset
# --------------------------------------------

data = {
    "fever":            [1,1,0,0,1,0,1,0],
    "cough":            [1,1,0,1,0,0,1,0],
    "headache":         [1,0,1,0,1,0,1,0],
    "fatigue":          [1,1,1,0,1,1,0,0],
    "stomach_pain":     [0,0,1,1,0,1,0,0],
    "chest_pain":       [0,0,0,1,0,0,1,0],
    "shortness_breath": [0,0,0,1,0,0,1,0],
    "disease": [
        "Flu",
        "Cold",
        "Migraine",
        "Heart Disease",
        "Viral Fever",
        "Food Poisoning",
        "COVID-19",
        "Healthy"
    ]
}

df = pd.DataFrame(data)

# --------------------------------------------
# Features and Labels
# --------------------------------------------

X = df.drop("disease", axis=1)
y = df["disease"]

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# --------------------------------------------
# Train AI Model
# --------------------------------------------

model = DecisionTreeClassifier()
model.fit(X, y_encoded)

# --------------------------------------------
# Streamlit App UI
# --------------------------------------------

st.title("AI Healthcare Chatbot")

st.write("Select your symptoms")

fever = st.checkbox("Fever")
cough = st.checkbox("Cough")
headache = st.checkbox("Headache")
fatigue = st.checkbox("Fatigue")
stomach_pain = st.checkbox("Stomach Pain")
chest_pain = st.checkbox("Chest Pain")
shortness_breath = st.checkbox("Shortness of Breath")

# --------------------------------------------
# Predict Disease
# --------------------------------------------

if st.button("Predict Health Condition"):

    symptoms = []

    symptoms.append(int(fever))
    symptoms.append(int(cough))
    symptoms.append(int(headache))
    symptoms.append(int(fatigue))
    symptoms.append(int(stomach_pain))
    symptoms.append(int(chest_pain))
    symptoms.append(int(shortness_breath))

    prediction = model.predict(
        pd.DataFrame([symptoms], columns=X.columns)
    )

    disease_name = encoder.inverse_transform(prediction)

    st.subheader("Predicted Health Condition")
    st.success(disease_name[0])

    # Health Suggestions

    if disease_name[0] == "Flu":
        st.info("Rest, drink fluids, and monitor temperature.")

    elif disease_name[0] == "Cold":
        st.info("Take warm fluids and proper rest.")

    elif disease_name[0] == "Migraine":
        st.info("Reduce stress and rest in a dark room.")

    elif disease_name[0] == "Heart Disease":
        st.error("Consult a cardiologist immediately.")

    elif disease_name[0] == "COVID-19":
        st.warning("Isolate yourself and consult a doctor.")

    elif disease_name[0] == "Food Poisoning":
        st.info("Drink ORS and avoid outside food.")

    elif disease_name[0] == "Viral Fever":
        st.info("Stay hydrated and monitor fever.")

    else:
        st.success("You seem healthy. Maintain a good lifestyle.")