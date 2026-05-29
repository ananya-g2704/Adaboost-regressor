import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Insurance Charges Prediction",
    page_icon="💰",
    layout="wide"
)

# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

df = pd.read_csv("data/insurance.csv")

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

model = pickle.load(
    open("models/adaboost_regressor.pkl", "rb")
)

scaler = pickle.load(
    open("models/scaler.pkl", "rb")
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

.stApp{
    background: linear-gradient(
        to right,
        #ff9a9e,
        #fecfef,
        #fcb69f
    );
}

.title{
    text-align:center;
    font-size:55px;
    font-weight:bold;
    color:#8b004f;
}

.subtitle{
    text-align:center;
    font-size:20px;
    color:#6d214f;
    margin-bottom:25px;
}

.card{
    background:rgba(255,255,255,0.45);
    padding:20px;
    border-radius:20px;
    text-align:center;
    color:#6d214f;
    box-shadow:0px 4px 12px rgba(0,0,0,0.1);
}

.glass{
    background:rgba(255,255,255,0.35);
    padding:30px;
    border-radius:20px;
}

.stButton > button{
    width:100%;
    height:55px;
    font-size:20px;
    font-weight:bold;
    border-radius:12px;
    border:none;
    background:linear-gradient(
        to right,
        #ff758c,
        #ff7eb3
    );
    color:white;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.markdown(
    "<div class='title'>💰 Insurance Charges Prediction</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>AdaBoost Regression Model</div>",
    unsafe_allow_html=True
)

# --------------------------------------------------
# DATASET OVERVIEW
# --------------------------------------------------

st.subheader("📊 Dataset Overview")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(
        f"<div class='card'><h3>📄 Rows</h3><h2>{df.shape[0]}</h2></div>",
        unsafe_allow_html=True
    )

with c2:
    st.markdown(
        f"<div class='card'><h3>📊 Columns</h3><h2>{df.shape[1]}</h2></div>",
        unsafe_allow_html=True
    )

with c3:
    st.markdown(
        f"<div class='card'><h3>❌ Missing</h3><h2>{df.isnull().sum().sum()}</h2></div>",
        unsafe_allow_html=True
    )

with c4:
    st.markdown(
        f"<div class='card'><h3>🎯 Features</h3><h2>{df.shape[1]-1}</h2></div>",
        unsafe_allow_html=True
    )

# --------------------------------------------------
# DATASET PREVIEW
# --------------------------------------------------

st.subheader("🔍 Dataset Preview")

st.dataframe(df.head())

# --------------------------------------------------
# ONE SMALL PLOT
# --------------------------------------------------

st.subheader("📈 Charges Distribution")

left, center, right = st.columns([1,2,1])

with center:

    fig, ax = plt.subplots(figsize=(4,3))

    ax.hist(
        df["charges"],
        bins=15
    )

    ax.set_title("Insurance Charges Distribution")

    st.pyplot(fig)

# --------------------------------------------------
# INPUT SECTION
# --------------------------------------------------

st.subheader("📝 Customer Details")

left, center, right = st.columns([1,3,1])

with center:


    col1, col2, col3 = st.columns(3)

    with col1:

        age = st.slider(
            "Age",
            18, 100, 30
        )

        bmi = st.slider(
            "BMI",
            10.0, 60.0, 25.0
        )

    with col2:

        children = st.slider(
            "Children",
            0, 10, 1
        )

        sex_display = st.selectbox(
            "Sex",
            ["female", "male"]
        )

        sex = 0 if sex_display == "female" else 1

    with col3:

        smoker_display = st.selectbox(
            "Smoker",
            ["no", "yes"]
        )

        smoker = 0 if smoker_display == "no" else 1

        region_display = st.selectbox(
            "Region",
            [
                "northeast",
                "northwest",
                "southeast",
                "southwest"
            ]
        )

        region_map = {
            "northeast": 0,
            "northwest": 1,
            "southeast": 2,
            "southwest": 3
        }

        region = region_map[region_display]

    predict = st.button(
        "Predict Charges",
        use_container_width=True
    )

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if predict:

    input_data = np.array([[
        age,
        sex,
        bmi,
        children,
        smoker,
        region
    ]])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]


    st.markdown(
        "<h1 style='color:#d63384;text-align:center;'>💰 Insurance Charges</h1>",
        unsafe_allow_html=True
    )

    st.markdown(
        f"<h1 style='color:#8b004f;text-align:center;font-size:55px;'>₹ {prediction:,.2f}</h1>",
        unsafe_allow_html=True
    )

    st.markdown("</div>", unsafe_allow_html=True)