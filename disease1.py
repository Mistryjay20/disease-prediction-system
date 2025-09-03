import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import streamlit as st


# Load dataset
def load_data(file) -> pd.DataFrame:
    return pd.read_csv(file)


# Preprocessing helper
def preprocess_dataset(symptoms_df: pd.DataFrame, precaution_df: pd.DataFrame):
    disease_col = 'Disease'

    symptom_cols = [c for c in symptoms_df.columns if 'Symptom' in c]
    precaution_cols = [c for c in precaution_df.columns if 'Precaution' in c]

    # Merge datasets on Disease column
    df = pd.merge(symptoms_df, precaution_df, on=disease_col, how="left")

    disease_to_symptoms = {
        row[disease_col]: [s for s in row[symptom_cols].dropna().astype(str)]
        for _, row in df.iterrows()
    }

    disease_to_precautions = {
        row[disease_col]: [p for p in row[precaution_cols].dropna().astype(str)]
        for _, row in df.iterrows()
    }

    all_symptoms = sorted({s for symptoms in disease_to_symptoms.values() for s in symptoms})

    X = [[1 if s in symptoms else 0 for s in all_symptoms] for symptoms in disease_to_symptoms.values()]
    y = list(disease_to_symptoms.keys())

    return np.array(X), np.array(y), all_symptoms, disease_to_symptoms, disease_to_precautions


# Model training
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = MultinomialNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return model, accuracy


# ✅ Cache preprocessing
@st.cache_data
def cached_preprocess(symptoms_df, precaution_df):
    return preprocess_dataset(symptoms_df, precaution_df)


# ✅ Cache model training
@st.cache_resource
def cached_train_model(X, y):
    return train_model(X, y)


# Streamlit app
def main():
    st.set_page_config(page_title="Disease Prediction",layout="wide")
    st.title("Disease Prediction and Precaution System (Naïve Bayes)")

    with st.sidebar:
        st.header("Upload Dataset")
        sym_file = st.file_uploader("Upload Symptoms CSV", type=["csv"], key="sym")
        pre_file = st.file_uploader("Upload Precautions CSV", type=["csv"], key="pre")

    if sym_file is None or pre_file is None:
        st.info("Please upload both symptoms.csv and precaution.csv to continue.")
        st.stop()

    # Load datasets
    symptoms_df = load_data(sym_file)
    precaution_df = load_data(pre_file)

    # Preprocess (cached)
    X, y, all_symptoms, disease_to_symptoms, disease_to_precautions = cached_preprocess(symptoms_df, precaution_df)

    # Train model (cached)
    model, acc = cached_train_model(X, y)
    st.sidebar.success(f"Model trained successfully (Accuracy: {acc:.2f})")

    # Tabs for 3 modes
    tab1, tab2, tab3 = st.tabs(
        ["Disease → Symptoms", "Symptoms → Disease", "Disease → Precautions"]
    )

    # --- Tab 1: Disease → Symptoms ---
    with tab1:
        st.subheader("Find Symptoms from Disease")
        disease = st.selectbox("Select a disease", sorted(disease_to_symptoms.keys()))
        if disease:
            st.write("### Symptoms:")
            st.write("\n".join([f"- {s}" for s in disease_to_symptoms[disease]]))  # bullet points

    # --- Tab 2: Symptoms → Disease ---
    with tab2:
        st.subheader("Predict Disease from Symptoms")
        selected_syms = st.multiselect("Select symptoms", options=all_symptoms)
        if st.button("Predict Disease"):
            x = np.array([[1 if s in selected_syms else 0 for s in all_symptoms]])
            probs = model.predict_proba(x)[0]
            classes = model.classes_
            order = np.argsort(probs)[::-1]
            
            # Convert to percentage
            top = [(classes[i], probs[i] * 100) for i in order[:5]]
            df_top = pd.DataFrame(top, columns=["Disease", "Probability (%)"])
            
            # Show best prediction
            st.success(f"Predicted: {top[0][0]} ({top[0][1]:.1f}% confidence)")
            
            # Show table
            st.write("### Top Candidates:")
            st.table(df_top)
            
            # Show bar chart
            st.write("### Probability Distribution:")
            st.bar_chart(df_top.set_index("Disease"))

    # --- Tab 3: Disease → Precautions ---
    with tab3:
        st.subheader("Find Precautions from Disease")
        disease = st.selectbox(
            "Select a disease", sorted(disease_to_precautions.keys()), key="precautions"
        )
        if disease:
            st.write("### Precautions:")
            st.write("\n".join([f"- {p}" for p in disease_to_precautions[disease]]))


if __name__ == "__main__":
    main()
