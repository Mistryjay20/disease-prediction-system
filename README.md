# 🧑‍⚕️ Disease Prediction and Precaution System (Naïve Bayes)

A **Streamlit-based web app** that predicts diseases from user-selected symptoms and suggests relevant precautions.  
This project uses **Naïve Bayes classification** on a custom dataset of symptoms and precautions.

---

## 🚀 Features
- Upload custom `symptoms.csv` and `precautions.csv` datasets.
- **Disease → Symptoms**: View symptoms linked to a selected disease.
- **Symptoms → Disease**: Predict possible diseases from chosen symptoms.
- **Disease → Precautions**: Get precautionary steps for a disease.
- Interactive **probability distribution** (bar chart + top predictions).
- Displays **model accuracy** after training.

---

## 📂 Dataset Format

### `symptoms.csv`
| Disease         | Symptom_1 | Symptom_2 | Symptom_3 | ... |
|-----------------|-----------|-----------|-----------|-----|
| Diabetes        | Fatigue   | Thirst    | Frequent urination | ... |
| Hypertension    | Headache  | Dizziness | Nosebleed | ... |

### `precautions.csv`
| Disease   | Precaution_1     | Precaution_2       | Precaution_3   | Precaution_4 |
|-----------|------------------|--------------------|----------------|--------------|
| Diabetes  | Healthy diet     | Exercise regularly | Monitor sugar  | Take insulin |
| Flu       | Stay hydrated    | Rest properly      | Take medicine  | Avoid crowds |

---

## Run Application
- streamlit run app.py



