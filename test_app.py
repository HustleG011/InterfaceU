import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from tensorflow.keras.models import load_model

st.set_page_config(page_title="WNBA Position Prediction", layout="wide")

XGB_MODEL_FILE = "model_xgb.pkl"
XGB_ENCODER_FILE = "label_encoder_xgb.pkl"
XGB_FEATURES_FILE = "xgb_feature_names.pkl"

RF_MODEL_FILE = "model_rf.pkl"
RF_FEATURES_FILE = "rf_feature_names.pkl"

NN_MODEL_FILE = "model_nn.keras"
NN_SCALER_FILE = "scaler_nn.pkl"
NN_ENCODER_FILE = "label_encoder_nn.pkl"
NN_FEATURES_FILE = "nn_feature_names.pkl"

PCA_MODEL_FILE = "pca_transform.pkl"
RAW_FEATURES_FILE = "raw_feature_names.pkl"

METRICS_FILE = "model_metrics.pkl"


def xgb_files_exist():
    return (
        Path(XGB_MODEL_FILE).exists()
        and Path(XGB_ENCODER_FILE).exists()
        and Path(XGB_FEATURES_FILE).exists()
    )


def rf_files_exist():
    return (
        Path(RF_MODEL_FILE).exists()
        and Path(RF_FEATURES_FILE).exists()
    )


def nn_files_exist():
    return (
        Path(NN_MODEL_FILE).exists()
        and Path(NN_SCALER_FILE).exists()
        and Path(NN_ENCODER_FILE).exists()
        and Path(NN_FEATURES_FILE).exists()
    )


def pca_files_exist():
    return (
        Path(PCA_MODEL_FILE).exists()
        and Path(RAW_FEATURES_FILE).exists()
    )


def metrics_file_exist():
    return Path(METRICS_FILE).exists()


@st.cache_resource
def load_xgb_files():
    model = joblib.load(XGB_MODEL_FILE)
    label_encoder = joblib.load(XGB_ENCODER_FILE)
    feature_names = joblib.load(XGB_FEATURES_FILE)
    return model, label_encoder, feature_names


@st.cache_resource
def load_rf_files():
    model = joblib.load(RF_MODEL_FILE)
    feature_names = joblib.load(RF_FEATURES_FILE)
    return model, feature_names


@st.cache_resource
def load_nn_files():
    model = load_model(NN_MODEL_FILE)
    scaler = joblib.load(NN_SCALER_FILE)
    label_encoder = joblib.load(NN_ENCODER_FILE)
    feature_names = joblib.load(NN_FEATURES_FILE)
    return model, scaler, label_encoder, feature_names


@st.cache_resource
def load_pca_files():
    pca_model = joblib.load(PCA_MODEL_FILE)
    raw_feature_names = joblib.load(RAW_FEATURES_FILE)
    return pca_model, raw_feature_names


@st.cache_resource
def load_metrics_file():
    return joblib.load(METRICS_FILE)


st.title("Prediction du poste d une joueuse WNBA")
st.write(
    "Saisissez les statistiques d une joueuse. "
    "L application applique automatiquement le PCA puis affiche la classification des 3 modeles."
)

st.info(
    "Cette interface permet de saisir un nouvel objet a classer, "
    "d afficher les predictions des 3 modeles et leurs metriques d evaluation."
)

if not pca_files_exist():
    st.error("Fichiers PCA manquants : pca_transform.pkl et raw_feature_names.pkl")
    st.stop()

try:
    pca_model, raw_feature_names = load_pca_files()
except Exception as e:
    st.error(f"Erreur de chargement PCA : {e}")
    st.stop()

st.subheader("Saisie d un nouvel objet a classer")

input_data = {}
col1, col2 = st.columns(2)

for i, feature in enumerate(raw_feature_names):
    target_col = col1 if i % 2 == 0 else col2
    with target_col:
        input_data[feature] = st.number_input(
            feature,
            value=0.0,
            step=0.1,
            format="%.4f"
        )

if st.button("Classifier la joueuse"):
    try:
        raw_df = pd.DataFrame([input_data])
        raw_df = raw_df[raw_feature_names]

        pca_values = pca_model.transform(raw_df)
        pc_columns = [f"PC_{i+1}" for i in range(pca_values.shape[1])]
        pca_df = pd.DataFrame(pca_values, columns=pc_columns)

        st.subheader("Statistiques entrees")
        st.write(input_data)

        st.subheader("Composantes principales calculees")
        st.write(pca_df)

        prediction_results = {}

        # XGBoost
        if xgb_files_exist():
            model_xgb, label_encoder_xgb, xgb_feature_names = load_xgb_files()
            xgb_input = pca_df[xgb_feature_names]
            pred_xgb_encoded = model_xgb.predict(xgb_input)[0]
            pred_xgb_label = label_encoder_xgb.inverse_transform([int(pred_xgb_encoded)])[0]
            prediction_results["XGBoost"] = pred_xgb_label
        else:
            prediction_results["XGBoost"] = "Fichiers manquants"

        # Random Forest
        if rf_files_exist():
            model_rf, rf_feature_names = load_rf_files()
            rf_input = pca_df[rf_feature_names]
            pred_rf_label = model_rf.predict(rf_input)[0]
            prediction_results["Random Forest"] = pred_rf_label
        else:
            prediction_results["Random Forest"] = "Fichiers manquants"

        # Neural Network
        if nn_files_exist():
            model_nn, scaler_nn, label_encoder_nn, nn_feature_names = load_nn_files()
            nn_input = pca_df[nn_feature_names]
            nn_input_scaled = scaler_nn.transform(nn_input)
            pred_nn_probs = model_nn.predict(nn_input_scaled, verbose=0)
            pred_nn_encoded = pred_nn_probs.argmax(axis=1)[0]
            pred_nn_label = label_encoder_nn.inverse_transform([int(pred_nn_encoded)])[0]
            prediction_results["Neural Network"] = pred_nn_label
        else:
            prediction_results["Neural Network"] = "Fichiers manquants"

        st.subheader("Resultats de classification")
        results_df = pd.DataFrame(
            {
                "Modele": list(prediction_results.keys()),
                "Poste predit": list(prediction_results.values())
            }
        )
        st.table(results_df)

        st.subheader("Metriques d evaluation")
        if metrics_file_exist():
            metrics_df = load_metrics_file()
            st.table(metrics_df)
        else:
            metrics_df = pd.DataFrame(
                {
                    "Modele": ["Random Forest", "XGBoost", "Neural Network"],
                    "Accuracy": [0.4802, 0.5463, 0.4273],
                    "Precision": [0.4593, 0.5476, 0.4814],
                    "Recall": [0.4802, 0.5463, 0.4273],
                    "F1-score": [0.4631, 0.5417, 0.4150]
                }
            )
            st.table(metrics_df)

    except Exception as e:
        st.error(f"Erreur de connexion : {e}")