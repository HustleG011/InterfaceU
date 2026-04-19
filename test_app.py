import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from tensorflow.keras.models import load_model

st.set_page_config(
    page_title="WNBA Position Prediction",
    layout="wide",
)

# =========================
# FILES
# =========================
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


# =========================
# STYLE
# =========================
def inject_custom_css():
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 55%, #1d4ed8 100%);
        color: #f8fafc;
    }

    [data-testid="stHeader"] {
        background: rgba(0,0,0,0);
    }

    [data-testid="stSidebar"] {
        background: #111827;
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1300px;
    }

    .main-card {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.10);
        padding: 24px;
        border-radius: 20px;
        margin-bottom: 20px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.18);
    }

    .main-title {
        font-size: 2rem;
        font-weight: 800;
        color: white;
        margin-bottom: 8px;
    }

    .main-subtitle {
        font-size: 1rem;
        color: #dbeafe;
        line-height: 1.5;
    }

    .section-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #ffffff;
        margin-top: 8px;
        margin-bottom: 12px;
    }

    .info-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.08);
        padding: 16px;
        border-radius: 16px;
        margin-bottom: 14px;
    }

    .result-card {
        background: rgba(59,130,246,0.16);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 18px;
        padding: 18px;
        text-align: center;
        box-shadow: 0 8px 20px rgba(0,0,0,0.16);
    }

    .result-model {
        font-size: 1rem;
        color: #cbd5e1;
        font-weight: 600;
        margin-bottom: 8px;
    }

    .result-value {
        font-size: 1.6rem;
        color: #ffffff;
        font-weight: 800;
    }

    .status-badge {
        display: inline-block;
        padding: 8px 12px;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 700;
        margin: 4px 6px 4px 0;
    }

    .badge-ok {
        background: rgba(16,185,129,0.18);
        color: #6ee7b7;
        border: 1px solid rgba(16,185,129,0.35);
    }

    .badge-missing {
        background: rgba(239,68,68,0.18);
        color: #fca5a5;
        border: 1px solid rgba(239,68,68,0.35);
    }

    div[data-baseweb="input"] > div {
        background-color: rgba(255,255,255,0.06) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
    }

    label, .stNumberInput label {
        color: #e5e7eb !important;
        font-weight: 600 !important;
    }

    .stButton > button, div.stFormSubmitButton > button {
        width: 100%;
        border: none;
        border-radius: 14px;
        padding: 0.85rem 1.1rem;
        font-size: 1rem;
        font-weight: 700;
        color: white;
        background: linear-gradient(135deg, #2563eb, #4f46e5);
        box-shadow: 0 8px 18px rgba(37,99,235,0.28);
        transition: 0.2s ease-in-out;
    }

    .stButton > button:hover, div.stFormSubmitButton > button:hover {
        transform: translateY(-1px);
    }

    .stDataFrame, .stTable {
        background: rgba(255,255,255,0.04);
        border-radius: 14px;
        overflow: hidden;
    }

    [data-testid="metric-container"] {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.08);
        padding: 12px;
        border-radius: 16px;
    }
    </style>
    """, unsafe_allow_html=True)


# =========================
# FILE CHECKS
# =========================
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


# =========================
# LOADERS
# =========================
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


# =========================
# UI HELPERS
# =========================
def section_title(text):
    st.markdown(f"<div class='section-title'>{text}</div>", unsafe_allow_html=True)


def status_badge(label, ok=True):
    cls = "badge-ok" if ok else "badge-missing"
    text = "Disponible" if ok else "Manquant"
    return f"<div class='status-badge {cls}'>{label} : {text}</div>"


def result_card(model_name, prediction):
    return f"""
    <div class="result-card">
        <div class="result-model">{model_name}</div>
        <div class="result-value">{prediction}</div>
    </div>
    """


# =========================
# START APP
# =========================
inject_custom_css()

st.markdown("""
<div class="main-card">
    <div class="main-title">Prédiction du poste d’une joueuse WNBA</div>
    <div class="main-subtitle">
        Entrez les statistiques d’une joueuse pour obtenir la prédiction du poste
        avec plusieurs modèles.
    </div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## Tableau de bord")
    st.markdown(status_badge("PCA", pca_files_exist()), unsafe_allow_html=True)
    st.markdown(status_badge("XGBoost", xgb_files_exist()), unsafe_allow_html=True)
    st.markdown(status_badge("Random Forest", rf_files_exist()), unsafe_allow_html=True)
    st.markdown(status_badge("Neural Network", nn_files_exist()), unsafe_allow_html=True)
    st.markdown(status_badge("Métriques", metrics_file_exist()), unsafe_allow_html=True)

if not pca_files_exist():
    st.error("Fichiers PCA manquants : pca_transform.pkl et raw_feature_names.pkl")
    st.stop()

try:
    pca_model, raw_feature_names = load_pca_files()
except Exception as e:
    st.error(f"Erreur de chargement PCA : {e}")
    st.stop()

left_col, right_col = st.columns([1.5, 1], gap="large")

with left_col:
    section_title("Saisie des statistiques")

    input_data = {}

    with st.form("prediction_form"):
        cols = st.columns(3)

        for i, feature in enumerate(raw_feature_names):
            with cols[i % 3]:
                input_data[feature] = st.number_input(
                    feature,
                    value=0.0,
                    step=0.1,
                    format="%.4f"
                )

        submitted = st.form_submit_button("Lancer la prédiction")

with right_col:
    section_title("Informations")
    st.markdown("""
    <div class="info-card">
        Les données entrées sont transformées avec le PCA avant d’être envoyées aux modèles.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
        L’application compare les résultats de XGBoost, Random Forest et Neural Network.
    </div>
    """, unsafe_allow_html=True)

if submitted:
    try:
        raw_df = pd.DataFrame([input_data])
        raw_df = raw_df[raw_feature_names]

        pca_values = pca_model.transform(raw_df)
        pc_columns = [f"PC_{i+1}" for i in range(pca_values.shape[1])]
        pca_df = pd.DataFrame(pca_values, columns=pc_columns)

        prediction_results = {}

        if xgb_files_exist():
            model_xgb, label_encoder_xgb, xgb_feature_names = load_xgb_files()
            xgb_input = pca_df[xgb_feature_names]
            pred_xgb_encoded = model_xgb.predict(xgb_input)[0]
            pred_xgb_label = label_encoder_xgb.inverse_transform([int(pred_xgb_encoded)])[0]
            prediction_results["XGBoost"] = pred_xgb_label
        else:
            prediction_results["XGBoost"] = "Indisponible"

        if rf_files_exist():
            model_rf, rf_feature_names = load_rf_files()
            rf_input = pca_df[rf_feature_names]
            pred_rf_label = model_rf.predict(rf_input)[0]
            prediction_results["Random Forest"] = pred_rf_label
        else:
            prediction_results["Random Forest"] = "Indisponible"

        if nn_files_exist():
            model_nn, scaler_nn, label_encoder_nn, nn_feature_names = load_nn_files()
            nn_input = pca_df[nn_feature_names]
            nn_input_scaled = scaler_nn.transform(nn_input)
            pred_nn_probs = model_nn.predict(nn_input_scaled, verbose=0)
            pred_nn_encoded = pred_nn_probs.argmax(axis=1)[0]
            pred_nn_label = label_encoder_nn.inverse_transform([int(pred_nn_encoded)])[0]
            prediction_results["Neural Network"] = pred_nn_label
        else:
            prediction_results["Neural Network"] = "Indisponible"

        st.markdown("---")
        section_title("Résultats")

        c1, c2, c3 = st.columns(3)
        models = list(prediction_results.items())

        with c1:
            st.markdown(result_card(models[0][0], models[0][1]), unsafe_allow_html=True)
        with c2:
            st.markdown(result_card(models[1][0], models[1][1]), unsafe_allow_html=True)
        with c3:
            st.markdown(result_card(models[2][0], models[2][1]), unsafe_allow_html=True)

        tab1, tab2, tab3 = st.tabs(["Données", "PCA", "Métriques"])

        with tab1:
            stats_df = pd.DataFrame([input_data]).T.reset_index()
            stats_df.columns = ["Variable", "Valeur"]
            st.dataframe(stats_df, use_container_width=True, hide_index=True)

        with tab2:
            st.dataframe(pca_df, use_container_width=True, hide_index=True)

        with tab3:
            if metrics_file_exist():
                metrics_df = load_metrics_file()
            else:
                metrics_df = pd.DataFrame(
                    {
                        "Modèle": ["Random Forest", "XGBoost", "Neural Network"],
                        "Accuracy": [0.4802, 0.5463, 0.4273],
                        "Precision": [0.4593, 0.5476, 0.4814],
                        "Recall": [0.4802, 0.5463, 0.4273],
                        "F1-score": [0.4631, 0.5417, 0.4150]
                    }
                )

            st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Erreur de prédiction : {e}")
