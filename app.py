import pickle
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Attack Risk Classifier",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Global */
  html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; }

  /* Hero banner */
  .hero {
    background: linear-gradient(135deg, #c0392b 0%, #8e44ad 100%);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    color: white;
    margin-bottom: 1.5rem;
  }
  .hero h1 { font-size: 2rem; margin: 0 0 0.4rem 0; }
  .hero p  { font-size: 1rem; margin: 0; opacity: 0.9; }

  /* Section headers */
  .section-title {
    font-size: 1.3rem;
    font-weight: 700;
    color: #2c3e50;
    border-left: 4px solid #c0392b;
    padding-left: 0.6rem;
    margin: 1.5rem 0 1rem 0;
  }

  /* Card wrapper */
  .card {
    background: #ffffff;
    border-radius: 12px;
    padding: 1.4rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.07);
    margin-bottom: 1rem;
  }

  /* Result boxes */
  .result-high {
    background: #fdecea;
    border: 1.5px solid #e74c3c;
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
  }
  .result-low {
    background: #eafaf1;
    border: 1.5px solid #27ae60;
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
  }
  .result-na {
    background: #fef9e7;
    border: 1.5px solid #f39c12;
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
  }
  .result-label { font-size: 0.8rem; font-weight: 600; color: #7f8c8d; margin-bottom: 0.3rem; }
  .result-icon  { font-size: 2rem; }
  .result-text  { font-size: 1rem; font-weight: 700; margin-top: 0.3rem; }

  /* Ensemble banner */
  .ensemble-high {
    background: linear-gradient(90deg, #e74c3c, #c0392b);
    color: white; border-radius: 12px; padding: 1.2rem 1.5rem;
    text-align: center; font-size: 1.2rem; font-weight: 700;
  }
  .ensemble-low {
    background: linear-gradient(90deg, #27ae60, #1e8449);
    color: white; border-radius: 12px; padding: 1.2rem 1.5rem;
    text-align: center; font-size: 1.2rem; font-weight: 700;
  }

  /* Predict button */
  div.stButton > button {
    background: linear-gradient(135deg, #c0392b, #8e44ad);
    color: white; border: none; border-radius: 10px;
    padding: 0.65rem 2rem; font-size: 1rem; font-weight: 600;
    width: 100%; cursor: pointer; transition: opacity 0.2s;
  }
  div.stButton > button:hover { opacity: 0.88; }

  /* Input labels */
  label { font-weight: 600 !important; color: #2c3e50 !important; }

  /* Divider */
  hr { border: none; border-top: 1px solid #ecf0f1; margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)

# ── Load models ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    rf = pickle.load(open("rf_model.pkl", "rb"))
    lstm_model = lstm_scaler = cnn_model = cnn_scaler = None
    try:
        from tensorflow.keras.models import load_model
        lstm_model = load_model("lstm_model.h5")
        with open("lstm_scaler.pkl", "rb") as f:
            lstm_scaler = pickle.load(f)
    except Exception:
        pass
    try:
        from tensorflow.keras.models import load_model
        cnn_model = load_model("cnn_model.h5")
        with open("cnn_scaler.pkl", "rb") as f:
            cnn_scaler = pickle.load(f)
    except Exception:
        pass
    return rf, lstm_model, lstm_scaler, cnn_model, cnn_scaler

rf_model, lstm_model, lstm_scaler, cnn_model, cnn_scaler = load_artifacts()

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>❤️ Heart Attack Risk Classifier</h1>
  <p>Enter patient vitals below. Three ML models (Random Forest, LSTM, CNN) predict risk simultaneously.</p>
</div>
""", unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🩺 Prediction", "📊 Analytics & Insights"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-title">Patient Information</div>', unsafe_allow_html=True)

    with st.form("patient_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**Vitals**")
            Age          = st.number_input("Age", min_value=20, max_value=100, value=50)
            RestingBP    = st.number_input("Resting BP (mm Hg)", min_value=0, max_value=300, value=120)
            Cholesterol  = st.number_input("Cholesterol (mg/dl)", min_value=0, max_value=600, value=200)
            MaxHR        = st.number_input("Max Heart Rate", min_value=60, max_value=600, value=150)

        with c2:
            st.markdown("**Clinical**")
            Oldpeak        = st.number_input("Oldpeak", min_value=-3.0, max_value=10.0, value=1.0, step=0.1)
            FastingBS      = st.selectbox("Fasting Blood Sugar > 120", (0, 1))
            gender         = st.selectbox("Sex", ("M", "F"))

        with c3:
            st.markdown("**Categorical**")
            ChestPainType  = st.selectbox("Chest Pain Type", ("ATA", "NAP", "ASY", "TA"))
            RestingECG     = st.selectbox("Resting ECG", ("Normal", "ST", "LVH"))
            ExerciseAngina = st.selectbox("Exercise Angina", ("N", "Y"))
            ST_Slope       = st.selectbox("ST Slope", ("Up", "Flat", "Down"))

        submitted = st.form_submit_button("🔍 Predict Risk", use_container_width=True)

    # ── Encoding helpers ───────────────────────────────────────────────────────
    def encode_rf(Age, RestingBP, Cholesterol, FastingBS, MaxHR, Oldpeak,
                  ExerciseAngina, gender, ChestPainType, RestingECG, ST_Slope):
        df = pd.DataFrame({
            "Age": [Age], "RestingBP": [RestingBP], "Cholesterol": [Cholesterol],
            "FastingBS": [FastingBS], "MaxHR": [MaxHR], "Oldpeak": [Oldpeak],
            "Exercise_Angina": [1 if ExerciseAngina == "Y" else 0],
            "Sex_F": [1 if gender == "F" else 0],
            "Sex_M": [1 if gender == "M" else 0],
            "Chest_PainType": [{"ASY": 3, "NAP": 2, "ATA": 1, "TA": 0}[ChestPainType]],
            "Resting_ECG":    [{"Normal": 0, "LVH": 1, "ST": 2}[RestingECG]],
            "st_Slope":       [{"Down": 0, "Up": 1, "Flat": 2}[ST_Slope]],
        })
        sc = StandardScaler()
        df[["Age", "RestingBP", "Cholesterol", "MaxHR"]] = sc.fit_transform(
            df[["Age", "RestingBP", "Cholesterol", "MaxHR"]])
        return df

    def encode_dl(Age, RestingBP, Cholesterol, FastingBS, MaxHR, Oldpeak,
                  ExerciseAngina, gender, ChestPainType, RestingECG, ST_Slope, scaler):
        row = np.array([[Age,
                         {"M": 1, "F": 0}[gender],
                         {"ATA": 0, "NAP": 2, "ASY": 1, "TA": 3}[ChestPainType],
                         RestingBP, Cholesterol, FastingBS,
                         {"LVH": 0, "Normal": 1, "ST": 2}[RestingECG],
                         MaxHR,
                         {"N": 0, "Y": 1}[ExerciseAngina],
                         Oldpeak,
                         {"Down": 0, "Flat": 1, "Up": 2}[ST_Slope]]], dtype=np.float32)
        return scaler.transform(row)

    def result_card(pred, label, available=True):
        if not available:
            return f"""<div class="result-na">
              <div class="result-label">{label}</div>
              <div class="result-icon">⚙️</div>
              <div class="result-text">Model not trained</div>
            </div>"""
        if pred == 1:
            return f"""<div class="result-high">
              <div class="result-label">{label}</div>
              <div class="result-icon">⚠️</div>
              <div class="result-text" style="color:#c0392b;">High Risk</div>
            </div>"""
        return f"""<div class="result-low">
          <div class="result-label">{label}</div>
          <div class="result-icon">✅</div>
          <div class="result-text" style="color:#27ae60;">Low Risk</div>
        </div>"""

    # ── Run predictions ────────────────────────────────────────────────────────
    if submitted:
        st.markdown('<div class="section-title">Model Predictions</div>', unsafe_allow_html=True)

        # RF
        rf_pred = int(rf_model.predict(
            encode_rf(Age, RestingBP, Cholesterol, FastingBS, MaxHR, Oldpeak,
                      ExerciseAngina, gender, ChestPainType, RestingECG, ST_Slope))[0])

        # LSTM
        lstm_pred = None
        if lstm_model and lstm_scaler:
            x = encode_dl(Age, RestingBP, Cholesterol, FastingBS, MaxHR, Oldpeak,
                          ExerciseAngina, gender, ChestPainType, RestingECG, ST_Slope, lstm_scaler)
            lstm_pred = int(lstm_model.predict(x.reshape(1, 1, x.shape[1]), verbose=0)[0][0] >= 0.5)

        # CNN
        cnn_pred = None
        if cnn_model and cnn_scaler:
            x = encode_dl(Age, RestingBP, Cholesterol, FastingBS, MaxHR, Oldpeak,
                          ExerciseAngina, gender, ChestPainType, RestingECG, ST_Slope, cnn_scaler)
            cnn_pred = int(cnn_model.predict(x.reshape(1, x.shape[1], 1), verbose=0)[0][0] >= 0.5)

        col_rf, col_lstm, col_cnn = st.columns(3)
        with col_rf:
            st.markdown(result_card(rf_pred, "Random Forest"), unsafe_allow_html=True)
        with col_lstm:
            st.markdown(result_card(lstm_pred, "LSTM", lstm_pred is not None), unsafe_allow_html=True)
        with col_cnn:
            st.markdown(result_card(cnn_pred, "CNN", cnn_pred is not None), unsafe_allow_html=True)

        # Ensemble
        st.markdown("<br>", unsafe_allow_html=True)
        votes = [p for p in [rf_pred, lstm_pred, cnn_pred] if p is not None]
        majority = int(sum(votes) > len(votes) / 2)
        cls = "ensemble-high" if majority == 1 else "ensemble-low"
        icon = "⚠️ High Risk of Heart Attack" if majority == 1 else "✅ Low Risk of Heart Attack"
        st.markdown(f'<div class="{cls}">Ensemble Verdict &nbsp;|&nbsp; {icon}</div>',
                    unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ANALYTICS & INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:

    # ── Feature Importance ─────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Feature Importance</div>', unsafe_allow_html=True)

    feat_names = ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR",
                  "Oldpeak", "Exercise_Angina", "Sex_F", "Sex_M",
                  "Chest_PainType", "Resting_ECG", "st_Slope"]
    importances = rf_model.feature_importances_
    fi_df = pd.DataFrame({"Feature": feat_names, "Importance": importances})
    fi_df = fi_df.sort_values("Importance")

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.RdPu(np.linspace(0.3, 0.9, len(fi_df)))
    ax.barh(fi_df["Feature"], fi_df["Importance"], color=colors)
    ax.set_xlabel("Importance Score", fontsize=10)
    ax.set_title("Random Forest Feature Importance", fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── EDA ────────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Exploratory Data Analysis (EDA)</div>',
                unsafe_allow_html=True)
    st.caption("Upload a CSV file for visualization")

    uploaded = st.file_uploader("", type=["csv"], label_visibility="collapsed")

    if uploaded is None:
        st.info("Upload a CSV file to explore visualizations.")
    else:
        eda_df = pd.read_csv(uploaded)
        st.success(f"Loaded {eda_df.shape[0]} rows × {eda_df.shape[1]} columns")

        st.markdown("**Preview**")
        st.dataframe(eda_df.head(10), use_container_width=True)

        num_cols = eda_df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = eda_df.select_dtypes(exclude=np.number).columns.tolist()

        if num_cols:
            st.markdown("**Numeric Summary**")
            st.dataframe(eda_df[num_cols].describe().T.style.background_gradient(cmap="RdPu"),
                         use_container_width=True)

            st.markdown("**Distribution**")
            sel_col = st.selectbox("Select column", num_cols, key="dist_col")
            fig2, ax2 = plt.subplots(figsize=(7, 3))
            ax2.hist(eda_df[sel_col].dropna(), bins=30, color="#c0392b", edgecolor="white", alpha=0.85)
            ax2.set_title(f"Distribution of {sel_col}", fontweight="bold")
            ax2.spines[["top", "right"]].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()

        if len(num_cols) >= 2:
            st.markdown("**Correlation Heatmap**")
            fig3, ax3 = plt.subplots(figsize=(8, 5))
            import seaborn as sns
            sns.heatmap(eda_df[num_cols].corr(), annot=True, fmt=".2f",
                        cmap="RdPu", ax=ax3, linewidths=0.5)
            ax3.set_title("Correlation Matrix", fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig3)
            plt.close()

        if cat_cols:
            st.markdown("**Categorical Column Counts**")
            sel_cat = st.selectbox("Select column", cat_cols, key="cat_col")
            fig4, ax4 = plt.subplots(figsize=(6, 3))
            eda_df[sel_cat].value_counts().plot(kind="bar", ax=ax4, color="#8e44ad", edgecolor="white")
            ax4.set_title(f"Value Counts — {sel_cat}", fontweight="bold")
            ax4.spines[["top", "right"]].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig4)
            plt.close()
