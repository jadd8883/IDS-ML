
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_PATH = Path("models/ids_model.joblib")

st.set_page_config(
    page_title="ML-Based Intrusion Detection System",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ ML-Based Intrusion Detection System")
st.write(
    "Upload labeled-schema network-flow data to identify "
    "records that the trained model considers suspicious."
)

if not MODEL_PATH.exists():
    st.error("Model not found. Run python train.py first.")
    st.stop()

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
features = bundle["features"]

uploaded_file = st.file_uploader(
    "Upload a network-flow CSV",
    type=["csv"],
)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, low_memory=False)
        df.columns = df.columns.str.strip()

        # Use only the features used during training.
        missing = [col for col in features if col not in df.columns]

        if missing:
            st.error(
                "Missing required model features: "
                + ", ".join(missing[:15])
            )
            st.stop()

        X = df[features].copy()

        # Handle malformed or infinite numeric values.
        for col in features:
            X[col] = pd.to_numeric(X[col], errors="coerce")

        X = X.replace([np.inf, -np.inf], np.nan)

        # Model pipeline handles missing values.
        predictions = model.predict(X)

        results = df.copy()
        results["IDS_Prediction"] = np.where(
            predictions == 1,
            "Potential Attack",
            "Benign",
        )

        # Probability is useful when supported by the classifier.
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X)
            attack_index = list(model.classes_).index(1)
            results["Attack_Score"] = probabilities[:, attack_index]

        total = len(results)
        attacks = int((predictions == 1).sum())
        benign = total - attacks

        col1, col2, col3 = st.columns(3)
        col1.metric("Records analyzed", f"{total:,}")
        col2.metric("Potential attacks", f"{attacks:,}")
        col3.metric(
            "Flagged percentage",
            f"{attacks / total * 100:.2f}%" if total else "0%",
        )

        st.subheader("Prediction distribution")
        st.bar_chart(
            pd.Series(
                {"Benign": benign, "Potential Attack": attacks},
                name="Records",
            )
        )

        st.subheader("Potentially suspicious records")
        st.dataframe(
            results[results["IDS_Prediction"] == "Potential Attack"],
            use_container_width=True,
        )

        st.subheader("All predictions")
        st.dataframe(results, use_container_width=True)

        st.download_button(
            "Download prediction results",
            data=results.to_csv(index=False).encode("utf-8"),
            file_name="ids_predictions.csv",
            mime="text/csv",
        )

        st.caption(
            "Predictions are model estimates, not proof of an attack. "
            "Validate alerts with additional security evidence."
        )

    except Exception as exc:
        st.error(f"Could not process the CSV: {exc}")
else:
    st.info(
        "Upload a CSV containing the same feature columns "
        "used during model training."
    )
