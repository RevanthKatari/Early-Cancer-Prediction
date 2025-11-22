from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from .config import APP_DESCRIPTION, APP_TITLE, MODEL_HIGHLIGHTS
from .predictors import (
    get_cervical_fieldset,
    infer_brain,
    infer_cervical,
    infer_oral,
    load_cervical_defaults,
)


st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🩺",
    layout="wide",
)


def _inject_styles():
    st.markdown(
        """
        <style>
        .metric-card {
            background: linear-gradient(135deg, #0f172a, #1d4ed8);
            color: #f8fafc;
            padding: 1.2rem;
            border-radius: 1rem;
            box-shadow: 0 12px 30px rgba(15,23,42,0.35);
            min-height: 136px;
        }
        .metric-card h3 {
            margin-bottom: 0.2rem;
            font-size: 1.05rem;
            letter-spacing: .02em;
        }
        .metric-value {
            font-size: 2rem;
            font-weight: 600;
        }
        .metric-caption {
            font-size: 0.9rem;
            opacity: .85;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 1rem;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 0.75rem 1.5rem;
            border-radius: 999px;
            background: rgba(15,23,42,0.06);
        }
        .stTabs [data-baseweb="tab"]:hover {
            background: rgba(59,130,246,0.15);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _hero_section():
    st.title(APP_TITLE)
    st.write(APP_DESCRIPTION)
    cols = st.columns(len(MODEL_HIGHLIGHTS))
    for col, card in zip(cols, MODEL_HIGHLIGHTS):
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <h3>{card['title']}</h3>
                    <div class="metric-value">{card['value']}</div>
                    <div class="metric-caption">{card['metric']} · {card['caption']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _probability_chart(classes, probs, title):
    df = pd.DataFrame({"class": classes, "probability": probs})
    fig = px.bar(
        df,
        x="probability",
        y="class",
        orientation="h",
        text=df["probability"].map(lambda x: f"{x*100:.1f}%"),
        range_x=[0, 1],
        color="probability",
        color_continuous_scale="blues",
        title=title,
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=30, t=40, b=10),
    )
    fig.update_traces(textposition="outside")
    return fig


def _brain_tab():
    st.subheader("Brain MRI classifier")
    st.markdown(
        "Upload a single axial MRI slice (any common format). "
        "The bt-cnn2 model was trained on 180×180 crops from the Kaggle brain tumor dataset."
    )
    uploaded = st.file_uploader("Brain MRI image", type=["jpg", "jpeg", "png"], key="brain_upload")
    if not uploaded:
        st.info("Upload an MRI image to run a prediction.")
        return
    preview, probs, label, confidence = infer_brain(uploaded.getvalue())
    col_img, col_chart = st.columns([1, 2])
    with col_img:
        st.image(preview, caption=f"Uploaded image · Prediction: {label}", use_column_width=True)
        st.metric("Confidence", f"{confidence*100:.1f}%", delta=label.title())
    with col_chart:
        st.plotly_chart(_probability_chart(["Glioma", "Meningioma", "No tumor", "Pituitary"], probs, "Class likelihood"), use_container_width=True)


def _oral_tab():
    st.subheader("Oral cancer screening (Random Forest)")
    st.markdown(
        "Images are resized to 64×64 and flattened before inference, matching the notebook pipeline. "
        "The model currently performs best on well-lit intraoral photographs."
    )
    uploaded = st.file_uploader("Oral cavity photograph", type=["jpg", "jpeg", "png"], key="oral_upload")
    if not uploaded:
        st.info("Upload an oral image to evaluate the random forest classifier.")
        return
    preview, probs, label, confidence = infer_oral(uploaded.getvalue())
    labels = ["Non-cancer", "Cancer"]
    col_img, col_chart = st.columns([1, 2])
    with col_img:
        st.image(preview, caption=f"Prediction: {label}", use_column_width=True)
        st.metric("Cancer probability", f"{confidence*100:.1f}%", delta=label)
    with col_chart:
        st.plotly_chart(_probability_chart(labels, probs, "Probability"), use_container_width=True)


def _cervical_tab():
    st.subheader("Cervical cancer risk factors (Bagging Ensemble)")
    st.markdown(
        "All inputs are scaled with the same StandardScaler used during training. "
        "Toggle between manual entry and CSV batch scoring."
    )
    defaults = load_cervical_defaults()
    fieldset = get_cervical_fieldset()
    sections = defaultdict(list)
    for field in fieldset:
        sections[field.section].append(field)

    with st.form("cervical_form"):
        st.markdown("#### Manual entry")
        payload = {}
        for section, items in sections.items():
            with st.expander(section, expanded=section in {"Demographics", "Diagnostics"}):
                cols = st.columns(2)
                for idx, field in enumerate(items):
                    target_col = cols[idx % 2]
                    with target_col:
                        default_val = defaults.get(field.key, 0.0)
                        if field.input_type == "binary":
                            payload[field.key] = 1.0 if st.toggle(field.label, value=bool(default_val), key=f"{field.key}_toggle") else 0.0
                        else:
                            number_kwargs = {
                                "value": float(default_val),
                                "step": field.step,
                                "key": f"{field.key}_number",
                            }
                            if field.min_value is not None:
                                number_kwargs["min_value"] = float(field.min_value)
                            if field.max_value is not None:
                                number_kwargs["max_value"] = float(field.max_value)
                            payload[field.key] = st.number_input(field.label, **number_kwargs)
        submitted = st.form_submit_button("Predict risk")

    batch_file = st.file_uploader("Optional batch scoring (.csv with the same 33 columns)", type=["csv"], key="cervical_batch")

    if submitted:
        probs, risk = infer_cervical(payload)
        risk_label = "High risk" if risk >= 0.65 else "Moderate risk" if risk >= 0.35 else "Low risk"
        st.success(f"{risk_label} · Probability of biopsy positive: {risk*100:.1f}%")
        st.plotly_chart(_probability_chart(["Negative", "Positive"], probs, "Biopsy probability"), use_container_width=True)

    if batch_file:
        df = pd.read_csv(batch_file)
        missing = [col for col in defaults.keys() if col not in df.columns]
        if missing:
            st.error(f"Missing columns in CSV: {', '.join(missing)}")
        else:
            risks = []
            for _, row in df.iterrows():
                features = defaults.copy()
                for key in features:
                    if key in row and not pd.isna(row[key]):
                        features[key] = float(row[key])
                _, risk = infer_cervical(features)
                risks.append(risk)
            df_result = df.copy()
            df_result["Predicted Biopsy Risk"] = np.array(risks)
            df_result["Risk label"] = pd.cut(
                df_result["Predicted Biopsy Risk"],
                bins=[-0.01, 0.35, 0.65, 1.0],
                labels=["Low", "Moderate", "High"],
            )
            st.dataframe(df_result[["Predicted Biopsy Risk", "Risk label"]])


def main():
    _inject_styles()
    _hero_section()
    brain_tab, cervical_tab, oral_tab = st.tabs(["🧠 Brain MRI", "🧬 Cervical risk", "🦷 Oral screening"])
    with brain_tab:
        _brain_tab()
    with cervical_tab:
        _cervical_tab()
    with oral_tab:
        _oral_tab()


if __name__ == "__main__":
    main()
