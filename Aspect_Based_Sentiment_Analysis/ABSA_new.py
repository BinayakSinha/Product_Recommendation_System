# ==============================================================
# 🌈 Universal ABSA Dashboard — User-defined Aspects
# - Better visuals, user-defined aspects, deeper analysis
# - Works across domains, not just electronics
# ==============================================================

import os
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"

import streamlit as st
import pandas as pd
import numpy as np
import re
from transformers import pipeline
import plotly.express as px
import plotly.graph_objects as go
from sentence_transformers import SentenceTransformer, util

# -------------------------------------------------------------
#  Streamlit setup
# -------------------------------------------------------------
st.set_page_config(page_title="💬 Smart ABSA Dashboard", layout="wide", page_icon="🌈")
st.title("Aspect-Based Sentiment Analyzer")
st.caption("Analyze **any user-defined aspects** with colorful insights, charts, and summaries.")

# -------------------------------------------------------------
# Sidebar Configuration
# -------------------------------------------------------------
st.sidebar.header("⚙️ Settings")
use_gpu = st.sidebar.checkbox("Use GPU", value=False)
device_index = 0 if use_gpu else -1
soft_weighted_scores = st.sidebar.checkbox("Weighted scoring (confidence aware)", True)
normalize_aspects = st.sidebar.checkbox("Use semantic normalization", False)

# -------------------------------------------------------------
# Load Model
# -------------------------------------------------------------
@st.cache_resource
def load_models(device):
    absa_model = pipeline(
        "text-classification",
        model="yangheng/deberta-v3-base-absa-v1.1",
        tokenizer="yangheng/deberta-v3-base-absa-v1.1",
        device=device
    )
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return absa_model, embed_model

classifier, embedder = load_models(device_index)
st.sidebar.success("✅ Models ready!")

# -------------------------------------------------------------
# Utilities
# -------------------------------------------------------------
def map_label(label):
    s = label.lower()
    if "pos" in s: return "Positive"
    if "neg" in s: return "Negative"
    return "Neutral"

def decide_label(probs):
    if probs["Positive"] > probs["Negative"] + 0.1: return "Positive"
    elif probs["Negative"] > probs["Positive"] + 0.1: return "Negative"
    return "Neutral"

def parse_output(raw):
    if isinstance(raw, list) and len(raw) > 0 and isinstance(raw[0], list): raw = raw[0]
    probs = {"Positive": 0, "Neutral": 0, "Negative": 0}
    for r in raw:
        probs[map_label(r["label"])] += float(r["score"])
    total = sum(probs.values()) or 1
    return {k: v/total for k,v in probs.items()}

def normalize_aspect(aspect, aspects):
    if not normalize_aspects or not embedder:
        return aspect
    emb = embedder.encode(aspect, convert_to_tensor=True)
    sims = {a: float(util.cos_sim(emb, embedder.encode(a, convert_to_tensor=True)))} 
    best = max(sims, key=sims.get)
    return best if sims[best] > 0.45 else aspect

def analyze_review(review, aspects):
    rows = []
    for asp in aspects:
        try:
            raw = classifier(review, text_pair=asp, return_all_scores=True, truncation=True, padding=True)
            probs = parse_output(raw)
            label = decide_label(probs)
            score = 5 + 5*(probs["Positive"] - probs["Negative"])
            if soft_weighted_scores:
                score *= (0.8 + 0.2 * max(probs.values()))
            rows.append({
                "Aspect": asp,
                "Sentiment": label,
                "Positive": round(probs["Positive"],3),
                "Neutral": round(probs["Neutral"],3),
                "Negative": round(probs["Negative"],3),
                "Score (1–10)": round(np.clip(score,1,10),2),
                "Confidence (%)": round(max(probs.values())*100,1)
            })
        except Exception:
            pass
    return pd.DataFrame(rows)

# -------------------------------------------------------------
# Visuals
# -------------------------------------------------------------
def plot_bar(df):
    fig = px.bar(
        df, x="Aspect", y="Score (1–10)", color="Sentiment",
        text="Score (1–10)",
        color_discrete_map={"Positive":"#00CC96","Neutral":"#636EFA","Negative":"#EF553B"},
        title="Aspect-wise Sentiment Scores"
    )
    fig.update_traces(textposition="outside", hovertemplate="<b>%{x}</b><br>Score: %{y}<extra></extra>")
    fig.update_layout(template="seaborn", yaxis=dict(range=[0,10]))
    st.plotly_chart(fig, use_container_width=True)

def plot_pie(df):
    counts = df["Sentiment"].value_counts().reset_index()
    counts.columns = ["Sentiment","Count"]
    fig = px.pie(counts, names="Sentiment", values="Count", hole=0.4,
                 color="Sentiment",
                 color_discrete_map={"Positive":"#00CC96","Neutral":"#636EFA","Negative":"#EF553B"},
                 title="Overall Sentiment Breakdown")
    fig.update_traces(textinfo="percent+label")
    st.plotly_chart(fig, use_container_width=True)

def plot_radar(df):
    if len(df) < 3: return
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=df["Score (1–10)"], theta=df["Aspect"], fill="toself",
        line_color="#1482E9", name="Aspect Score"
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0,10])),
        template="seaborn", title="Aspect Strength Radar"
    )
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------------------
# Insights
# -------------------------------------------------------------
def show_insights(df):
    avg = round(df["Score (1–10)"].mean(),2)
    pos = df[df["Sentiment"]=="Positive"]["Aspect"].tolist()
    neg = df[df["Sentiment"]=="Negative"]["Aspect"].tolist()
    conf = round(df["Confidence (%)"].mean(),1)

    st.markdown(f"### 🌟 Average Sentiment Score: `{avg}/10`")
    st.markdown(f"🧩 **High-confidence average:** `{conf}%`")
    if pos: st.success(f"✅ Strengths: {', '.join(pos)}")
    if neg: st.error(f"⚠️ Weak Points: {', '.join(neg)}")

# -------------------------------------------------------------
# 🧩 Input UI
# -------------------------------------------------------------
st.sidebar.header("🧠 Define Your Aspects")
custom_input = st.sidebar.text_area("Enter aspects (comma/newline separated):", "battery, screen, performance, camera, price", height=100)
aspects = [x.strip() for x in re.split(r'[,\n]+', custom_input) if x.strip()]

st.subheader("📝 Single Review Analysis")
review = st.text_area("Paste a review here:", "The display is stunning, battery lasts long, but it’s slightly overpriced.", height=150)

if st.button("🔍 Analyze"):
    if not review.strip():
        st.warning("Enter a review first.")
    elif not aspects:
        st.warning("Please define at least one aspect.")
    else:
        with st.spinner("Analyzing aspects..."):
            df = analyze_review(review, aspects)
        if df.empty:
            st.error("No sentiments could be extracted.")
        else:
            st.success("✅ Analysis Complete")
            st.dataframe(df)
            c1, c2 = st.columns(2)
            with c1: plot_pie(df)
            with c2: plot_bar(df)
            plot_radar(df)
            show_insights(df)

st.markdown("---")
st.caption("💡 Tip: You can use your own domain aspects (e.g., 'delivery', 'packaging', 'durability'). The system will adapt dynamically.")
