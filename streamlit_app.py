#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# --- CONFIG ---
st.set_page_config(page_title="Néo-Banque Dashboard", layout="wide")

# --- LOAD DATA ---
try:
    clients = pd.read_csv("data/clients.csv").reset_index().rename(columns={"index": "client_id"})
except FileNotFoundError:
    st.error("Le fichier 'data/clients.csv' est introuvable dans le dossier 'data'.")
    st.stop()

# --- SIDEBAR FILTERS ---
st.sidebar.title("🔍 Filtrer les clients")
age_filter = st.sidebar.slider("Âge", int(clients.age.min()), int(clients.age.max()), (18, 70))
income_filter = st.sidebar.slider("Revenu annuel (€)", int(clients.revenu.min()), int(clients.revenu.max()), (1000, 100000))
incident_filter = st.sidebar.slider("Nombre d'incidents de paiement", int(clients.nb_incidents.min()), int(clients.nb_incidents.max()), (0, 10))

filtered_clients = clients[
    (clients.age >= age_filter[0]) & (clients.age <= age_filter[1]) &
    (clients.revenu >= income_filter[0]) & (clients.revenu <= income_filter[1]) &
    (clients.nb_incidents >= incident_filter[0]) & (clients.nb_incidents <= incident_filter[1])
]

st.sidebar.markdown(f"**Clients filtrés : {len(filtered_clients)}**")

# --- CLIENT SELECTION ---
st.title("📊 Dashboard de Scoring Crédit")
selected_client_id = st.selectbox("Sélectionner un client", filtered_clients.client_id)

client = filtered_clients[filtered_clients.client_id == selected_client_id].iloc[0]

# --- API CALL ---
API_URL = "https://dashboard-neo-bank.onrender.com/predict"
score = None
shap_factors = []

if st.button("Calculer le score"):
    payload = {
        "age": int(client.age),
        "revenu": float(client.revenu),
        "anciennete": int(client.anciennete),
        "nb_incidents": int(client.nb_incidents),
        "score_credit": float(client.score_credit),
    }
    try:
        r = requests.post(API_URL, json=payload, timeout=10)
        r.raise_for_status()
        resp = r.json()
        score = resp.get("score", None)
        shap_factors = resp.get("explanations", [])
    except Exception as e:
        st.error(f"Erreur API : {e}")

# --- MAIN LAYOUT ---
col_summary, col_details = st.columns([1, 2])

# --- SCORE GAUGE ---
with col_summary:
    st.subheader("Score de Crédit")
    if score is not None:
        gauge_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score*100,
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#4CAF50"},
                'steps': [
                    {'range': [0, 50], 'color': "#F44336"},
                    {'range': [50, 80], 'color': "#FFC107"},
                    {'range': [80, 100], 'color': "#4CAF50"}
                ],
            },
            number={'suffix': "%"}
        ))
        st.plotly_chart(gauge_fig, use_container_width=True)

        if score >= 0.8:
            st.success("✅ Éligible : Score élevé")
        elif score >= 0.5:
            st.warning("⚠️ Potentiellement éligible : Score moyen")
        else:
            st.error("❌ Non éligible : Score faible")
    else:
        st.info("Cliquez sur 'Calculer le score' pour obtenir l'évaluation.")

# --- CLIENT INFO TABLE ---
with col_details:
    st.subheader("Informations Client")
    info_df = pd.DataFrame({
        "Caractéristique": ["ID Client", "Âge", "Revenu", "Ancienneté", "Incidents", "Score crédit initial"],
        "Valeur": [client.client_id, client.age, f"{client.revenu:,.0f} €", client.anciennete, client.nb_incidents, client.score_credit]
    })
    st.table(info_df)

# --- SHAP FACTORS RADAR ---
if score is not None and shap_factors:
    st.subheader("Facteurs d'influence (Explainability)")
    features = [f.split("'")[1] for f in shap_factors]
    contributions = [float(f.split()[-1]) for f in shap_factors]
    # normalize for radar chart
    max_val = max(abs(np.array(contributions))) or 1
    contributions_norm = [c/max_val for c in contributions]

    radar_fig = go.Figure()
    radar_fig.add_trace(go.Scatterpolar(
        r=[abs(v) for v in contributions_norm],
        theta=features,
        fill='toself',
        name='Impact absolu',
        line_color="#1976D2"
    ))
    radar_fig.update_layout(polar=dict(
        radialaxis=dict(visible=True, range=[0,1])
    ), showlegend=False)
    st.plotly_chart(radar_fig, use_container_width=True)

    st.markdown("**Résumé des facteurs :**")
    for f in shap_factors:
        name = f.split("'")[1]
        value = float(f.split()[-1])
        direction = "réduit le risque" if value < 0 else "augmente le risque"
        color = "#4CAF50" if value < 0 else "#F44336"
        st.markdown(f"<span style='color:{color};'>• {name} : {direction} ({abs(value):.2f})</span>", unsafe_allow_html=True)

# --- CLIENT TABLE VIEW ---
st.subheader("Tableau de tous les clients filtrés")
st.dataframe(filtered_clients[['client_id','age','revenu','anciennete','nb_incidents','score_credit']], use_container_width=True)

# --- END ---
