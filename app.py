import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 1. CONFIGURATION DE LA PAGE
st.set_page_config(page_title="Alpha Balde | FinTech Optimizer", page_icon="🚀", layout="centered")

# --- EN-TÊTE : LOGO ET LANGUE ---
col_logo, col_lang = st.columns([3, 1])

with col_logo:
    # Recherche automatique du logo (png ou jpg)
    logo_path = "logo.png" if os.path.exists("logo.png") else "logo.jpg"
    if os.path.exists(logo_path):
        st.image(logo_path, width=120)
    else:
        st.subheader("🚀 Autonomous Portfolio Optimizer & Risk Engine AI")

with col_lang:
    lang = st.selectbox("🌐", ("English", "Français"))

# 2. DICTIONNAIRE DES TEXTES
textes = {
    "English": {
        "title": "Portfolio Risk Engine",
        "intro": "Developed by **Alpha Balde**",
        "input_label": "Enter tickers (comma separated):",
        "btn_label": "Analyze & Optimize 🎯",
        "loading": "Simulating 1,500 portfolios...",
        "error_data": "Error: Invalid tickers or not enough data.",
        "price_chart": "📈 Price Evolution (Base 100)",
        "sim_title": "🎯 Expected Performance",
        "ideal": "Recommended Allocation:",
        "gain": "Annual Return",
        "risk": "Risk Level"
    },
    "Français": {
        "title": "Optimiseur de Portefeuille",
        "intro": "Développé par **Alpha Balde**",
        "input_label": "Entrez les symboles (ex: AAPL, BTC-USD) :",
        "btn_label": "Analyser & Optimiser 🎯",
        "loading": "Simulation de 1 500 portefeuilles...",
        "error_data": "Erreur : Symboles invalides ou données insuffisantes.",
        "price_chart": "📈 Évolution des prix (Base 100)",
        "sim_title": "🎯 Performance Attendue",
        "ideal": "Répartition Recommandée :",
        "gain": "Gain annuel",
        "risk": "Niveau de risque"
    }
}

t = textes[lang]

# 3. INTERFACE PRINCIPALE
st.title(t["title"])
st.caption(t["intro"])

# --- ZONE DE SAISIE ET BOUTON ---
st.divider()
input_tickers = st.text_input(t["input_label"], value="AAPL, TSLA, NVDA, BAC")
analyze_btn = st.button(t["btn_label"], type="primary", use_container_width=True)

# L'analyse ne se lance QUE si on clique sur le bouton
if analyze_btn:
    tickers = [tick.strip().upper() for tick in input_tickers.split(",") if tick.strip()]
    
    if len(tickers) >= 2:
        with st.spinner(t["loading"]):
            data = yf.download(tickers, start="2021-01-01")['Close']
            data = data.dropna(axis=1, how='all')

        if not data.empty and len(data.columns) >= 2:
            # --- 1. GRAPHIQUE DES PRIX ---
            st.subheader(t["price_chart"])
            data_norm = (data / data.iloc[0]) * 100
            st.line_chart(data_norm)

            # --- 2. CALCULS STATISTIQUES ---
            returns = data.pct_change().dropna()
            num_portfolios = 1500
            all_weights = np.zeros((num_portfolios, len(tickers)))
            ret_arr = np.zeros(num_portfolios)
            vol_arr = np.zeros(num_portfolios)
            
            for ind in range(num_portfolios):
                weights = np.array(np.random.random(len(tickers)))
                weights /= np.sum(weights)
                all_weights[ind,:] = weights
                ret_arr[ind] = np.sum((returns.mean() * weights) * 252)
                vol_arr[ind] = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
            
            sharpe_arr = ret_arr / vol_arr
            max_sr_idx = sharpe_arr.argmax()

            # --- 3. AFFICHAGE DES RÉSULTATS (POURCENTAGES) ---
            st.divider()
            st.subheader(t["sim_title"])
            
            # Affichage "Metric" pour le Gain et le Risque
            m1, m2 = st.columns(2)
            m1.metric(t["gain"], f"{ret_arr[max_sr_idx]*100:.1f}%")
            m2.metric(t["risk"], f"{vol_arr[max_sr_idx]*100:.1f}%")

            st.write(f"### {t['ideal']}")
            # Affichage des poids des actions
            cols = st.columns(len(tickers))
            for i, tick in enumerate(tickers):
                with cols[i]:
                    st.markdown(f"**{tick}**")
                    st.markdown(f"## {all_weights[max_sr_idx, i]*100:.1f}%")

            # --- 4. LE GRAPHIQUE DE MONTE CARLO ---
            fig, ax = plt.subplots(figsize=(10, 6))
            plt.style.use('dark_background')
            scatter = ax.scatter(vol_arr, ret_arr, c=sharpe_arr, cmap='viridis', alpha=0.4)
            ax.scatter(vol_arr[max_sr_idx], ret_arr[max_sr_idx], c='red', s=150, edgecolors='white', label="Best")
            ax.set_xlabel("Volatility (Risk)")
            ax.set_ylabel("Return (Gains)")
            st.pyplot(fig)
        else:
            st.error(t["error_data"])
    else:
        st.warning("⚠️ Please enter at least 2 tickers.")
else:
    st.info("👆 Enter your tickers and click the button to start the analysis.")

