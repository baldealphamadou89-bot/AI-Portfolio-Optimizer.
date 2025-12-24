import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. CONFIGURATION DE LA PAGE
st.set_page_config(page_title="Alpha Balde | FinTech Optimizer", page_icon="🚀", layout="wide")

# --- EN-TÊTE : LOGO ET LANGUE ---
col_logo, col_empty, col_lang = st.columns([1, 2, 1])

with col_logo:
    try:
        st.image("logo.png", width=100)
    except:
        st.subheader("🚀 FinTeach AI")

with col_lang:
    lang = st.selectbox("🌐 Language", ("English", "Français"), label_visibility="collapsed")

# 2. DICTIONNAIRE DES TEXTES
textes = {
    "English": {
        "title": "Portfolio Risk Engine",
        "intro": "Developed by **Alpha Balde**",
        "input_label": "Modify your portfolio tickers here:",
        "loading": "Fetching data...",
        "error_data": "Error: Invalid tickers found.",
        "price_chart": "📈 Price Evolution (Base 100)",
        "risk_chart": "⚠️ Risk Analysis",
        "sim_title": "🎯 Optimal Allocation (Monte Carlo)",
        "ideal": "Recommended Strategy:",
        "gain": "Expected Return",
        "risk": "Risk Level"
    },
    "Français": {
        "title": "Optimiseur de Portefeuille",
        "intro": "Développé par **Alpha Balde**",
        "input_label": "Modifiez vos symboles ici :",
        "loading": "Analyse en cours...",
        "error_data": "Erreur : Symboles invalides.",
        "price_chart": "📈 Évolution des prix (Base 100)",
        "risk_chart": "⚠️ Analyse du Risque",
        "sim_title": "🎯 Allocation Optimale (Monte Carlo)",
        "ideal": "Stratégie Recommandée :",
        "gain": "Gain Attendu",
        "risk": "Niveau de Risque"
    }
}

t = textes[lang]

# 3. AFFICHAGE DU CONTENU
st.title(t["title"])
st.caption(t["intro"])
st.divider()

# On définit les tickers par défaut au cas où
default_tickers = "AAPL, TSLA, NVDA, BAC"

# --- ÉTAPE 2 : ANALYSE (On place les calculs avant l'affichage) ---
# Nous utilisons un container pour pouvoir placer la saisie TOUT EN BAS plus tard
main_container = st.container()

# --- ÉTAPE 3 : LA BARRE DE SAISIE (PLACÉE EN BAS VISUELLEMENT) ---
st.write("---") # Séparateur visuel
input_tickers = st.text_input(f"👇 {t['input_label']}", value="AAPL, TSLA, NVDA, BAC")
tickers = [tick.strip().upper() for tick in input_tickers.split(",") if tick.strip()]

with main_container:
    if len(tickers) >= 2:
        with st.spinner(t["loading"]):
            data = yf.download(tickers, start="2021-01-01")['Close']
            data = data.dropna(axis=1, how='all')

        if not data.empty and len(data.columns) >= 2:
            # Graphiques
            tab1, tab2 = st.tabs([t["price_chart"], t["sim_title"]])
            
            with tab1:
                data_norm = (data / data.iloc[0]) * 100
                st.line_chart(data_norm)
                
                returns = data.pct_change().dropna()
                st.subheader(t["risk_chart"])
                volatility = returns.std() * np.sqrt(252) * 100
                st.bar_chart(volatility)

            with tab2:
                # Simulation simplifiée pour la démo
                num_portfolios = 1000
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
                
                st.success(f"✅ {t['ideal']}")
                cols = st.columns(len(tickers))
                for i, tick in enumerate(tickers):
                    cols[i].metric(tick, f"{all_weights[max_sr_idx, i]*100:.1f}%")
                
                fig, ax = plt.subplots()
                ax.scatter(vol_arr, ret_arr, c=sharpe_arr, cmap='viridis', alpha=0.5)
                ax.scatter(vol_arr[max_sr_idx], ret_arr[max_sr_idx], c='red', s=50)
                st.pyplot(fig)
        else:
            st.error(t["error_data"])
    else:
        st.info("💡 Enter at least 2 tickers to start.")
