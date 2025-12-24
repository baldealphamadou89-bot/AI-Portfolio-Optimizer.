import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. CONFIGURATION DE LA LANGUE
st.sidebar.title("Language / Langue")
lang = st.sidebar.radio("Select Language:", ("English", "Français"))

# Dictionnaire des textes
textes = {
    "English": {
        "title": "🚀 FinTech Portfolio Optimizer",
        "intro": "This AI-driven tool optimizes your asset allocation using statistics.",
        "config": "Configuration",
        "input": "Enter tickers separated by commas (ex: AAPL, NVDA, TSLA):",
        "error_min": "Please select at least 2 stocks to compare.",
        "loading": "Fetching market data...",
        "error_data": "Error: Could not find these tickers. Check spelling.",
        "price_chart": "📈 Price Evolution (Base 100)",
        "risk_chart": "⚠️ Risk Analysis (Volatility)",
        "sim_title": "🎯 Best Mix Simulation (Monte Carlo)",
        "success": "Best mix found to maximize returns vs risk:",
        "ideal": "**Ideal Allocation:**",
        "perf": "**Expected Performance:**",
        "gain": "Annual Return",
        "risk": "Risk Level",
        "label_x": "Risk (Volatility)",
        "label_y": "Return (Gains)",
        "sharpe": "Quality Score (Sharpe Ratio)"
    },
    "Français": {
        "title": "🚀 Optimiseur de Portefeuille FinTech",
        "intro": "Cet outil utilise l'IA et les stats pour optimiser ton argent.",
        "config": "Configuration",
        "input": "Entre les symboles séparés par une virgule (ex: AAPL, NVDA, TSLA) :",
        "error_min": "Choisis au moins 2 actions pour comparer.",
        "loading": "Récupération des données boursières...",
        "error_data": "Erreur : Symboles introuvables. Vérifie l'orthographe.",
        "price_chart": "📈 Évolution des prix (Base 100)",
        "risk_chart": "⚠️ Analyse du Danger (Volatilité)",
        "sim_title": "🎯 Simulation du Meilleur Mélange (Monte Carlo)",
        "success": "Meilleur mélange trouvé pour maximiser les gains vs risque :",
        "ideal": "**Répartition idéale :**",
        "perf": "**Performance attendue :**",
        "gain": "Gain annuel",
        "risk": "Niveau de risque",
        "label_x": "Risque (Volatilité)",
        "label_y": "Gain (Rendement)",
        "sharpe": "Score de Qualité (Ratio de Sharpe)"
    }
}

t = textes[lang]

# 2. INTERFACE
st.title(t["title"])
st.write(t["intro"])

# Sidebar
st.sidebar.header(t["config"])
input_tickers = st.sidebar.text_input(t["input"], "AAPL, TSLA, NVDA, DIS")
tickers = [tick.strip().upper() for tick in input_tickers.split(",")]

if len(tickers) < 2:
    st.warning(t["error_min"])
else:
    with st.spinner(t["loading"]):
        data = yf.download(tickers, start="2021-01-01")['Close']
    
    if data.empty:
        st.error(t["error_data"])
    else:
        # Graphique des prix
        st.subheader(t["price_chart"])
        data_norm = (data / data.iloc[0]) * 100
        st.line_chart(data_norm)

        # Analyse du risque
        st.subheader(t["risk_chart"])
        returns = data.pct_change().dropna()
        volatility = returns.std() * np.sqrt(252) * 100
        st.bar_chart(volatility)

        # Simulation
        st.subheader(t["sim_title"])
        num_portfolios = 2000
        all_weights = np.zeros((num_portfolios, len(tickers)))
        ret_arr = np.zeros(num_portfolios)
        vol_arr = np.zeros(num_portfolios)
        sharpe_arr = np.zeros(num_portfolios)

        for ind in range(num_portfolios):
            weights = np.array(np.random.random(len(tickers)))
            weights /= np.sum(weights)
            all_weights[ind,:] = weights
            ret_arr[ind] = np.sum((returns.mean() * weights) * 252)
            vol_arr[ind] = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
            sharpe_arr[ind] = ret_arr[ind] / vol_arr[ind]

        max_sr_idx = sharpe_arr.argmax()
        best_weights = all_weights[max_sr_idx,:]

        st.success(t["success"])
        c1, c2 = st.columns(2)
        with c1:
            st.write(t["ideal"])
            for i, ticker in enumerate(tickers):
                st.write(f"- {ticker} : {best_weights[i]*100:.1f}%")
        with c2:
            st.write(t["perf"])
            st.write(f"{t['gain']} : {ret_arr[max_sr_idx]*100:.1f}%")
            st.write(f"{t['risk']} : {vol_arr[max_sr_idx]*100:.1f}%")

        # Nuage de points
        fig, ax = plt.subplots()
        scatter = ax.scatter(vol_arr, ret_arr, c=sharpe_arr, cmap='viridis')
        ax.scatter(vol_arr[max_sr_idx], ret_arr[max_sr_idx], c='red', s=50)
        ax.set_xlabel(t["label_x"])
        ax.set_ylabel(t["label_y"])
        plt.colorbar(scatter, label=t["sharpe"])
        st.pyplot(fig)

st.sidebar.markdown("---")
st.sidebar.info("Projet FinTeach - Portfolio Optimizer")
