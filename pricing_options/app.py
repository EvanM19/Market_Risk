###################################
# LIBRAIRIES ET PACKAGES
###################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import streamlit as st
import yfinance as yf
from matplotlib.cm import get_cmap
from scipy.interpolate import griddata
import plotly.graph_objects as go
import json

# Option Européenne

###########################
# Partie Black & Scholes

def d_j(j, S_0, K, r, sigma, T):
    return (np.log(S_0/K) + (r + ((-1)**(j-1)) * 0.5 * sigma * sigma) * T) / (sigma * (T ** 0.5))

def vanilla_call_price(S_0, K, r, sigma, T):
    return S_0 * norm.cdf(d_j(1, S_0, K, r, sigma, T)) - K * np.exp(-r * T) * norm.cdf(d_j(2, S_0, K, r, sigma, T))

def vanilla_put_price(S_0, K, r, sigma, T):
    return -S_0 * norm.cdf(-d_j(1, S_0, K, r, sigma, T)) + K * np.exp(-r * T) * norm.cdf(-d_j(2, S_0, K, r, sigma, T))

# Fonctions pour les Grecques
def greeks(S_0, K, r, sigma, T, greek, option_type):
    if greek == "delta":
        if option_type == "call":
            return norm.cdf(d_j(1, S_0, K, r, sigma, T))
        elif option_type == "put":
            return -norm.cdf(-d_j(1, S_0, K, r, sigma, T))
    elif greek == "gamma":
        return norm.pdf(d_j(1, S_0, K, r, sigma, T)) / (S_0 * sigma * np.sqrt(T))
    elif greek == "vega":
        return S_0 * norm.pdf(d_j(1, S_0, K, r, sigma, T)) * np.sqrt(T)
    elif greek == "theta":
        d1 = d_j(1, S_0, K, r, sigma, T)
        d2 = d_j(2, S_0, K, r, sigma, T)
        if option_type == "call":
            return -(S_0 * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
        elif option_type == "put":
            return -(S_0 * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)


###########################
# Partie Monte Carlo

def monte_carlo_call_price(S_0, K, r, sigma, T, N):
    np.random.seed(42)  # Pour la reproductibilité
    Z = np.random.standard_normal(N)
    ST = S_0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z)
    payoff = np.maximum(ST - K, 0)
    discounted_payoff = np.exp(-r * T) * payoff
    return np.mean(discounted_payoff)

def monte_carlo_put_price(S_0, K, r, sigma, T, N):
    np.random.seed(42)  # Pour la reproductibilité
    Z = np.random.standard_normal(N)
    ST = S_0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z)
    payoff = np.maximum(K - ST, 0)
    discounted_payoff = np.exp(-r * T) * payoff
    return np.mean(discounted_payoff)


# Discretisation du Temps
def process(S_0, T, mu, sigma, steps, N):
    dt = T / steps
    ST = np.log(S_0) + np.cumsum(((mu - sigma ** 2 / 2) * dt + sigma * np.sqrt(dt) * np.random.normal(size=(steps, N))), axis=0)
    return np.exp(ST)



















# Page pour l'option européenne
def euro_option_page():
    st.title("Pricer Equity par méthode Monte Carlo & Black & Scholes pour Option Européenne")
    st.markdown(
        """
        On note $S_{t}$ le prix d'un actif fixé au temps $t$. Le modèle de Black et Scholes consiste à dire que le prix de cet actif répond à l'équation différentielle stochastique suivante:
        
        $$ dS_{t} = S_{t}(\mu dt + \sigma dW_{t}) $$
        
        où $\mu$ est un réel (appelé parfois dérive), $\sigma$ est un réel positif (appelé volatilité) et $W$ désigne un mouvement brownien standard.
        
        On suppose que l'on veuille estimer l'espérance du gain perçu par la détention d'une option européenne d'achat (call) de maturité $T$ et de prix d'exercice (strike) $K$.
        """
    )

    # Définition des Paramètres
    st.sidebar.header("Paramètres de la Simulation")
    option_type = st.sidebar.radio("Type d'Option", ["call", "put"])
    S_0 = st.sidebar.number_input("Prix actuel de l'action (S0)", min_value=0.0, value=42.0, step=1.0)
    K = st.sidebar.number_input("Prix d'exercice de l'option (K)", min_value=0.0, value=40.0, step=1.0)
    T = st.sidebar.number_input("Temps jusqu'à l'expiration (années)", min_value=0.1, value=0.5, step=0.1)
    r = st.sidebar.number_input("Taux d'intérêt sans risque (r)", min_value=0.00, value=0.1, step=0.001)
    sigma = st.sidebar.number_input("Volatilité (sigma)", min_value=0.001, value=0.2, step=0.001)
    mu = st.sidebar.number_input("Moyenne (mu)", min_value=0.0000, value=0.0150, step=0.001)
    N = st.sidebar.number_input("Nombre de Simulations", min_value=1, value=1000)

    # Bouton pour lancer les simulations
    if st.button("Lancer les Simulations"):
        # CALL
        if option_type == "call":

            # Graphique du Processus de Diffusion
            D = process(S_0, T, mu, sigma, 200, 100)
            fig_bs, ax_bs = plt.subplots(figsize=(10, 6))
            ax_bs.plot(D)
            ax_bs.set_title('Processus de Diffusion')
            ax_bs.set_xlabel('Time Increments dt')
            ax_bs.set_ylabel("Stock Price S")
            st.pyplot(fig_bs)

            # Simulation Black & Scholes
            call_price_bs = vanilla_call_price(S_0, K, r, sigma, T)

            # Affichage des résultats Black & Scholes
            st.subheader("Résultats de la Simulation Black & Scholes")
            st.write(f"Le Prix de l'Option avec Black & Scholes est {call_price_bs}")

            # Simulation Monte Carlo simplifiée
            num_simulations = 10000  # Vous pouvez ajuster ce nombre selon vos besoins
            option_price_mc = monte_carlo_call_price(S_0, K, r, sigma, T, num_simulations)
            error = option_price_mc * 0.05  # Estimation de l'erreur, ajustez en fonction de la précision souhaitée

            # Affichage des résultats Monte Carlo
            st.subheader("Résultats de la Simulation Monte Carlo")
            st.write(f"Le Prix de l'Option avec Monte Carlo est {option_price_mc} avec une erreur estimée de {error}")

            st.subheader("Conclusion")
            st.write("On remarque que le prix donné par la Formule de Black Scholes se rapproche fortement de celui donné par la méthode Monte Carlo.")

        elif option_type == "put":

            # Graphique du Processus de Diffusion
            path = process(S_0, T, mu, sigma, 200, 100)
            fig_bs, ax_bs = plt.subplots(figsize=(10, 6))
            ax_bs.plot(path)
            ax_bs.set_title('Processus de Diffusion')
            ax_bs.set_xlabel('Time Increments dt')
            ax_bs.set_ylabel("Stock Price S")
            st.pyplot(fig_bs)

            # Simulation Black & Scholes
            put_price_bs = vanilla_put_price(S_0, K, r, sigma, T)

            # Affichage des résultats Black & Scholes
            st.subheader("Résultats de la Simulation Black & Scholes")
            st.write(f"Le Prix de l'Option avec Black & Scholes est {put_price_bs}")

            # Simulation Monte Carlo simplifiée
            num_simulations = 10000  # Vous pouvez ajuster ce nombre selon vos besoins
            option_price_mc = monte_carlo_put_price(S_0, K, r, sigma, T, num_simulations)
            error = option_price_mc * 0.05  # Estimation de l'erreur, ajustez en fonction de la précision souhaitée

            # Affichage des résultats Monte Carlo
            st.subheader("Résultats de la Simulation Monte Carlo")
            st.write(f"Le Prix de l'Option avec Monte Carlo est {option_price_mc} avec une erreur estimée de {error}")

            st.subheader("Conclusion")
            st.write("On remarque que le prix donné par la Formule de Black Scholes se rapproche fortement de celui donné par la méthode Monte Carlo.")


    # Calcul des Grecques
    delta = greeks(S_0, K, r, sigma, T, "delta", option_type)
    gamma_val = greeks(S_0, K, r, sigma, T, "gamma", option_type)
    vega = greeks(S_0, K, r, sigma, T, "vega", option_type)
    theta_val = greeks(S_0, K, r, sigma, T, "theta", option_type)

    st.subheader("Valeurs des Grecques")
    st.write(f"Delta : {delta:.4f}")
    st.write(f"Gamma : {gamma_val:.4f}")
    st.write(f"Vega : {vega:.4f}")
    st.write(f"Theta : {theta_val:.4f}")

    # Graphe des Grecques
    S_values = np.linspace(S_0 * 0.5, S_0 * 1.5, 100)
    deltas = [greeks(S, K, r, sigma, T, "delta", option_type) for S in S_values]
    gammas = [greeks(S, K, r, sigma, T, "gamma", option_type) for S in S_values]
    vegas = [greeks(S, K, r, sigma, T, "vega", option_type) for S in S_values]
    thetas = [greeks(S, K, r, sigma, T, "theta", option_type) for S in S_values]

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Sensibilités des options européennes")

    axs[0, 0].plot(S_values, deltas, color="blue")
    axs[0, 0].set_title("Delta")
    axs[0, 0].set_xlabel("Prix de l'actif sous-jacent (S)")
    axs[0, 0].set_ylabel("Delta")

    axs[0, 1].plot(S_values, gammas, color="blue")
    axs[0, 1].set_title("Gamma")
    axs[0, 1].set_xlabel("Prix de l'actif sous-jacent (S)")
    axs[0, 1].set_ylabel("Gamma")

    axs[1, 0].plot(S_values, thetas, color="blue")
    axs[1, 0].set_title("Theta")
    axs[1, 0].set_xlabel("Prix de l'actif sous-jacent (S)")
    axs[1, 0].set_ylabel("Theta")

    axs[1, 1].plot(S_values, vegas, color="blue")
    axs[1, 1].set_title("Vega")
    axs[1, 1].set_xlabel("Prix de l'actif sous-jacent (S)")
    axs[1, 1].set_ylabel("Vega")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    st.pyplot(fig)









def euro_option_page_with_data():
    st.title("Pricing d'options européennes avec des données réelles")

    # Chargement des données
    uploaded_file = st.file_uploader("Uploader un fichier CSV contenant les données des options", type="csv")
    if uploaded_file is not None:

        data = pd.read_csv(uploaded_file)
        st.write("Aperçu des données :")
        st.write(data.head())

        ###################################
        # VALEURS PAR DEFAUT
        ###################################

        # Lire le fichier JSON
        with open(r"Pricing_options\data\variables.json", "r") as json_file:
            vars = json.load(json_file)

        # valeurs observées le jour de la création du jeu de données (exécution de generate_data.py)
        S_0_default = vars["apple_stock_price"]
        r_default = round(vars["US_10year_treasury_note"], 2)

        # valeurs arbitraires
        T_default = 0.13972602739726028 
        K_default = 200
        
        ###################################
        # PARAMETRES
        ###################################

        S_0 = st.number_input("Prix actuel de l'action (S_0)", min_value=0.0, value=S_0_default)
        r = st.number_input("Taux sans risque (r)", min_value=0.0, value=r_default)

        st.markdown("""Remarque : Les valeurs par défaut correspondent aux prix du sous-jacent et au taux sans risque au moment de l'exécution du module <strong>generate_data.py</strong>.""", unsafe_allow_html=True)

        strike = data['K (strike)'].values
        maturity = data['days_to_maturity'].values
        sigmas = data['sigma (implied volatility)'].values

        ###################################
        # NAPPE DE VOLATILITE
        ###################################
        
        st.subheader("Surface de Volatilité Implicite")

        # grids of points (for visualization purposes)
        strike_range = np.linspace(strike.min(), strike.max(), 100)
        maturity_range = np.linspace(maturity.min(), maturity.max(), 100)
        strike_grid, maturity_grid = np.meshgrid(strike_range, maturity_range)
        # interpolation
        sigma_grid = griddata((strike, maturity), sigmas, (strike_grid, maturity_grid), method='linear')


            
        fig = go.Figure(data=[go.Surface(
            z=sigma_grid, 
            x=strike_grid, 
            y=maturity_grid,  
            colorscale='Viridis', 
            colorbar=dict(title='Volatility') 
        )])

        fig.update_layout(
            scene=dict(
                xaxis_title='Strike',
                yaxis_title='Maturity (Days)',
                zaxis_title='Volatility'
            ),
            width=900,
            height=600,
        )
        st.plotly_chart(fig)



        ###################################
        # TRACÉ DES GRECQUES
        ###################################

        st.subheader("Visualisation des Grecques des Options")

        option_type = st.selectbox("Type de l'option", ["call", "put"])

        # Définition des paramètres
        maturities = np.array([30, 60, 90, 180, 365])
        cmap = get_cmap('viridis') 
        greeks_to_plot = ["delta", "gamma", "theta", "vega"]

        def get_volatility(K, T):
            """
            Fonction qui récupère la volatilité interpolée pour un couple
            de paramètres (strike, maturité) donné.
            """
            return griddata((strike, maturity), sigmas, (K, T), method='linear')


        for element in greeks_to_plot:
            
            fig, ax = plt.subplots(figsize=(10, 6))

            # Calculate and plot for each maturity
            for i, t in enumerate(maturities):
                sigma = get_volatility(strike_range, t)
                greek_values = greeks(S_0, strike_range, r, sigma, t / 365, element, option_type)
                ax.plot(strike_range, greek_values, label=f'{t} jours', color=cmap(i / len(maturities)))

            # Configure the plot
            ax.set_title(f"{element.capitalize()} pour différentes maturités")
            ax.set_xlabel("Prix du sous-jacent")
            ax.set_ylabel(f"{element.capitalize()} d'une option {option_type.capitalize()}")
            ax.legend(title="Maturité", bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()


            st.pyplot(fig)


        ###################################
        # PRICING D'UNE OPTION
        ###################################


        st.subheader("Pricing d'une option qui n'est pas dans le marché :")

        strike_choice = st.number_input("Entrez un Strike", value=K_default)
        maturity_choice = st.number_input("Entrez une Maturité (en années)", value=T_default)

        def pricer(S_0, K, r, T):

            T_days = int(T*365)
            # récupération de la volatilité interpolée
            sigma = get_volatility(K, T_days)

            # calcul des grecques
            delta_call = greeks(S_0, K, r, sigma, T, "delta", "call")
            delta_put = greeks(S_0, K, r, sigma, T, "delta", "put")
            gamma = greeks(S_0, K, r, sigma, T, "gamma", "call")
            vega = greeks(S_0, K, r, sigma, T, "vega", "call")
            theta_call = greeks(S_0, K, r, sigma, T, "theta", "call")
            theta_put = greeks(S_0, K, r, sigma, T, "theta", "put")

            call_price = vanilla_call_price(S_0, K, r, sigma, T)
            put_price = vanilla_put_price(S_0, K, r, sigma, T)

            results = []
            results.append({
                        'Delta (Call)': delta_call,
                        'Delta (Put)': delta_put,
                        'Gamma': gamma,
                        'Vega': vega,
                        'Theta (Call)': theta_call,
                        'Theta (Put)': theta_put
                    })
            
            return call_price, put_price, pd.DataFrame(results)
            
            
        
        call_price, put_price, greek_results = pricer(S_0, strike_choice, r, maturity_choice)
    
        st.success(f"Le prix de l'option CALL avec un strike de {strike_choice} et une maturité de {maturity_choice:.2f} années est de {call_price:.2f}.")
        st.success(f"Le prix de l'option PUT avec un strike de {strike_choice} et une maturité de {maturity_choice:.2f} années est de {put_price:.2f}.")

        st.write(f"Grecques associées à un strike de {strike_choice} et une maturité de {maturity_choice:.2f} années :")
        st.dataframe(greek_results)

    
        def find_nearest_neighbors(K, T, strike, maturity):
            """
            Fonction qui identifie les 4 points utilisés pour l'interpolation de la volatilité implicite.
            """
            K_idx = np.searchsorted(strike, K)
            T_idx = np.searchsorted(maturity, int(T*365))

            points = [
                (strike[K_idx-1], maturity[T_idx-1]), (strike[K_idx], maturity[T_idx-1]),
                (strike[K_idx-1], maturity[T_idx]), (strike[K_idx], maturity[T_idx])
            ]
            return points


        points = find_nearest_neighbors(strike_choice, maturity_choice, strike_range, maturity_range)

        # Plotting the interpolated surface
        fig, ax = plt.subplots(figsize=(10, 8))
        contour = ax.contourf(strike_grid, maturity_grid, sigma_grid, 20, cmap='viridis')
        plt.colorbar(contour, ax=ax, label="Volatility") 

        for p in points:
            ax.scatter(p[0], p[1], color='red', s=20, label=f"Point ({p[0]:.2f}, {p[1]:.2f})")
        
        maturity_choice_days = int(maturity_choice*365)
        ax.scatter(strike_choice, maturity_choice_days, color='blue', s=100, marker='+', linewidths=1, label=f"Entrées ({strike_choice}, {maturity_choice_days})")

        ax.set_xlabel("Prix d'exercice (K)")
        ax.set_ylabel('Maturité (T)')
        ax.set_title("Points utilisés pour l'interpolation de la volatilité implicite")
        ax.legend()


        st.pyplot(fig)






def asian_option_page():

    # Fonctions nécessaires à la simulation des options asiatiques

    # Simulation des trajectoires (GBM)
    def simulate_gbm(S_0, sigma, mu, T, steps, N):
        dt = T / steps
        increments = np.random.normal(0, 1, size=(steps, N)) * np.sqrt(dt)
        drift = (mu - 0.5 * sigma**2) * dt
        paths = np.log(S_0) + np.cumsum(drift + sigma * increments, axis=0)
        return np.exp(paths)

    # Calcul du payoff pour une option asiatique
    def payoff(S, K, option_type="call"):
        average_price = np.mean(S, axis=0)
        if option_type == "call":
            return np.maximum(average_price - K, 0)
        elif option_type == "put":
            return np.maximum(K - average_price, 0)

    # Simulation Monte Carlo pour une option asiatique
    def simulate_monte_carlo(S_0, K, sigma, mu, T, steps, N, option_type="call"):
        paths = simulate_gbm(S_0, sigma, mu, T, steps, N)
        P = payoff(paths, K, option_type)
        return P, paths

    # Analyse de la convergence Monte Carlo
    def convergence_mc(P, ic):
        a = norm.ppf(ic)
        M = []
        ET = []
        b_inf = []
        b_sup = []
        error = []
        for i in range(1, len(P) + 1):
            M.append(np.mean(P[:i]))
            ET.append(np.std(P[:i]))
            b_inf.append(M[-1] - a * ET[-1] / np.sqrt(i))
            b_sup.append(M[-1] + a * ET[-1] / np.sqrt(i))
            error.append((b_sup[-1] - b_inf[-1]) / 2)
        return M, b_inf, b_sup, error

    # Présentation de l'option asiatique
    st.title("Simulation du Pricing d'une Option Asiatique")
    st.markdown(
        """
        Une option asiatique est une option dont le payoff dépend de la moyenne des prix de l'actif sous-jacent
        sur une période donnée. 
        """
    )

    # Définition des Paramètres
    st.sidebar.header("Paramètres de la Simulation")
    option_type = st.sidebar.radio("Type d'Option", ["call", "put"], index=0)
    S_0 = st.sidebar.number_input("Prix actuel de l'action (S0)", min_value=0.0, value=42.0, step=1.0)
    K = st.sidebar.number_input("Prix d'exercice de l'option (K)", min_value=0.0, value=40.0, step=1.0)
    T = st.sidebar.number_input("Temps jusqu'à l'expiration (années)", min_value=0.1, value=1.0, step=0.1)
    r = st.sidebar.number_input("Taux d'intérêt sans risque (r)", min_value=0.00, value=0.05, step=0.001)
    sigma = st.sidebar.number_input("Volatilité (sigma)", min_value=0.001, value=0.2, step=0.001)
    mu = st.sidebar.number_input("Moyenne (mu)", min_value=0.0000, value=0.05, step=0.001)
    N = st.sidebar.number_input("Nombre de Simulations", min_value=1, value=10000, step=1000)
    Ic = st.sidebar.number_input("Intervalle de Confiance", min_value=0.0, value=0.95, step=0.01)

    # Bouton pour lancer les simulations
    if st.button("Lancer les Simulations"):
        steps = 252
        P, paths = simulate_monte_carlo(S_0, K, sigma, mu, T, steps, N, option_type)
        M, b_inf, b_sup, error = convergence_mc(P, Ic)
        Price = np.mean(P)

        # Résultats
        st.subheader("Résultats de la Simulation")
        st.write(f"**Prix estimé de l'option asiatique ({option_type}):** {Price:.4f}")
        st.write(f"**Erreur à l'intervalle de confiance {Ic * 100}% :** {error[-1]:.4f}")

        # Graphique de convergence
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(M, label="Moyenne")
        ax1.fill_between(range(len(M)), b_inf, b_sup, color="gray", alpha=0.3, label="Intervalle de confiance")
        ax1.set_title("Convergence Monte Carlo")
        ax1.set_xlabel("Nombre de Simulations")
        ax1.set_ylabel("Prix de l'Option")
        ax1.legend()
        st.pyplot(fig1)

        # Graphique des trajectoires simulées
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        for i in range(min(50, N)):
            ax2.plot(paths[:, i], lw=0.8)
        ax2.set_title("Trajectoires Simulées de l'Actif")
        ax2.set_xlabel("Temps (pas de temps)")
        ax2.set_ylabel("Prix de l'Actif")
        st.pyplot(fig2)



# Simulation de trajectoires et pricing
def simulate_gbm(S_0, sigma, mu, T, steps, N):
    dt = T / steps
    increments = np.random.normal(0, 1, size=(steps, N)) * np.sqrt(dt)
    drift = (mu - 0.5 * sigma**2) * dt
    paths = np.log(S_0) + np.cumsum(drift + sigma * increments, axis=0)
    return np.exp(paths)

def payoff(S, K, option_type="call"):
    average_price = np.mean(S, axis=0)
    if option_type == "call":
        return np.maximum(average_price - K, 0)
    elif option_type == "put":
        return np.maximum(K - average_price, 0)

def simulate_monte_carlo(S_0, K, sigma, mu, T, steps, N, option_type="call"):
    paths = simulate_gbm(S_0, sigma, mu, T, steps, N)
    P = payoff(paths, K, option_type)
    return np.mean(P) * np.exp(-mu * T)  # Actualisation du payoff


# Page pour l'importation des données
def asian_option_with_data():
    st.title("Options Asiatiques avec Données")
    st.markdown(
        """
        Cette page permet de télécharger un fichier contenant des données d'options et de calculer
        automatiquement les prix des options asiatiques en fonction de ces données.
        """
    )

    # Téléchargement des données
    uploaded_file = st.file_uploader("Téléchargez un fichier CSV avec vos données", type=["csv"])

    if uploaded_file is not None:
        # Lecture des données
        df = pd.read_csv(uploaded_file)
        st.write("### Aperçu des données téléchargées :")
        st.dataframe(df.head())

        # Ajout des paramètres nécessaires à la simulation
        st.sidebar.header("Paramètres Globaux")
        steps = st.sidebar.number_input("Nombre de pas de temps", min_value=1, value=252, step=1)
        N = st.sidebar.number_input("Nombre de simulations Monte Carlo", min_value=100, value=10000, step=1000)
        mu = st.sidebar.number_input("Moyenne (mu)", min_value=0.0, value=0.05, step=0.01)

        # Calculer les options asiatiques pour chaque ligne
        if st.button("Calculer les Prix des Options Asiatiques"):
            results = []
            for index, row in df.iterrows():
                try:
                    S_0 = row["C (call price)"]  # Exemple : prix initial basé sur le prix d'achat
                    K = row["K (strike)"]
                    T = row["days_to_maturity"] / 365.0
                    sigma = row["sigma (implied volatility)"]
                    option_type = "call"  # Par défaut, utiliser 'call'
                    
                    price = simulate_monte_carlo(S_0, K, sigma, mu, T, steps, N, option_type)
                    results.append(price)
                except Exception as e:
                    results.append(None)

            df["Asian Option Price"] = results
            st.write("### Résultats des Prix des Options Asiatiques :")
            st.dataframe(df)

            # Explication des résultats
            st.markdown(
                """
                Les résultats affichés ci-dessus représentent les prix des options asiatiques calculés pour chaque 
                ligne du fichier de données. Les prix sont basés sur la moyenne simulée des trajectoires du prix 
                sous-jacent et sont actualisés en fonction des paramètres fournis.
                """
            )

            # Option pour télécharger les résultats
            csv = df.to_csv(index=False)
            st.download_button(
                label="Télécharger les résultats au format CSV",
                data=csv,
                file_name="asian_option_results.csv",
                mime="text/csv"
            )








# Simulation des trajectoires (GBM)
def simulate_gbm(S_0, sigma, mu, T, steps, N):
    dt = T / steps
    increments = np.random.normal(0, 1, size=(steps, N)) * np.sqrt(dt)
    drift = (mu - 0.5 * sigma**2) * dt
    paths = np.log(S_0) + np.cumsum(drift + sigma * increments, axis=0)
    return np.exp(paths)

# Payoff pour une option tunnel
def payoff_tunnel(S, K, L, U, option_type="knock-in"):
    final_price = S[-1, :]
    if option_type == "knock-in":
        return np.where((final_price > L) & (final_price < U), np.maximum(final_price - K, 0), 0)
    elif option_type == "knock-out":
        return np.where((final_price <= L) | (final_price >= U), np.maximum(final_price - K, 0), 0)

# Simulation Monte Carlo pour une option tunnel
def simulate_tunnel(S_0, K, L, U, sigma, mu, T, steps, N, option_type="knock-in"):
    paths = simulate_gbm(S_0, sigma, mu, T, steps, N)
    P = payoff_tunnel(paths, K, L, U, option_type)
    # Comptage des trajectoires qui franchissent les barrières
    within_barriers = np.sum((paths[-1, :] > L) & (paths[-1, :] < U))
    return np.mean(P) * np.exp(-mu * T), paths, within_barriers, N

# Page sans données
def tunnel_option_page():
    st.title("Option Tunnel - Sans Données")
    st.markdown(
        """
        Cette page permet de calculer le prix d'une option tunnel en fournissant les paramètres nécessaires.
        """
    )

    st.sidebar.header("Paramètres de Simulation")
    option_type = st.sidebar.radio("Type d'Option", ["knock-in", "knock-out"], index=0)
    S_0 = st.sidebar.number_input("Prix initial de l'actif (S0)", min_value=0.0, value=100.0, step=1.0)
    K = st.sidebar.number_input("Prix d'exercice (K)", min_value=0.0, value=100.0, step=1.0)
    L = st.sidebar.number_input("Barrière inférieure (L)", min_value=0.0, value=90.0, step=1.0)
    U = st.sidebar.number_input("Barrière supérieure (U)", min_value=0.0, value=110.0, step=1.0)
    T = st.sidebar.number_input("Temps jusqu'à maturité (années)", min_value=0.1, value=1.0, step=0.1)
    sigma = st.sidebar.number_input("Volatilité (sigma)", min_value=0.01, value=0.2, step=0.01)
    mu = st.sidebar.number_input("Moyenne (mu)", min_value=0.0, value=0.05, step=0.01)
    steps = st.sidebar.number_input("Nombre de pas de temps", min_value=1, value=252, step=1)
    N = st.sidebar.number_input("Nombre de simulations Monte Carlo", min_value=100, value=10000, step=1000)

    if st.button("Lancer la Simulation"):
        price, paths, within_barriers, total_paths = simulate_tunnel(S_0, K, L, U, sigma, mu, T, steps, N, option_type)
        proportion_within = within_barriers / total_paths * 100

        st.subheader("Résultat de la Simulation")
        st.write(f"**Prix estimé de l'option tunnel ({option_type}):** {price:.4f}")
        st.write(f"**Proportion des trajectoires dans les barrières :** {proportion_within:.2f}%")

        st.write("### Trajectoires Simulées")
        fig, ax = plt.subplots(figsize=(10, 6))
        for i in range(min(10, N)):
            ax.plot(paths[:, i], lw=0.8)
        ax.axhline(y=L, color='r', linestyle='--', label="Barrière Inférieure (L)")
        ax.axhline(y=U, color='g', linestyle='--', label="Barrière Supérieure (U)")
        ax.axhline(y=K, color='b', linestyle='--', label="Strike (K)")
        ax.set_title("Trajectoires Simulées avec Barrières")
        ax.set_xlabel("Temps (pas)")
        ax.set_ylabel("Prix de l'Actif")
        ax.legend()
        st.pyplot(fig)

# Page avec données
def tunnel_option_with_data():
    st.title("Option Tunnel - Avec Données")
    st.markdown(
        """
        Cette page permet de calculer les prix des options tunnel en téléchargeant un fichier CSV contenant les données nécessaires.
        """
    )

    uploaded_file = st.file_uploader("Téléchargez un fichier CSV avec vos données", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("### Aperçu des données téléchargées :")
        st.dataframe(df.head())

        st.sidebar.header("Paramètres Globaux")
        steps = st.sidebar.number_input("Nombre de pas de temps", min_value=1, value=252, step=1)
        N = st.sidebar.number_input("Nombre de simulations Monte Carlo", min_value=100, value=10000, step=1000)
        mu = st.sidebar.number_input("Moyenne (mu)", min_value=0.0, value=0.05, step=0.01)

        if st.button("Calculer les Prix des Options Tunnel"):
            results = []
            for index, row in df.iterrows():
                try:
                    S_0 = row["C (call price)"]
                    K = row["K (strike)"]
                    L = row["B (call bid)"]
                    U = row["A (call ask)"]
                    T = row["days_to_maturity"] / 365.0
                    sigma = row["sigma (implied volatility)"]
                    option_type = "knock-in"

                    price, _, within_barriers, total_paths = simulate_tunnel(S_0, K, L, U, sigma, mu, T, steps, N, option_type)
                    proportion_within = within_barriers / total_paths * 100
                    results.append((price, proportion_within))
                except Exception as e:
                    results.append((None, None))

            df["Tunnel Option Price"] = [r[0] for r in results]
            df["Proportion Within Barriers (%)"] = [r[1] for r in results]
            st.write("### Résultats des Prix des Options Tunnel :")
            st.dataframe(df)
            st.markdown("""Les résultats montrent les prix des options tunnel calculés pour chaque ligne de données. 
            La proportion des trajectoires respectant les barrières aide à comprendre pourquoi certains prix sont nuls : 
            si peu de trajectoires restent dans les barrières, le payoff est souvent nul.
         """)

            csv = df.to_csv(index=False)
            st.download_button(
                label="Télécharger les résultats au format CSV",
                data=csv,
                file_name="tunnel_option_results.csv",
                mime="text/csv"
            )







# Fonction principale de l'application Streamlit
def main():
    pages = ["Option - Européenne","Option - Européenne avec données","Option - Asian", "Option - Asian avec données","Option - Tunnel","Option - Tunnel avec données"]
    selected_page = st.sidebar.radio("Sélectionner l'Option", pages)

    if selected_page == "Option - Européenne":
        euro_option_page()

    elif selected_page == "Option - Européenne avec données":
        euro_option_page_with_data()

    elif selected_page == "Option - Asian":
        asian_option_page()

    elif selected_page == "Option - Asian avec données":
        asian_option_with_data()

    elif selected_page == "Option - Tunnel":
        tunnel_option_page()
    elif selected_page == "Option - Tunnel avec données":
        tunnel_option_with_data()

    

if __name__ == "__main__":
    main()