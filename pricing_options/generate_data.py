###################################
# LIBRAIRIES ET PACKAGES
###################################

import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from datetime import datetime
import yfinance as yf
from app import vanilla_call_price, vanilla_put_price
import json
from datetime import datetime

###################################
# CHARGEMENT DES DONNEES
###################################

# action américaine ('AAPL' pour Apple)
apple_ticker = yf.Ticker("AAPL")

# dates d'expiration des options
expirations = apple_ticker.options

# listes pour stocker les données
call_data = []
put_data = []

for expiration in expirations:

    # obtention des chaînes d'options pour la date d'expiration donnée
    opt_chain = apple_ticker.option_chain(expiration)
    calls = opt_chain.calls
    puts = opt_chain.puts

    for _, call_row in calls.iterrows():
        strike = call_row['strike']
        maturity = expiration
        C = call_row['lastPrice']
        call_volume = call_row['volume']
        call_ask = call_row['ask']
        call_bid = call_row['bid']


        call_data.append({
            'K (strike)': strike,
            'T (maturity)': maturity,
            'C (call price)': C,
            'V (call volume)': call_volume,
            'A (call ask)': call_ask,
            'B (call bid)': call_bid,
        })


    for _, put_row in puts.iterrows():
        strike = put_row['strike']
        maturity = expiration
        P = put_row['lastPrice']
        put_volume = put_row['volume']
        put_ask = put_row['ask']
        put_bid = put_row['bid']

        put_data.append({
            'K (strike)': strike,
            'T (maturity)': maturity,
            'P (put price)': P,
            'V (put volume)': put_volume,
            'A (put ask)': put_ask,
            'B (put bid)': put_bid,
        })


# conversion des listes en dataframes
call_data = pd.DataFrame(call_data)
put_data = pd.DataFrame(put_data)


# jointure sur strike et maturité
df = pd.merge(call_data, put_data, on=['T (maturity)', 'K (strike)'], how='inner')



###################################
# VARIABLES
###################################

# valeur de l'action d'Apple hier/aujourd'hui - dernier prix de clôture
S0 = apple_ticker.history(period="1d")['Close'][0]

# rendement des obligations du Trésor à 10 ans
treasury_10y = yf.Ticker("^TNX")
r = treasury_10y.history(period="1d")['Close'][0] / 100


# Paramètres de l'optimisation pour le calcul de la vol implicite
initial_sigma = 0.2
bounds = [(1e-6, 5.0)]


###################################
# PRETRAITEMENT
###################################

# filtrage sur le volume ("V (call volume)" ou "P (put volume)" est > 5)
df = df[(df['V (call volume)'] > 5) & (df['V (put volume)'] > 5)]

today = pd.to_datetime(datetime.today().strftime('%Y-%m-%d'))
df['days_to_maturity'] = (pd.to_datetime(df['T (maturity)']) - today).dt.days

# filtrage des maturités inférieures ou égales à 30 jours
df = df[df['days_to_maturity'] > 30]



###################################
# VOLATILITE IMPLICITE COMBINEE
###################################


def objective(sigma, S, K, r, T, price, option_type):
    
    if option_type == 'call':
        model_price =  vanilla_call_price(S, K, r, sigma, T)
    elif option_type == 'put':
        model_price = vanilla_put_price(S, K, r, sigma, T)
    else:
        raise ValueError("Option type must be 'call' or 'put'")

    return (model_price - price) ** 2


def implied_vol(S, K, r, T, Call, Put, V_call, V_put, initial_sigma, bounds):
    # l'option la plus liquide est retenue pour le calcul de la volatilité
    if V_call > V_put:
        option_type = 'call'
        price = Call
    else:
        option_type = 'put'
        price = Put

    result = minimize(
        objective,
        x0=initial_sigma,
        args=(S, K, r, T, price, option_type),
        bounds=bounds,
        method="trust-constr"  # Méthode d'optimisation contraint
    )

    if result.success:
        return result.x[0]
    else:
        raise ValueError("Optimization failed to converge")
    

###################################
# AJOUT DE LA VOLALITE AU DF
###################################

sigmas = []

for _, row in df.iterrows():
    K = row['K (strike)']
    T = row['days_to_maturity'] / 365.0 # conversion en années (parce qu'on considère un taux sans risque annualisé)
    Call = row['C (call price)']
    Put = row['P (put price)']
    V_call = row['V (call volume)']
    V_put =  row['V (put volume)']

    try:
        sigma = implied_vol(S0, K, r, T, Call, Put, V_call, V_put, initial_sigma, bounds)

    except ValueError:
        sigma = np.nan  # np.nan as a placeholder

    sigmas.append(sigma)

df['sigma (implied volatility)'] = sigmas


###################################
# EXPORTATION DES DONNEES
###################################

variables = {
    "date": datetime.now().strftime("%Y-%m-%d"),
    "apple_stock_price": S0,
    "US_10year_treasury_note": r
}

with open(r"Pricing_options\data\variables.json", "w") as json_file:
    json.dump(variables, json_file)

###################################
# Le fichier CSV suivant devra être téléversé sur l'application Streamlit (pricing.py) pour les onglets :
# - "Option - Européenne avec données"
# - "Option - Asian avec données"
# - "Option - Tunnel avec données".
###################################

df.to_csv(r'Pricing_options\data\data_28jan.csv', index=False)

