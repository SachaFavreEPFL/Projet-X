import json
import pandas as pd
import numpy as np

# Chargement des données
with open('dataset/stocks_data.json', 'r') as f:
    stocks_data = json.load(f)['stocks']
with open('dataset/stocks_history.json', 'r') as f:
    stocks_history = json.load(f)['stocks']

def extract_features(symbol, stock):
    overview = stock.get('overview', {}).get('data', {})
    sector = overview.get('sector', None)
    
    # Ratios de valorisation
    pe = overview.get('pe_ratio', None)
    pb = overview.get('pb_ratio', None)
    ps = overview.get('ps_ratio', None)
    
    # Qualité
    roe = overview.get('return_on_equity', None)
    roa = overview.get('return_on_assets', None)
    operating_margins = overview.get('operating_margins', None)
    profit_margins = overview.get('profit_margins', None)
    debt_to_equity = overview.get('debt_to_equity', None)
    current_ratio = overview.get('current_ratio', None)
    
    # Croissance
    earnings_growth = overview.get('earnings_growth', None)
    revenue_growth = overview.get('revenue_growth', None)
    
    # Marché
    beta = overview.get('beta', None)
    market_cap = overview.get('market_cap', None)
    
    return {
        'symbol': symbol,
        'sector': sector,
        # Valorisation
        'pe_ratio': pe,
        'pb_ratio': pb,
        'ps_ratio': ps,
        # Qualité
        'roe': roe,
        'roa': roa,
        'operating_margins': operating_margins,
        'profit_margins': profit_margins,
        'debt_to_equity': debt_to_equity,
        'current_ratio': current_ratio,
        # Croissance
        'earnings_growth': earnings_growth,
        'revenue_growth': revenue_growth,
        # Marché
        'beta': beta,
        'market_cap': market_cap
    }

# Extraction des features
data = [extract_features(sym, stock) for sym, stock in stocks_data.items()]
df = pd.DataFrame(data)

# Suppression des lignes sans données essentielles
essential_cols = ['sector', 'pe_ratio', 'pb_ratio', 'ps_ratio', 'roe', 'operating_margins']
df = df.dropna(subset=essential_cols)

# Définition des colonnes par catégorie
valuation_cols = ['pe_ratio', 'pb_ratio', 'ps_ratio']
quality_cols = ['roe', 'roa', 'operating_margins', 'profit_margins']
growth_cols = ['earnings_growth', 'revenue_growth']
market_cols = ['beta']

# Calcul des médianes par secteur pour les ratios de valorisation
sector_medians = df.groupby('sector')[valuation_cols].median()
sector_medians = sector_medians.rename(columns={col: f'{col}_sector_median' for col in valuation_cols})
df = df.merge(sector_medians, left_on='sector', right_index=True)

# Calcul des écarts en pourcentage
for col in valuation_cols:
    df[f'{col}_gap'] = (df[col] - df[f'{col}_sector_median']) / df[f'{col}_sector_median'] * 100

# Scores par catégorie
df['valuation_score'] = (
    0.4 * df['pe_ratio_gap'] +
    0.3 * df['pb_ratio_gap'] +
    0.3 * df['ps_ratio_gap']
)

# Normalisation des scores de qualité (0-100)
for col in quality_cols:
    if col in df.columns:
        df[f'{col}_normalized'] = (df[col] - df[col].min()) / (df[col].max() - df[col].min()) * 100

df['quality_score'] = (
    0.3 * df['roe_normalized'] +
    0.2 * df['roa_normalized'] +
    0.3 * df['operating_margins_normalized'] +
    0.2 * df['profit_margins_normalized']
)

# Normalisation des scores de croissance
for col in growth_cols:
    if col in df.columns:
        df[f'{col}_normalized'] = (df[col] - df[col].min()) / (df[col].max() - df[col].min()) * 100

df['growth_score'] = (
    0.5 * df['earnings_growth_normalized'] +
    0.5 * df['revenue_growth_normalized']
)

# Score de marché (inverse du beta, normalisé)
if 'beta' in df.columns:
    df['market_score'] = (1 / df['beta']).fillna(0)
    df['market_score'] = (df['market_score'] - df['market_score'].min()) / (df['market_score'].max() - df['market_score'].min()) * 100
else:
    df['market_score'] = 50  # Valeur neutre si beta non disponible

# Score global pondéré
df['final_score'] = (
    0.4 * df['valuation_score'] +
    0.3 * df['quality_score'] +
    0.2 * df['growth_score'] +
    0.1 * df['market_score']
)

# Classification
def classify_valuation(score):
    if score <= -30:
        return 'Fortement sous-évaluée'
    elif score <= -10:
        return 'Légèrement sous-évaluée'
    elif score <= 10:
        return 'Normale'
    elif score <= 30:
        return 'Légèrement surévaluée'
    else:
        return 'Fortement surévaluée'

df['valuation_class'] = df['final_score'].apply(classify_valuation)

# Sauvegarde du dataset
train_cols = [
    'symbol', 'sector',
    # Valorisation
    'pe_ratio', 'pb_ratio', 'ps_ratio',
    # Qualité
    'roe', 'roa', 'operating_margins', 'profit_margins',
    'debt_to_equity', 'current_ratio',
    # Croissance
    'earnings_growth', 'revenue_growth',
    # Marché
    'beta', 'market_cap',
    # Scores
    'valuation_score', 'quality_score', 'growth_score', 'market_score',
    'final_score', 'valuation_class'
]

df[train_cols].to_csv('dataset/train_stocks_valuation.csv', index=False)

# Affichage des statistiques
print("\nDistribution des classes de valorisation :")
print(df['valuation_class'].value_counts(normalize=True).round(3) * 100)

print("\nCorrélations avec le score final :")
correlations = df[['valuation_score', 'quality_score', 'growth_score', 'market_score', 'final_score']].corr()
print(correlations['final_score'].sort_values(ascending=False))

print("\nDataset d'entraînement généré : dataset/train_stocks_valuation.csv") 