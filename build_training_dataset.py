import json
import pandas as pd
import numpy as np

def normalize_series(series):
    """Normalise une série en utilisant la méthode min-max"""
    min_val = series.min()
    max_val = series.max()
    if max_val == min_val:
        return pd.Series(0.5, index=series.index)
    return (series - min_val) / (max_val - min_val)

def calculate_valuation_score(row):
    """Calcule le score de valorisation"""
    scores = []
    weights = {
        'pe_ratio': -0.3,  # Plus bas est meilleur
        'pb_ratio': -0.2,  # Plus bas est meilleur
        'ps_ratio': -0.2,  # Plus bas est meilleur
        'peg_ratio': -0.2,  # Plus bas est meilleur
        'enterprise_to_ebitda': -0.1  # Plus bas est meilleur
    }
    
    for metric, weight in weights.items():
        if pd.notna(row[metric]):
            scores.append(weight * row[metric])
    
    return np.mean(scores) if scores else np.nan

def calculate_quality_score(row):
    """Calcule le score de qualité"""
    scores = []
    weights = {
        'return_on_equity': 0.2,
        'return_on_assets': 0.2,
        'operating_margins': 0.2,
        'profit_margins': 0.2,
        'debt_to_equity': -0.1,  # Plus bas est meilleur
        'current_ratio': 0.1
    }
    
    for metric, weight in weights.items():
        if pd.notna(row[metric]):
            scores.append(weight * row[metric])
    
    return np.mean(scores) if scores else np.nan

def calculate_growth_score(row):
    """Calcule le score de croissance"""
    scores = []
    weights = {
        'earnings_growth': 0.5,
        'revenue_growth': 0.5
    }
    
    for metric, weight in weights.items():
        if pd.notna(row[metric]):
            scores.append(weight * row[metric])
    
    return np.mean(scores) if scores else np.nan

def calculate_financial_health_score(row):
    """Calcule le score de solidité financière"""
    scores = []
    weights = {
        'debt_to_equity': -0.3,  # Plus bas est meilleur
        'current_ratio': 0.3,
        'quick_ratio': 0.2,
        'interest_coverage': 0.2
    }
    
    for metric, weight in weights.items():
        if pd.notna(row[metric]):
            scores.append(weight * row[metric])
    
    return np.mean(scores) if scores else np.nan

def calculate_momentum_score(row):
    """Calcule le score de momentum"""
    scores = []
    weights = {
        'market_daily_change': 0.5,
        'beta': -0.5  # Plus bas est meilleur
    }
    
    for metric, weight in weights.items():
        if pd.notna(row[metric]):
            scores.append(weight * row[metric])
    
    return np.mean(scores) if scores else np.nan

def calculate_dividend_score(row):
    """Calcule le score de dividende"""
    if pd.notna(row['dividend_yield']):
        return row['dividend_yield']
    return np.nan

def calculate_overvaluation_score(row):
    """Calcule un score de surévaluation combinant différents aspects"""
    scores = []
    weights = {
        'valuation_score': -0.4,  # Plus bas est meilleur
        'quality_score': 0.2,     # Plus haut est meilleur
        'growth_score': 0.2,      # Plus haut est meilleur
        'financial_health_score': 0.2  # Plus haut est meilleur
    }
    
    for metric, weight in weights.items():
        if pd.notna(row[metric]):
            scores.append(weight * row[metric])
    
    return np.mean(scores) if scores else np.nan

def clean_value(value):
    """Nettoie une valeur en remplaçant les valeurs aberrantes par None"""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if np.isinf(value) or np.isnan(value):
            return None
        if value < -1000 or value > 1000:
            return None
    return value

def extract_features(symbol, stock):
    """Extraction des features pour chaque action"""
    overview = stock.get('overview', {}).get('data', {})
    balance_sheet = stock.get('balance_sheet', {}).get('data', {})
    cash_flow = stock.get('cash_flow', {}).get('data', {})
    market_data = stock.get('market_data', {}).get('data', {})
    income_statement = stock.get('income_statement', {}).get('data', {})
    
    features = {}
    features['symbol'] = symbol
    features['sector'] = overview.get('sector', None)
    features['industry'] = overview.get('industry', None)
    for key in overview:
        features[key] = clean_value(overview.get(key, None))
    for key in balance_sheet:
        features[f'bs_{key}'] = clean_value(balance_sheet.get(key, None))
    for key in cash_flow:
        features[f'cf_{key}'] = clean_value(cash_flow.get(key, None))
    for key in market_data:
        features[f'market_{key}'] = clean_value(market_data.get(key, None))
    for key in income_statement:
        features[f'is_{key}'] = clean_value(income_statement.get(key, None))
    return features

# Chargement des données
with open('dataset/stocks_data.json', 'r') as f:
    stocks_data = json.load(f)['stocks']
    print(f"\nNombre total d'actions dans stocks_data.json : {len(stocks_data)}")

data = [extract_features(sym, stock) for sym, stock in stocks_data.items()]
df = pd.DataFrame(data)

# Liste des colonnes à conserver
columns_to_keep = [
    'pe_ratio', 'pb_ratio', 'ps_ratio', 'beta', 'earnings_growth', 'revenue_growth',
    'profit_margins', 'operating_margins', 'return_on_equity', 'return_on_assets',
    'debt_to_equity', 'current_ratio', 'name', 'dividend_yield', '52_week_high',
    '52_week_low', 'target_mean_price', 'target_median_price', 'recommendation',
    'number_of_analysts', 'quick_ratio', 'interest_coverage', 'enterprise_to_revenue',
    'enterprise_to_ebitda', 'bs_total_liabilities', 'bs_total_equity', 'bs_cash',
    'bs_short_term_investments', 'market_current_price', 'market_daily_change'
]

# On conserve symbol, sector, industry pour l'identification
id_columns = ['symbol', 'sector', 'industry']
final_columns = id_columns + [col for col in columns_to_keep if col in df.columns]
df_final = df[final_columns]

# Calcul de peg_ratio si pe_ratio et earnings_growth sont présents
if 'pe_ratio' in df_final.columns and 'earnings_growth' in df_final.columns:
    df_final['peg_ratio'] = df_final['pe_ratio'] / df_final['earnings_growth'].replace(0, np.nan)
    print("\npeg_ratio calculé à partir de pe_ratio et earnings_growth.")

# Calcul du pourcentage de valeurs manquantes par ligne
missing_percentage = df_final.isnull().mean(axis=1) * 100

# Ne garder que les lignes avec moins de 50% de valeurs manquantes
df_final = df_final[missing_percentage < 50]

print(f"\nNombre d'actions avant filtrage : {len(df)}")
print(f"Nombre d'actions après filtrage : {len(df_final)}")
print(f"Nombre de colonnes extraites : {len(df_final.columns)}")
print(f"Colonnes conservées : {df_final.columns.tolist()}")

# Calcul des scores
print("\nCalcul des scores...")
df_final['valuation_score'] = df_final.apply(calculate_valuation_score, axis=1)
df_final['quality_score'] = df_final.apply(calculate_quality_score, axis=1)
df_final['growth_score'] = df_final.apply(calculate_growth_score, axis=1)
df_final['financial_health_score'] = df_final.apply(calculate_financial_health_score, axis=1)
df_final['momentum_score'] = df_final.apply(calculate_momentum_score, axis=1)
df_final['dividend_score'] = df_final.apply(calculate_dividend_score, axis=1)

# Normalisation des scores
score_columns = ['valuation_score', 'quality_score', 'growth_score', 
                'financial_health_score', 'momentum_score', 'dividend_score']

for col in score_columns:
    df_final[col] = normalize_series(df_final[col])

# Calcul du score de surévaluation
print("\nCalcul du score de surévaluation...")
df_final['overvaluation_score'] = df_final.apply(calculate_overvaluation_score, axis=1)
df_final['overvaluation_score'] = normalize_series(df_final['overvaluation_score'])

# Interprétation du score :
# - Proche de 1 : Action potentiellement surévaluée
# - Proche de 0 : Action potentiellement sous-évaluée
# - Proche de 0.5 : Action correctement valorisée

# Sauvegarde du dataset filtré avec scores
output_path = 'dataset/stocks_extracted_filtered.csv'
df_final.to_csv(output_path, index=False)
print(f"\nDataset filtré avec scores sauvegardé : {output_path}") 