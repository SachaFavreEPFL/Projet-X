import json
import time
from datetime import datetime, timedelta
import yfinance as yf
import os
from typing import Dict, Any, Optional, Union
import pandas as pd

class StockAnalyzer:
    def __init__(self):
        self.input_file = "dataset/stocks_to_analyze.json"
        self.output_file = "dataset/stocks_data.json"
        self.log_file = "dataset/analysis_log.json"
        self.history_file = "dataset/stocks_history.json"
        self.request_delay = 10  # Délai entre chaque requête en secondes
        self.history_data = self.load_history()

    def load_history(self) -> Dict:
        """Charge les données historiques"""
        try:
            # Structure de base pour l'historique
            base_structure = {
                "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "stocks": {}
            }
            
            # Si le fichier existe et n'est pas vide, charger les données
            if os.path.exists(self.history_file) and os.path.getsize(self.history_file) > 0:
                try:
                    with open(self.history_file, 'r') as f:
                        data = json.load(f)
                        # Vérifier si la structure est correcte
                        if isinstance(data, dict) and "stocks" in data:
                            return data
                except json.JSONDecodeError:
                    print("⚠️ Fichier d'historique corrompu, création d'une nouvelle structure")
            
            # Si le fichier n'existe pas ou est vide, créer le dossier et sauvegarder la structure de base
            os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
            with open(self.history_file, 'w') as f:
                json.dump(base_structure, f, indent=4)
            
            return base_structure
            
        except Exception as e:
            print(f"⚠️ Erreur lors du chargement de l'historique : {str(e)}")
            return base_structure

    def save_history(self, symbol: str, data: Dict) -> None:
        """Sauvegarde les données dans l'historique"""
        try:
            current_date = datetime.now().strftime("%Y-%m-%d")
            
            if symbol not in self.history_data["stocks"]:
                self.history_data["stocks"][symbol] = {}
                
            # Sauvegarder les données historiques
            self.history_data["stocks"][symbol][current_date] = {
                "market_data": data["market_data"]["data"],
                "valuation_metrics": {
                    "pe_ratio": data["overview"]["data"]["pe_ratio"],
                    "pb_ratio": data["overview"]["data"]["pb_ratio"],
                    "ps_ratio": data["overview"]["data"]["ps_ratio"]
                },
                "quality_metrics": {
                    "roe": data["overview"]["data"]["return_on_equity"],
                    "operating_margins": data["overview"]["data"]["operating_margins"],
                    "debt_to_equity": data["overview"]["data"]["debt_to_equity"],
                    "interest_coverage": data["overview"]["data"]["interest_coverage"]
                },
                "growth_metrics": {
                    "revenue_growth": data["overview"]["data"]["revenue_growth"],
                    "earnings_growth": data["overview"]["data"]["earnings_growth"]
                }
            }
            
            # Créer le dossier dataset s'il n'existe pas
            os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
            
            with open(self.history_file, 'w') as f:
                json.dump(self.history_data, f, indent=4)
                
        except Exception as e:
            print(f"⚠️ Erreur lors de la sauvegarde de l'historique : {str(e)}")

    def check_last_update(self, symbol: str) -> bool:
        """Vérifie si l'action a été mise à jour dans les dernières 24h"""
        try:
            if os.path.exists(self.output_file):
                with open(self.output_file, 'r') as f:
                    data = json.load(f)
                    if symbol in data["stocks"]:
                        last_update = datetime.fromisoformat(data["stocks"][symbol]["overview"]["last_update"])
                        time_diff = datetime.now() - last_update
                        if time_diff < timedelta(hours=24):
                            print(f"⏭️ Action {symbol} mise à jour il y a moins de 24h ({int(time_diff.total_seconds()/3600)}h)")
                            return False
            return True
        except Exception as e:
            print(f"⚠️ Erreur lors de la vérification de la dernière mise à jour : {str(e)}")
            return True

    def log_attempt(self, symbol: str, success: bool, error_message: str = None) -> None:
        """Enregistre une tentative dans le fichier de log"""
        try:
            if os.path.exists(self.log_file):
                with open(self.log_file, 'r') as f:
                    logs = json.load(f)
            else:
                logs = {"attempts": []}
            
            log_entry = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "symbol": symbol,
                "success": success,
                "error_message": error_message,
                "hour_of_day": datetime.now().hour,
                "day_of_week": datetime.now().strftime("%A")
            }
            
            logs["attempts"].append(log_entry)
            
            with open(self.log_file, 'w') as f:
                json.dump(logs, f, indent=4)
                
        except Exception as e:
            print(f"⚠️ Erreur lors de la journalisation : {str(e)}")

    def get_stock_data(self, symbol: str) -> Optional[Dict]:
        """Récupère les données pour un symbole donné"""
        try:
            print(f"\n{'='*50}")
            print(f"TENTATIVE DE RÉCUPÉRATION POUR {symbol}")
            print(f"{'='*50}")
            
            stock = yf.Ticker(symbol)
            
            # Récupération de toutes les données en une seule fois
            print("\nRécupération des données...")
            info = stock.info
            if not info:
                print("❌ Données non disponibles")
                self.log_attempt(symbol, False, "Données non disponibles")
                return None
                
            # Données fondamentales
            overview = {
                "symbol": symbol,
                "name": info.get("longName", ""),
                "sector": info.get("sector", ""),
                "industry": info.get("industry", ""),
                "pe_ratio": float(info.get("trailingPE", 0)),
                "pb_ratio": float(info.get("priceToBook", 0)),
                "ps_ratio": float(info.get("priceToSalesTrailing12Months", 0)),
                "market_cap": float(info.get("marketCap", 0)),
                "dividend_yield": float(info.get("dividendYield", 0)) * 100,
                "beta": float(info.get("beta", 0)),
                "52_week_high": float(info.get("fiftyTwoWeekHigh", 0)),
                "52_week_low": float(info.get("fiftyTwoWeekLow", 0)),
                "target_mean_price": float(info.get("targetMeanPrice", 0)),
                "target_median_price": float(info.get("targetMedianPrice", 0)),
                "recommendation": info.get("recommendationKey", ""),
                "number_of_analysts": int(info.get("numberOfAnalystOpinions", 0)),
                "earnings_growth": float(info.get("earningsQuarterlyGrowth", 0)) * 100,
                "revenue_growth": float(info.get("revenueGrowth", 0)) * 100,
                "profit_margins": float(info.get("profitMargins", 0)) * 100,
                "operating_margins": float(info.get("operatingMargins", 0)) * 100,
                "return_on_equity": float(info.get("returnOnEquity", 0)) * 100,
                "return_on_assets": float(info.get("returnOnAssets", 0)) * 100,
                "debt_to_equity": float(info.get("debtToEquity", 0)),
                "current_ratio": float(info.get("currentRatio", 0)),
                "quick_ratio": float(info.get("quickRatio", 0)),
                "interest_coverage": float(info.get("interestCoverage", 0)),
                "enterprise_value": float(info.get("enterpriseValue", 0)),
                "enterprise_to_revenue": float(info.get("enterpriseToRevenue", 0)),
                "enterprise_to_ebitda": float(info.get("enterpriseToEbitda", 0))
            }
            
            # Données de marché (dernière journée)
            market_data = stock.history(period="1d")
            if market_data.empty:
                print("❌ Données de marché non disponibles")
                self.log_attempt(symbol, False, "Données de marché non disponibles")
                return None
                
            current_price = market_data["Close"].iloc[-1]
            previous_close = market_data["Open"].iloc[0]
            daily_change = ((current_price - previous_close) / previous_close * 100) if previous_close else 0
            
            market = {
                "current_price": float(current_price),
                "volume": float(market_data["Volume"].iloc[-1]),
                "daily_change": float(daily_change)
            }
            
            # Données financières
            financials = stock.financials
            income_statement = {
                "revenue": float(financials.loc["Total Revenue"].iloc[0] if financials is not None and not financials.empty and "Total Revenue" in financials.index else 0),
                "gross_profit": float(financials.loc["Gross Profit"].iloc[0] if financials is not None and not financials.empty and "Gross Profit" in financials.index else 0),
                "net_income": float(financials.loc["Net Income"].iloc[0] if financials is not None and not financials.empty and "Net Income" in financials.index else 0),
                "ebit": float(financials.loc["EBIT"].iloc[0] if financials is not None and not financials.empty and "EBIT" in financials.index else 0),
                "ebitda": float(financials.loc["EBITDA"].iloc[0] if financials is not None and not financials.empty and "EBITDA" in financials.index else 0)
            }
            
            balance_sheet = stock.balance_sheet
            balance = {
                "total_assets": float(balance_sheet.loc["Total Assets"].iloc[0] if balance_sheet is not None and not balance_sheet.empty and "Total Assets" in balance_sheet.index else 0),
                "total_liabilities": float(balance_sheet.loc["Total Liabilities"].iloc[0] if balance_sheet is not None and not balance_sheet.empty and "Total Liabilities" in balance_sheet.index else 0),
                "total_equity": float(balance_sheet.loc["Total Stockholder Equity"].iloc[0] if balance_sheet is not None and not balance_sheet.empty and "Total Stockholder Equity" in balance_sheet.index else 0),
                "total_debt": float(balance_sheet.loc["Total Debt"].iloc[0] if balance_sheet is not None and not balance_sheet.empty and "Total Debt" in balance_sheet.index else 0),
                "cash": float(balance_sheet.loc["Cash"].iloc[0] if balance_sheet is not None and not balance_sheet.empty and "Cash" in balance_sheet.index else 0),
                "short_term_investments": float(balance_sheet.loc["Short Term Investments"].iloc[0] if balance_sheet is not None and not balance_sheet.empty and "Short Term Investments" in balance_sheet.index else 0),
                "inventory": float(balance_sheet.loc["Inventory"].iloc[0] if balance_sheet is not None and not balance_sheet.empty and "Inventory" in balance_sheet.index else 0),
                "accounts_receivable": float(balance_sheet.loc["Accounts Receivable"].iloc[0] if balance_sheet is not None and not balance_sheet.empty and "Accounts Receivable" in balance_sheet.index else 0)
            }
            
            cash_flow = stock.cashflow
            cash = {
                "operating_cash_flow": float(cash_flow.loc["Operating Cash Flow"].iloc[0] if cash_flow is not None and not cash_flow.empty and "Operating Cash Flow" in cash_flow.index else 0),
                "investing_cash_flow": float(cash_flow.loc["Investing Cash Flow"].iloc[0] if cash_flow is not None and not cash_flow.empty and "Investing Cash Flow" in cash_flow.index else 0),
                "financing_cash_flow": float(cash_flow.loc["Financing Cash Flow"].iloc[0] if cash_flow is not None and not cash_flow.empty and "Financing Cash Flow" in cash_flow.index else 0),
                "free_cash_flow": float(cash_flow.loc["Free Cash Flow"].iloc[0] if cash_flow is not None and not cash_flow.empty and "Free Cash Flow" in cash_flow.index else 0),
                "capital_expenditure": float(cash_flow.loc["Capital Expenditure"].iloc[0] if cash_flow is not None and not cash_flow.empty and "Capital Expenditure" in cash_flow.index else 0)
            }
            
            result = {
                "overview": {
                    "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "data": overview
                },
                "income_statement": {
                    "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "data": income_statement
                },
                "balance_sheet": {
                    "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "data": balance
                },
                "cash_flow": {
                    "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "data": cash
                },
                "market_data": {
                    "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "data": market
                }
            }
            
            print(f"✅ Données récupérées avec succès pour {symbol}")
            self.log_attempt(symbol, True)
            return result
            
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Erreur lors de la récupération des données: {error_msg}")
            self.log_attempt(symbol, False, error_msg)
            return None

    def save_data(self, data: Dict) -> None:
        """Sauvegarde les données dans le fichier"""
        try:
            # Créer le dossier dataset s'il n'existe pas
            os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
            
            # Initialiser la structure de base
            output_data = {
                "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "stocks": {}
            }
            
            # Charger les données existantes si le fichier existe et n'est pas vide
            if os.path.exists(self.output_file) and os.path.getsize(self.output_file) > 0:
                try:
                    with open(self.output_file, 'r') as f:
                        existing_data = json.load(f)
                        if isinstance(existing_data, dict) and "stocks" in existing_data:
                            output_data = existing_data
                except json.JSONDecodeError:
                    print("⚠️ Fichier JSON corrompu, création d'une nouvelle structure")
            
            # Mettre à jour les données
            output_data["last_update"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if isinstance(data, dict) and "overview" in data and "data" in data["overview"]:
                symbol = data["overview"]["data"]["symbol"]
                output_data["stocks"][symbol] = data
                # Sauvegarder dans l'historique
                self.save_history(symbol, data)
            else:
                output_data["stocks"].update(data)
            
            # Sauvegarder les données
            with open(self.output_file, 'w') as f:
                json.dump(output_data, f, indent=4)
                
            print(f"✅ Données sauvegardées avec succès dans {self.output_file}")
                
        except Exception as e:
            print(f"❌ Erreur lors de la sauvegarde des données: {str(e)}")
            raise

    def run_analysis(self) -> None:
        """Lance l'analyse en boucle continue"""
        while True:
            try:
                symbols = self.load_symbols()
                if not symbols:
                    print("❌ Aucun symbole à analyser")
                    time.sleep(3600)  # Attendre 1 heure avant de réessayer
                    continue

                total_actions = len(symbols)
                print(f"\nDébut de l'analyse de {total_actions} actions")
                print("Ordre d'analyse :")
                for i, symbol in enumerate(symbols, 1):
                    print(f"{i}/{total_actions} - {symbol}")
                print("\n" + "="*50)
                
                actions_analysées = 0
                dernière_analyse_réussie = True  # Pour suivre si la dernière action a été analysée avec succès
                
                for index, symbol in enumerate(symbols, 1):
                    print(f"\n[{index}/{total_actions}] Analyse de {symbol} ({int(index/total_actions*100)}%)")
                    if self.check_last_update(symbol):
                        try:
                            data = self.get_stock_data(symbol)
                            if data:
                                self.save_data(data)
                                actions_analysées += 1
                                dernière_analyse_réussie = True
                                time.sleep(self.request_delay)  # Délai uniquement après une analyse réussie
                            else:
                                print(f"⏭️ Action {symbol} non disponible, passage à la suivante")
                                dernière_analyse_réussie = True
                                continue
                        except Exception as e:
                            error_msg = str(e)
                            if "possibly delisted" in error_msg or "No data found" in error_msg:
                                print(f"⏭️ Action {symbol} délistée, passage à la suivante")
                                dernière_analyse_réussie = True
                                continue
                            else:
                                print(f"\n❌ Erreur lors de l'analyse de {symbol}: {error_msg}")
                                print("⏳ Attente de 5 minutes avant de réessayer...")
                                time.sleep(300)  # Attente de 5 minutes en cas d'erreur
                                dernière_analyse_réussie = False
                    else:
                        if not dernière_analyse_réussie:
                            time.sleep(self.request_delay)  # Délai si la dernière action a échoué
                        print(f"⏭️ {symbol} déjà à jour, passage à la suivante")
                        dernière_analyse_réussie = True

                print(f"\nCycle d'analyse terminé : {actions_analysées}/{total_actions} actions mises à jour")
                print("Attente avant le prochain cycle...")
                time.sleep(3600)  # Attendre 1 heure avant de recommencer
                
            except KeyboardInterrupt:
                print("\n🛑 Arrêt demandé par l'utilisateur")
                break
            except Exception as e:
                print(f"\n❌ Erreur générale lors de l'analyse : {str(e)}")
                time.sleep(300)  # Attendre 5 minutes en cas d'erreur générale

    def load_symbols(self) -> list:
        """Charge la liste des symboles à analyser"""
        try:
            with open(self.input_file, 'r') as f:
                data = json.load(f)
                return data.get("stocks", [])
        except Exception as e:
            print(f"❌ Erreur lors de la lecture du fichier {self.input_file}: {str(e)}")
            return []

if __name__ == "__main__":
    analyzer = StockAnalyzer()
    analyzer.run_analysis() 