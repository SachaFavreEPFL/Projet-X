import json
import time
from datetime import datetime
import yfinance as yf
import os
from typing import Dict, Any, Optional, Union
import pandas as pd

class DelayTester:
    def __init__(self):
        self.output_file = "dataset/stocks_data.json"
        self.symbol = "AAPL"
        self.base_delay = 60  # Délai initial de 1 minute (60 secondes)
        self.max_attempts = 20  # Augmentation du nombre de tentatives
        self.current_delay = self.base_delay
        self.max_delay = 7200  # Maximum 2 heures (7200 secondes)
        self.request_delay = 10  # Délai entre chaque requête en secondes
        self.delay_increment = 60  # Incrément de 1 minute (60 secondes)

    def get_stock_data(self) -> Optional[Dict]:
        """Récupère les données pour AAPL"""
        try:
            print(f"\n{'='*50}")
            print(f"TENTATIVE DE RÉCUPÉRATION POUR {self.symbol}")
            print(f"Délai actuel: {self.current_delay} secondes")
            print(f"{'='*50}")
            
            # Récupération des données via yfinance
            stock = yf.Ticker(self.symbol)
            
            # 1. Données fondamentales
            print("\n1. Récupération des données fondamentales...")
            info = stock.info
            if not info:
                print("❌ Données fondamentales non disponibles")
                return None
            time.sleep(self.request_delay)  # Délai entre les requêtes
                
            overview = {
                "symbol": self.symbol,
                "name": info.get("longName", ""),
                "pe_ratio": float(info.get("trailingPE", 0)),
                "pb_ratio": float(info.get("priceToBook", 0)),
                "ps_ratio": float(info.get("priceToSalesTrailing12Months", 0)),
                "market_cap": float(info.get("marketCap", 0)),
                "dividend_yield": float(info.get("dividendYield", 0)) * 100
            }
            
            # 2. Compte de résultat
            print("2. Récupération du compte de résultat...")
            financials = stock.financials
            if financials is None or financials.empty:
                print("⚠️ Compte de résultat non disponible - continuation avec les données disponibles")
                income_statement = {
                    "revenue": 0,
                    "gross_profit": 0,
                    "net_income": 0
                }
            else:
                income_statement = {
                    "revenue": float(financials.loc["Total Revenue"].iloc[0] if "Total Revenue" in financials.index else 0),
                    "gross_profit": float(financials.loc["Gross Profit"].iloc[0] if "Gross Profit" in financials.index else 0),
                    "net_income": float(financials.loc["Net Income"].iloc[0] if "Net Income" in financials.index else 0)
                }
            time.sleep(self.request_delay)  # Délai entre les requêtes
            
            # 3. Bilan
            print("3. Récupération du bilan...")
            balance_sheet = stock.balance_sheet
            if balance_sheet is None or balance_sheet.empty:
                print("⚠️ Bilan non disponible - continuation avec les données disponibles")
                balance = {
                    "total_assets": 0,
                    "total_liabilities": 0,
                    "total_equity": 0
                }
            else:
                balance = {
                    "total_assets": float(balance_sheet.loc["Total Assets"].iloc[0] if "Total Assets" in balance_sheet.index else 0),
                    "total_liabilities": float(balance_sheet.loc["Total Liabilities"].iloc[0] if "Total Liabilities" in balance_sheet.index else 0),
                    "total_equity": float(balance_sheet.loc["Total Stockholder Equity"].iloc[0] if "Total Stockholder Equity" in balance_sheet.index else 0)
                }
            time.sleep(self.request_delay)  # Délai entre les requêtes
            
            # 4. Flux de trésorerie
            print("4. Récupération des flux de trésorerie...")
            cash_flow = stock.cashflow
            if cash_flow is None or cash_flow.empty:
                print("⚠️ Flux de trésorerie non disponibles - continuation avec les données disponibles")
                cash = {
                    "operating_cash_flow": 0,
                    "investing_cash_flow": 0,
                    "financing_cash_flow": 0
                }
            else:
                cash = {
                    "operating_cash_flow": float(cash_flow.loc["Operating Cash Flow"].iloc[0] if "Operating Cash Flow" in cash_flow.index else 0),
                    "investing_cash_flow": float(cash_flow.loc["Investing Cash Flow"].iloc[0] if "Investing Cash Flow" in cash_flow.index else 0),
                    "financing_cash_flow": float(cash_flow.loc["Financing Cash Flow"].iloc[0] if "Financing Cash Flow" in cash_flow.index else 0)
                }
            time.sleep(self.request_delay)  # Délai entre les requêtes
            
            # 5. Données de marché
            print("5. Récupération des données de marché...")
            market_data = stock.history(period="1d")
            if market_data.empty:
                print("❌ Données de marché non disponibles")
                return None
                
            current_price = market_data["Close"].iloc[-1]
            previous_close = market_data["Open"].iloc[0]
            daily_change = ((current_price - previous_close) / previous_close * 100) if previous_close else 0
            
            market = {
                "current_price": float(current_price),
                "volume": float(market_data["Volume"].iloc[-1]),
                "daily_change": float(daily_change)
            }
            
            # Construction du résultat
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
            
            print(f"✅ Données récupérées avec succès pour {self.symbol}")
            return result
            
        except Exception as e:
            print(f"❌ Erreur lors de la récupération des données: {str(e)}")
            return None

    def save_data(self, data: Dict) -> None:
        """Sauvegarde les données dans le fichier"""
        try:
            # Charger les données existantes
            if os.path.exists(self.output_file):
                with open(self.output_file, 'r') as f:
                    output_data = json.load(f)
            else:
                output_data = {"last_update": "", "stocks": {}}
            
            # Mettre à jour les données
            output_data["stocks"][self.symbol] = data
            output_data["last_update"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Sauvegarder
            with open(self.output_file, 'w') as f:
                json.dump(output_data, f, indent=4)
                
            print("✅ Données sauvegardées avec succès")
            
        except Exception as e:
            print(f"❌ Erreur lors de la sauvegarde des données: {str(e)}")

    def run_test(self) -> None:
        """Lance le test avec délai croissant"""
        attempt = 1
        
        while attempt <= self.max_attempts:
            print(f"\nTentative {attempt}/{self.max_attempts}")
            print(f"Délai actuel: {self.current_delay} secondes")
            
            data = self.get_stock_data()
            if data:
                self.save_data(data)
                print(f"\n✅ Test réussi à la tentative {attempt}")
                break
            
            if attempt < self.max_attempts:
                # Augmentation du délai de 1 minute
                self.current_delay = min(self.current_delay + self.delay_increment, self.max_delay)
                print(f"\n⏳ Attente de {self.current_delay} secondes avant la prochaine tentative...")
                time.sleep(self.current_delay)
            
            attempt += 1
        
        if attempt > self.max_attempts:
            print(f"\n❌ Échec après {self.max_attempts} tentatives")
            print("Recommandation : Attendre 1-2 heures avant de réessayer")

if __name__ == "__main__":
    tester = DelayTester()
    tester.run_test() 