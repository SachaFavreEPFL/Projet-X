import json
import time
from datetime import datetime
import yfinance as yf
import os
from typing import Dict, Optional

class StockAnalyzer:
    def __init__(self):
        self.input_file = "dataset/stocks_to_analyze.json"
        self.output_file = "dataset/stocks_data.json"
        self.request_delay = 1  # Délai entre chaque requête en secondes

    def get_stock_data(self, symbol: str) -> Optional[Dict]:
        """Récupère les données pour un symbole donné"""
        try:
            print(f"\nAnalyse de {symbol}")
            
            stock = yf.Ticker(symbol)
            info = stock.info
            
            if not info:
                print(f"❌ Données non disponibles pour {symbol}")
                return None
                
            # Données essentielles
            overview = {
                "symbol": symbol,
                "sector": info.get("sector", ""),
                "pe_ratio": float(info.get("trailingPE", 0)),
                "pb_ratio": float(info.get("priceToBook", 0)),
                "ps_ratio": float(info.get("priceToSalesTrailing12Months", 0)),
                "market_cap": float(info.get("marketCap", 0)),
                "beta": float(info.get("beta", 0)),
                "earnings_growth": float(info.get("earningsQuarterlyGrowth", 0)) * 100,
                "revenue_growth": float(info.get("revenueGrowth", 0)) * 100,
                "profit_margins": float(info.get("profitMargins", 0)) * 100,
                "operating_margins": float(info.get("operatingMargins", 0)) * 100,
                "return_on_equity": float(info.get("returnOnEquity", 0)) * 100,
                "return_on_assets": float(info.get("returnOnAssets", 0)) * 100,
                "debt_to_equity": float(info.get("debtToEquity", 0)),
                "current_ratio": float(info.get("currentRatio", 0))
            }
            
            # Données historiques (1 an)
            hist = stock.history(period="1y")
            if not hist.empty:
                historical_data = {
                    "current_price": float(hist["Close"].iloc[-1]),
                    "52_week_high": float(hist["High"].max()),
                    "52_week_low": float(hist["Low"].min()),
                    "avg_volume": float(hist["Volume"].mean()),
                    "volatility": float(hist["Close"].pct_change().std() * 100),  # Volatilité en %
                    "price_change_1y": float(((hist["Close"].iloc[-1] / hist["Close"].iloc[0]) - 1) * 100)  # Variation sur 1 an en %
                }
            else:
                historical_data = {
                    "current_price": 0,
                    "52_week_high": 0,
                    "52_week_low": 0,
                    "avg_volume": 0,
                    "volatility": 0,
                    "price_change_1y": 0
                }
            
            result = {
                "overview": {
                    "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "data": overview
                },
                "historical": {
                    "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "data": historical_data
                }
            }
            
            print(f"✅ Données récupérées pour {symbol}")
            return result
            
        except Exception as e:
            print(f"❌ Erreur pour {symbol}: {str(e)}")
            return None

    def save_data(self, data: Dict) -> None:
        """Sauvegarde les données dans le fichier"""
        try:
            os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
            
            output_data = {
                "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "stocks": {}
            }
            
            if os.path.exists(self.output_file) and os.path.getsize(self.output_file) > 0:
                with open(self.output_file, 'r') as f:
                    output_data = json.load(f)
            
            if isinstance(data, dict) and "overview" in data and "data" in data["overview"]:
                symbol = data["overview"]["data"]["symbol"]
                output_data["stocks"][symbol] = data
            
            with open(self.output_file, 'w') as f:
                json.dump(output_data, f, indent=4)
                
            print(f"✅ Données sauvegardées dans {self.output_file}")
                
        except Exception as e:
            print(f"❌ Erreur de sauvegarde: {str(e)}")

    def run_analysis(self) -> None:
        """Lance l'analyse des actions"""
        try:
            with open(self.input_file, 'r') as f:
                symbols = json.load(f).get("stocks", [])
            
            if not symbols:
                print("❌ Aucun symbole à analyser")
                return

            print(f"\nAnalyse de {len(symbols)} actions")
            
            for symbol in symbols:
                data = self.get_stock_data(symbol)
                if data:
                    self.save_data(data)
                time.sleep(self.request_delay)
                
            print("\nAnalyse terminée")
                
        except KeyboardInterrupt:
            print("\n🛑 Arrêt demandé par l'utilisateur")
        except Exception as e:
            print(f"\n❌ Erreur: {str(e)}")

if __name__ == "__main__":
    analyzer = StockAnalyzer()
    analyzer.run_analysis() 