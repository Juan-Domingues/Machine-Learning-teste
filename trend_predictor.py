"""
Modelo de Machine Learning para Previsão de Tendências
Tech Challenge - Fase 2

Este script implementa um modelo para prever tendências (↑ ou ↓) em dados de séries temporais
com acurácia mínima de 75% usando os últimos 30 dias como conjunto de teste.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Bibliotecas de Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Bibliotecas para análise técnica
import yfinance as yf
import ta

class TrendPredictor:
    """
    Classe para previsão de tendências usando Machine Learning
    """
    
    def __init__(self, symbol='PETR4.SA', period='2y'):
        """
        Inicializa o preditor de tendências
        
        Args:
            symbol (str): Símbolo do ativo financeiro
            period (str): Período de dados históricos
        """
        self.symbol = symbol
        self.period = period
        self.data = None
        self.model = None
        self.scaler = StandardScaler()
        self.features = []
        
    def load_data(self):
        """
        Carrega dados financeiros usando yfinance ou cria dados sintéticos
        """
        print(f"Tentando carregar dados para {self.symbol}...")
        
        try:
            # Tentar baixar dados do Yahoo Finance
            ticker = yf.Ticker(self.symbol)
            self.data = ticker.history(period=self.period)
            
            if self.data.empty:
                raise ValueError("Dados vazios")
                
            print(f"✅ Dados reais carregados: {len(self.data)} registros")
            print(f"Período: {self.data.index.min()} até {self.data.index.max()}")
            
        except Exception as e:
            print(f"⚠️ Erro ao carregar dados reais: {str(e)}")
            print("🔄 Criando dados sintéticos...")
            self._create_synthetic_data()
            
        return self.data
    
    def _create_synthetic_data(self):
        """
        Cria dados sintéticos mais realistas para PETR4.SA
        """
        n_days = 730  # 2 anos
        dates = pd.date_range(start='2022-01-01', periods=n_days, freq='D')
        
        # Preço inicial similar à PETR4
        initial_price = 30.0
        
        # Criar tendências e padrões mais realistas
        trend_cycles = np.sin(np.linspace(0, 8*np.pi, n_days)) * 0.002
        momentum = np.cumsum(np.random.normal(0, 0.0008, n_days))
        
        # Gerar retornos com padrões
        returns = []
        for i in range(n_days):
            base_return = trend_cycles[i] + momentum[i]
            noise = np.random.normal(0, 0.018)
            volatility_regime = 1 + 0.3 * np.sin(i * 0.03)
            
            returns.append(base_return + noise * volatility_regime)
        
        # Criar preços
        prices = [initial_price]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        # Criar OHLCV sintético
        data = []
        for i, (date, price) in enumerate(zip(dates, prices)):
            high = price * (1 + abs(np.random.normal(0, 0.012)))
            low = price * (1 - abs(np.random.normal(0, 0.012)))
            open_price = price * (1 + np.random.normal(0, 0.008))
            volume = np.random.randint(5000000, 50000000)
            
            data.append({
                'Open': open_price,
                'High': max(high, price, open_price),
                'Low': min(low, price, open_price),
                'Close': price,
                'Volume': volume
            })
        
        self.data = pd.DataFrame(data, index=dates)
        print(f"📊 Dados sintéticos criados: {len(self.data)} registros")
        print(f"Período: {self.data.index.min()} até {self.data.index.max()}")
    
    def create_technical_indicators(self):
        """
        Cria indicadores técnicos para features do modelo
        """
        if self.data is None:
            raise ValueError("Dados não carregados. Execute load_data() primeiro.")
            
        print("Criando indicadores técnicos...")
        
        try:
            # Médias móveis usando TA-Lib
            self.data['SMA_5'] = ta.trend.sma_indicator(self.data['Close'], window=5)
            self.data['SMA_10'] = ta.trend.sma_indicator(self.data['Close'], window=10)
            self.data['SMA_20'] = ta.trend.sma_indicator(self.data['Close'], window=20)
            self.data['EMA_12'] = ta.trend.ema_indicator(self.data['Close'], window=12)
            self.data['EMA_26'] = ta.trend.ema_indicator(self.data['Close'], window=26)
            
            # RSI
            self.data['RSI'] = ta.momentum.rsi(self.data['Close'], window=14)
            
            # MACD
            macd = ta.trend.MACD(self.data['Close'])
            self.data['MACD'] = macd.macd()
            self.data['MACD_signal'] = macd.macd_signal()
            self.data['MACD_histogram'] = macd.macd_diff()
            
            # Bollinger Bands
            bollinger = ta.volatility.BollingerBands(self.data['Close'])
            self.data['BB_upper'] = bollinger.bollinger_hband()
            self.data['BB_lower'] = bollinger.bollinger_lband()
            self.data['BB_middle'] = bollinger.bollinger_mavg()
            
            # Estocástico
            self.data['Stoch_K'] = ta.momentum.stoch(self.data['High'], self.data['Low'], self.data['Close'])
            self.data['Stoch_D'] = ta.momentum.stoch_signal(self.data['High'], self.data['Low'], self.data['Close'])
            
            # Williams %R
            self.data['Williams_R'] = ta.momentum.williams_r(self.data['High'], self.data['Low'], self.data['Close'])
            
            # Volume indicators
            self.data['Volume_SMA'] = ta.volume.volume_sma(self.data['Close'], self.data['Volume'], window=20)
            
            # Volatilidade
            self.data['ATR'] = ta.volatility.average_true_range(self.data['High'], self.data['Low'], self.data['Close'])
            
            print("✅ Indicadores TA-Lib criados com sucesso!")
            
        except Exception as e:
            print(f"⚠️ Erro com TA-Lib: {e}")
            print("🔄 Criando indicadores manualmente...")
            self._create_manual_indicators()
        
        # Features adicionais sempre manuais
        self._create_additional_features()
        
        print("✅ Todos os indicadores técnicos criados!")
    
    def _create_manual_indicators(self):
        """
        Cria indicadores técnicos manualmente (fallback)
        """
        # Médias móveis simples
        self.data['SMA_5'] = self.data['Close'].rolling(window=5).mean()
        self.data['SMA_10'] = self.data['Close'].rolling(window=10).mean()
        self.data['SMA_20'] = self.data['Close'].rolling(window=20).mean()
        self.data['EMA_12'] = self.data['Close'].ewm(span=12).mean()
        self.data['EMA_26'] = self.data['Close'].ewm(span=26).mean()
        
        # RSI manual
        delta = self.data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.data['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD manual
        self.data['MACD'] = self.data['EMA_12'] - self.data['EMA_26']
        self.data['MACD_signal'] = self.data['MACD'].ewm(span=9).mean()
        self.data['MACD_histogram'] = self.data['MACD'] - self.data['MACD_signal']
        
        # Bollinger Bands manual
        sma_20 = self.data['Close'].rolling(20).mean()
        std_20 = self.data['Close'].rolling(20).std()
        self.data['BB_upper'] = sma_20 + (std_20 * 2)
        self.data['BB_lower'] = sma_20 - (std_20 * 2)
        self.data['BB_middle'] = sma_20
        
        # Estocástico manual simplificado
        high_14 = self.data['High'].rolling(14).max()
        low_14 = self.data['Low'].rolling(14).min()
        self.data['Stoch_K'] = 100 * ((self.data['Close'] - low_14) / (high_14 - low_14))
        self.data['Stoch_D'] = self.data['Stoch_K'].rolling(3).mean()
        
        # Williams %R manual
        self.data['Williams_R'] = -100 * ((high_14 - self.data['Close']) / (high_14 - low_14))
        
        # Volume SMA
        self.data['Volume_SMA'] = self.data['Volume'].rolling(20).mean()
        
        # ATR manual
        tr1 = self.data['High'] - self.data['Low']
        tr2 = abs(self.data['High'] - self.data['Close'].shift())
        tr3 = abs(self.data['Low'] - self.data['Close'].shift())
        true_range = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        self.data['ATR'] = true_range.rolling(14).mean()
    
    def _create_additional_features(self):
        """
        Cria features adicionais baseadas em preço e volume
        """
        # Features baseadas em preço (múltiplos períodos)
        for period in [1, 2, 3, 5]:
            self.data[f'Price_Change_{period}d'] = self.data['Close'].pct_change(period)
            
        # Ratios
        self.data['High_Low_Ratio'] = self.data['High'] / self.data['Low']
        self.data['Close_SMA20_Ratio'] = self.data['Close'] / self.data['SMA_20']
        self.data['Close_High_Ratio'] = self.data['Close'] / self.data['High']
        
        # Features de volume
        self.data['Volume_Change'] = self.data['Volume'].pct_change()
        self.data['Price_Volume'] = self.data['Close'] * self.data['Volume']
        self.data['Volume_Ratio'] = self.data['Volume'] / self.data['Volume_SMA']
        
        # Volatilidade
        self.data['Volatility_5'] = self.data['Price_Change_1d'].rolling(5).std()
        self.data['Volatility_10'] = self.data['Price_Change_1d'].rolling(10).std()
        
        # Bollinger position
        bb_range = self.data['BB_upper'] - self.data['BB_lower']
        self.data['BB_position'] = (self.data['Close'] - self.data['BB_lower']) / bb_range
        
        # Momentum
        self.data['Momentum_5'] = self.data['Close'] / self.data['Close'].shift(5)
        self.data['Momentum_10'] = self.data['Close'] / self.data['Close'].shift(10)
        
        # Features lag (valores anteriores)
        for lag in [1, 2]:
            self.data[f'RSI_lag_{lag}'] = self.data['RSI'].shift(lag)
            self.data[f'MACD_lag_{lag}'] = self.data['MACD'].shift(lag)
        
    def create_target_variable(self):
        """
        Cria a variável target (tendência: 1 para alta, 0 para baixa)
        """
        # Calcular o retorno do próximo dia
        self.data['Next_Return'] = self.data['Close'].shift(-1) / self.data['Close'] - 1
        
        # Criar target: 1 se próximo retorno > 0 (tendência de alta), 0 caso contrário
        self.data['Target'] = (self.data['Next_Return'] > 0).astype(int)
        
        print(f"Distribuição do target:")
        print(self.data['Target'].value_counts())
        print(f"Proporção de tendências de alta: {self.data['Target'].mean():.2%}")
        
    def prepare_features(self):
        """
        Prepara as features para o modelo com seleção das melhores
        """
        # Lista expandida de features disponíveis
        all_features = [
            'SMA_5', 'SMA_10', 'SMA_20', 'EMA_12', 'EMA_26',
            'RSI', 'MACD', 'MACD_signal', 'MACD_histogram',
            'BB_upper', 'BB_lower', 'BB_middle', 'BB_position',
            'Stoch_K', 'Stoch_D', 'Williams_R',
            'Volume_SMA', 'ATR', 'Volume_Change', 'Price_Volume', 'Volume_Ratio',
            'Price_Change_1d', 'Price_Change_2d', 'Price_Change_3d', 'Price_Change_5d',
            'High_Low_Ratio', 'Close_SMA20_Ratio', 'Close_High_Ratio',
            'Volatility_5', 'Volatility_10',
            'Momentum_5', 'Momentum_10',
            'RSI_lag_1', 'RSI_lag_2', 'MACD_lag_1', 'MACD_lag_2'
        ]
        
        # Remover valores NaN primeiro
        self.data = self.data.dropna()
        
        # Verificar quais features existem nos dados
        available_features = [f for f in all_features if f in self.data.columns]
        
        if len(available_features) > 10:
            # Calcular correlação com target para selecionar melhores features
            correlations = self.data[available_features + ['Target']].corr()['Target'].abs().sort_values(ascending=False)
            
            # Selecionar top features (excluindo o próprio target)
            top_features = correlations.head(21)[1:].index.tolist()  # Top 20 features
            self.features = top_features
            
            print(f"📊 Features selecionadas por correlação:")
            for i, feat in enumerate(self.features[:10]):
                corr = correlations[feat]
                print(f"  {i+1:2d}. {feat:<20} (corr: {corr:.3f})")
        else:
            # Se poucas features, usar todas disponíveis
            self.features = available_features
        
        print(f"\\nDataset final: {len(self.data)} registros")
        print(f"Features disponíveis: {len(available_features)}")
        print(f"Features utilizadas: {len(self.features)}")
        
        return self.features
        
    def split_data(self):
        """
        Divide os dados em treino e teste (últimos 30 dias para teste)
        """
        # Separar últimos 30 dias para teste
        test_size = 30
        train_data = self.data.iloc[:-test_size]
        test_data = self.data.iloc[-test_size:]
        
        X_train = train_data[self.features]
        y_train = train_data['Target']
        X_test = test_data[self.features]
        y_test = test_data['Target']
        
        print(f"Dados de treino: {len(X_train)} registros")
        print(f"Dados de teste: {len(X_test)} registros (últimos 30 dias)")
        
        return X_train, X_test, y_train, y_test
        
    def train_models(self, X_train, y_train):
        """
        Treina modelo de Regressão Logística com otimização
        """
        print("Treinando modelo de Regressão Logística...")
        
        # Testar diferentes configurações
        configs = [
            {'C': 0.1, 'solver': 'liblinear'},
            {'C': 1.0, 'solver': 'liblinear'},
            {'C': 10.0, 'solver': 'liblinear'},
            {'C': 0.1, 'solver': 'lbfgs'},
            {'C': 1.0, 'solver': 'lbfgs'},
            {'C': 10.0, 'solver': 'lbfgs'},
        ]
        
        best_score = 0
        best_model = None
        
        print("🔍 Testando configurações...")
        for i, config in enumerate(configs):
            model = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', LogisticRegression(random_state=42, max_iter=2000, **config))
            ])
            
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            mean_score = cv_scores.mean()
            
            print(f"  Config {i+1}: {mean_score:.4f} - {config}")
            
            if mean_score > best_score:
                best_score = mean_score
                best_model = model
        
        print(f"\n🏆 Melhor CV Score: {best_score:.4f}")
        
        # Treinar o melhor modelo
        self.model = best_model
        self.model.fit(X_train, y_train)
        
        print("✅ Modelo treinado com sucesso!")
        
        return self.model
    
    def evaluate_model(self, X_test, y_test):
        """
        Avalia o modelo no conjunto de teste
        """
        if self.model is None:
            raise ValueError("Modelo não treinado. Execute train_models() primeiro.")
            
        # Fazer previsões
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)
        
        # Calcular métricas
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n=== RESULTADOS DO MODELO ===")
        print(f"Acurácia no conjunto de teste: {accuracy:.4f} ({accuracy:.2%})")
        
        if accuracy >= 0.75:
            print("✅ Meta de 75% de acurácia ATINGIDA!")
        else:
            print("❌ Meta de 75% de acurácia NÃO atingida.")
        
        print(f"\nRelatório de classificação:")
        print(classification_report(y_test, y_pred, target_names=['Baixa ↓', 'Alta ↑']))
        
        # Matriz de confusão
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Baixa ↓', 'Alta ↑'], 
                    yticklabels=['Baixa ↓', 'Alta ↑'])
        plt.title('Matriz de Confusão')
        plt.ylabel('Valor Real')
        plt.xlabel('Predição')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return accuracy, y_pred, y_pred_proba
    
    def plot_feature_importance(self):
        """
        Plota a importância das features (coeficientes da regressão logística)
        """
        # Para regressão logística, usamos os coeficientes como importância
        coefficients = abs(self.model.named_steps['classifier'].coef_[0])
        feature_importance = pd.DataFrame({
            'feature': self.features,
            'importance': coefficients
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(data=feature_importance.head(15), x='importance', y='feature')
        plt.title('Top 15 Features Mais Importantes (Coeficientes)')
        plt.xlabel('Importância (|Coeficientes|)')
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_predictions(self, X_test, y_test, y_pred):
        """
        Plota as previsões vs valores reais
        """
        test_dates = self.data.index[-len(X_test):]
        
        plt.figure(figsize=(12, 8))
        
        # Subplot 1: Preço e previsões
        plt.subplot(2, 1, 1)
        prices = self.data['Close'].iloc[-len(X_test):]
        plt.plot(test_dates, prices, label='Preço de Fechamento', linewidth=2)
        
        # Marcar previsões corretas e incorretas
        correct_pred = (y_test == y_pred)
        incorrect_pred = ~correct_pred
        
        if sum(correct_pred) > 0:
            plt.scatter(test_dates[correct_pred], prices[correct_pred], 
                       color='green', s=50, alpha=0.7, label='Previsão Correta')
        
        if sum(incorrect_pred) > 0:
            plt.scatter(test_dates[incorrect_pred], prices[incorrect_pred], 
                       color='red', s=50, alpha=0.7, label='Previsão Incorreta')
        
        plt.title('Previsões do Modelo - Últimos 30 Dias')
        plt.ylabel('Preço (R$)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: Tendências
        plt.subplot(2, 1, 2)
        x_pos = range(len(test_dates))
        
        plt.bar([i for i, real in enumerate(y_test) if real == 1], 
                [1 for real in y_test if real == 1], 
                alpha=0.5, color='green', label='Real: Alta ↑', width=0.4)
        
        plt.bar([i for i, real in enumerate(y_test) if real == 0], 
                [1 for real in y_test if real == 0], 
                alpha=0.5, color='red', label='Real: Baixa ↓', width=0.4)
        
        plt.bar([i+0.4 for i, pred in enumerate(y_pred) if pred == 1], 
                [0.8 for pred in y_pred if pred == 1], 
                alpha=0.7, color='lightgreen', label='Pred: Alta ↑', width=0.4)
        
        plt.bar([i+0.4 for i, pred in enumerate(y_pred) if pred == 0], 
                [0.8 for pred in y_pred if pred == 0], 
                alpha=0.7, color='lightcoral', label='Pred: Baixa ↓', width=0.4)
        
        plt.xlabel('Dias de Teste')
        plt.ylabel('Tendência')
        plt.title('Comparação: Tendências Reais vs Previsões')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('predictions_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_complete_analysis(self):
        """
        Executa análise completa do modelo
        """
        print("=== INICIANDO ANÁLISE COMPLETA ===\n")
        
        # 1. Carregar dados
        self.load_data()
        
        # 2. Criar indicadores técnicos
        self.create_technical_indicators()
        
        # 3. Criar variável target
        self.create_target_variable()
        
        # 4. Preparar features
        self.prepare_features()
        
        # 5. Dividir dados
        X_train, X_test, y_train, y_test = self.split_data()
        
        # 6. Treinar modelos
        model = self.train_models(X_train, y_train)
        
        # 7. Avaliar modelo
        accuracy, y_pred, y_pred_proba = self.evaluate_model(X_test, y_test)
        
        # 8. Plotar resultados
        self.plot_feature_importance()
        self.plot_predictions(X_test, y_test, y_pred)
        
        print(f"\n=== ANÁLISE CONCLUÍDA ===")
        print(f"Acurácia final: {accuracy:.2%}")
        
        return accuracy, model

# Exemplo de uso
if __name__ == "__main__":
    # Criar preditor
    predictor = TrendPredictor(symbol='PETR4.SA', period='2y')
    
    # Executar análise completa
    accuracy, model = predictor.run_complete_analysis()
