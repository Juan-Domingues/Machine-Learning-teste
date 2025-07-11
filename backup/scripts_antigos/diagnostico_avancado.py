"""
DIAGNÓSTICO AVANÇADO - Machine Learning IBOVESPA
Análise profunda das limitações e busca por soluções
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split, cross_val_score, KFold, TimeSeriesSplit
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif

def carregar_dados_completos():
    """Carrega dados com maior volume de informações"""
    print("🔍 DIAGNÓSTICO AVANÇADO - Carregando dados...")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=15 * 365)  # 15 anos
    
    # IBOVESPA
    ibov = yf.download('^BVSP', start=start_date, end=end_date)
    if isinstance(ibov.columns, pd.MultiIndex):
        ibov.columns = ibov.columns.get_level_values(0)
    
    # Dados adicionais para contexto
    sp500 = yf.download('^GSPC', start=start_date, end=end_date)['Close']
    dollar = yf.download('USDBRL=X', start=start_date, end=end_date)['Close']
    
    # Combinar
    data = ibov.copy()
    data['SP500'] = sp500
    data['USD_BRL'] = dollar
    
    print(f"✓ {len(data)} dias carregados (15 anos)")
    return data

def analise_distribuicao_retornos(data):
    """Análise estatística detalhada dos retornos"""
    print("\n📊 ANÁLISE 1: Distribuição dos Retornos")
    
    data['Retorno'] = data['Close'].pct_change()
    retornos = data['Retorno'].dropna()
    
    # Estatísticas básicas
    print(f"Média: {retornos.mean():.6f}")
    print(f"Desvio: {retornos.std():.6f}")
    print(f"Skewness: {stats.skew(retornos):.3f}")
    print(f"Kurtosis: {stats.kurtosis(retornos):.3f}")
    print(f"Teste Jarque-Bera: {stats.jarque_bera(retornos)}")
    
    # Quartis
    q25, q50, q75 = retornos.quantile([0.25, 0.5, 0.75])
    print(f"Q25: {q25:.3%}, Mediana: {q50:.3%}, Q75: {q75:.3%}")
    
    # Movimento significativo
    threshold_levels = [0.001, 0.002, 0.005, 0.01, 0.02]
    print("\nMovimentos significativos:")
    for th in threshold_levels:
        pct = (abs(retornos) > th).mean()
        print(f"  >{th:.1%}: {pct:.1%} dos dias")
    
    return retornos

def analise_previsibilidade_temporal(data):
    """Análise de previsibilidade em diferentes horizontes"""
    print("\n⏰ ANÁLISE 2: Previsibilidade Temporal")
    
    retornos = data['Retorno'].dropna()
    
    # Autocorrelação em diferentes lags
    lags = [1, 2, 3, 5, 10, 20]
    print("Autocorrelação:")
    for lag in lags:
        corr = retornos.corr(retornos.shift(lag))
        print(f"  Lag {lag}: {corr:.4f}")
    
    # Teste de reversão à média
    print("\nTeste de reversão à média:")
    for threshold in [0.01, 0.02, 0.03]:
        big_moves = abs(retornos) > threshold
        next_day_opposite = ((retornos > 0) & (retornos.shift(-1) < 0)) | \
                           ((retornos < 0) & (retornos.shift(-1) > 0))
        
        reversal_rate = next_day_opposite[big_moves].mean()
        print(f"  Após movimento >{threshold:.1%}: {reversal_rate:.1%} reversões")

def criar_features_experimentais(data):
    """Cria features experimentais mais sofisticadas"""
    print("\n🔬 ANÁLISE 3: Features Experimentais")
    
    # Básicas
    data['Retorno'] = data['Close'].pct_change()
    data['Retorno_Abs'] = abs(data['Retorno'])
    
    # Contexto macroeconômico
    if 'SP500' in data.columns:
        data['SP500_Ret'] = data['SP500'].pct_change()
        data['Correlacao_SP500'] = data['Retorno'].rolling(20).corr(data['SP500_Ret'])
    
    if 'USD_BRL' in data.columns:
        data['USD_Ret'] = data['USD_BRL'].pct_change()
        data['Correlacao_USD'] = data['Retorno'].rolling(20).corr(data['USD_Ret'])
    
    # Volatilidade multi-escala
    for window in [5, 10, 20, 60]:
        data[f'Vol_{window}d'] = data['Retorno'].rolling(window).std()
        data[f'Vol_Rank_{window}d'] = data[f'Vol_{window}d'].rolling(252).rank(pct=True)
    
    # Momentum multi-escala
    for window in [5, 10, 20, 60]:
        data[f'Momentum_{window}d'] = data['Close'].pct_change(window)
        data[f'Mom_Rank_{window}d'] = data[f'Momentum_{window}d'].rolling(252).rank(pct=True)
    
    # Padrões de retorno
    data['Retorno_Lag1'] = data['Retorno'].shift(1)
    data['Retorno_Lag2'] = data['Retorno'].shift(2)
    data['Retorno_Lag3'] = data['Retorno'].shift(3)
    
    # Features de regime
    data['High_Vol_Regime'] = (data['Vol_20d'] > data['Vol_20d'].rolling(60).quantile(0.75)).astype(int)
    data['Trend_Regime'] = (data['Close'] > data['Close'].rolling(50).mean()).astype(int)
    
    # Features de sentimento
    data['RSI'] = compute_rsi(data['Close'], 14)
    data['RSI_Regime'] = pd.cut(data['RSI'], bins=[0, 30, 70, 100], labels=[0, 1, 2])
    
    # Volume features (se disponível)
    if 'Volume' in data.columns:
        data['Volume_MA'] = data['Volume'].rolling(20).mean()
        data['Volume_Ratio'] = data['Volume'] / data['Volume_MA']
        data['High_Volume'] = (data['Volume_Ratio'] > 1.5).astype(int)
    
    # Selecionar features válidas
    feature_cols = [col for col in data.columns if col not in 
                   ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'SP500', 'USD_BRL']]
    
    print(f"✓ {len(feature_cols)} features experimentais criadas")
    return data, feature_cols

def compute_rsi(prices, window=14):
    """Calcula RSI"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def teste_multiplos_targets(data):
    """Testa diferentes definições de target"""
    print("\n🎯 ANÁLISE 4: Múltiplos Targets")
    
    retornos = data['Retorno']
    
    targets = {}
    
    # Diferentes thresholds
    for threshold in [0.001, 0.002, 0.005, 0.01]:
        targets[f'Target_{threshold*100:.1f}pct'] = (retornos.shift(-1) > threshold).astype(int)
    
    # Target de magnitude
    targets['Target_BigMove'] = (abs(retornos.shift(-1)) > 0.01).astype(int)
    
    # Target de ranking
    targets['Target_Top25'] = (retornos.shift(-1) > retornos.shift(-1).rolling(60).quantile(0.75)).astype(int)
    
    # Target de regime
    vol_target = data['Vol_20d'].shift(-1) > data['Vol_20d'].rolling(60).quantile(0.75)
    targets['Target_HighVol'] = vol_target.astype(int)
    
    # Estatísticas dos targets
    for name, target in targets.items():
        valid = target.dropna()
        balance = valid.mean()
        print(f"  {name}: {balance:.1%} positivos")
    
    return targets

def teste_modelos_avancados(X, y_dict):
    """Testa modelos mais sofisticados"""
    print("\n🤖 ANÁLISE 5: Modelos Avançados")
    
    modelos = {
        'Logistic': LogisticRegression(random_state=42, max_iter=1000),
        'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
        'SVM_RBF': SVC(kernel='rbf', random_state=42, probability=True)
    }
    
    # Time Series Split para dados temporais
    tscv = TimeSeriesSplit(n_splits=5)
    
    resultados = {}
    
    for target_name, y in y_dict.items():
        print(f"\nTarget: {target_name}")
        resultados[target_name] = {}
        
        # Alinhar índices corretamente
        common_idx = X.index.intersection(y.index)
        X_aligned = X.loc[common_idx]
        y_aligned = y.loc[common_idx]
        
        # Dados válidos
        valid_idx = ~(X_aligned.isnull().any(axis=1) | y_aligned.isnull())
        X_valid = X_aligned[valid_idx]
        y_valid = y_aligned[valid_idx]
        
        if len(X_valid) < 100:
            print("  Dados insuficientes")
            continue
        
        for nome, modelo in modelos.items():
            try:
                # Pipeline com normalização
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('selector', SelectKBest(f_classif, k=min(10, X_valid.shape[1]))),
                    ('classifier', modelo)
                ])
                
                # Cross validation temporal
                scores = cross_val_score(pipeline, X_valid, y_valid, cv=tscv, scoring='accuracy')
                
                resultados[target_name][nome] = {
                    'mean': scores.mean(),
                    'std': scores.std()
                }
                
                # Baseline
                baseline = max(y_valid.mean(), 1 - y_valid.mean())
                improvement = scores.mean() - baseline
                
                print(f"  {nome}: {scores.mean():.1%} (±{scores.std():.1%}) | +{improvement*100:+.1f}pts")
                
            except Exception as e:
                print(f"  {nome}: ERRO - {e}")
    
    return resultados

def encontrar_melhor_setup(resultados):
    """Encontra a melhor combinação target + modelo"""
    print("\n🏆 MELHOR SETUP ENCONTRADO")
    
    melhor_score = 0
    melhor_setup = None
    
    for target_name, modelos in resultados.items():
        for modelo_name, metrics in modelos.items():
            score = metrics['mean']
            if score > melhor_score:
                melhor_score = score
                melhor_setup = (target_name, modelo_name, metrics)
    
    if melhor_setup:
        target, modelo, metrics = melhor_setup
        print(f"Target: {target}")
        print(f"Modelo: {modelo}")
        print(f"Acurácia: {metrics['mean']:.1%} (±{metrics['std']:.1%})")
    else:
        print("Nenhum setup válido encontrado")
    
    return melhor_setup

def main_diagnostico():
    """Pipeline completo de diagnóstico avançado"""
    print("="*70)
    print("DIAGNÓSTICO AVANÇADO - MACHINE LEARNING IBOVESPA")
    print("Análise profunda para encontrar sinais de previsibilidade")
    print("="*70)
    
    try:
        # 1. Carregar dados
        data = carregar_dados_completos()
        
        # 2. Análise de distribuição
        retornos = analise_distribuicao_retornos(data)
        
        # 3. Análise temporal
        analise_previsibilidade_temporal(data)
        
        # 4. Features experimentais
        data, features = criar_features_experimentais(data)
        
        # 5. Múltiplos targets
        targets = teste_multiplos_targets(data)
        
        # 6. Preparar dataset
        X = data[features].dropna()
        
        # 7. Teste modelos avançados
        resultados = teste_modelos_avancados(X, targets)
        
        # 8. Melhor setup
        melhor = encontrar_melhor_setup(resultados)
        
        print("\n" + "="*70)
        print("CONCLUSÕES DO DIAGNÓSTICO AVANÇADO")
        print("="*70)
        
        if melhor and melhor[2]['mean'] > 0.55:
            print("🎉 ENCONTRADO SINAL DE PREVISIBILIDADE!")
            print(f"A combinação {melhor[0]} + {melhor[1]} mostra potencial.")
        else:
            print("📊 MERCADO ALTAMENTE EFICIENTE")
            print("Os dados confirmam que o IBOVESPA é difícil de prever.")
            print("Possíveis razões:")
            print("- Eficiência do mercado brasileiro")
            print("- Alta volatilidade e ruído")
            print("- Influências externas (commodities, política)")
            print("- Necessidade de dados alternativos (sentimento, fluxo)")
        
        return resultados
        
    except Exception as e:
        print(f"Erro no diagnóstico: {e}")
        return None

if __name__ == "__main__":
    resultados = main_diagnostico()
