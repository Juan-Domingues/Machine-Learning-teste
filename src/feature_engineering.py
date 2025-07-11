"""
Engenharia de features para dados financeiros
"""

import pandas as pd
import numpy as np

def criar_features_basicas(data):
    """
    Cria features técnicas básicas
    
    Args:
        data (pd.DataFrame): Dados OHLCV
        
    Returns:
        tuple: (data_com_features, lista_features)
    """
    print("🔧 Criando features básicas...")
    
    # Calcular retornos
    data['Retorno'] = data['Close'].pct_change()
    
    # Médias móveis
    data['MM5'] = data['Close'].rolling(5).mean()
    data['MM20'] = data['Close'].rolling(20).mean()
    data['MM50'] = data['Close'].rolling(50).mean()
    
    # RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # Volatilidade
    data['Volatilidade'] = data['Retorno'].rolling(20).std()
    
    # Volume normalizado
    data['Volume_Norm'] = data['Volume'] / data['Volume'].rolling(20).mean()
    
    # Posição no canal
    data['Max_20'] = data['High'].rolling(20).max()
    data['Min_20'] = data['Low'].rolling(20).min()
    data['Channel_Position'] = (data['Close'] - data['Min_20']) / (data['Max_20'] - data['Min_20'])
    
    # Features de momentum
    data['Momentum_3'] = data['Close'] / data['Close'].shift(3) - 1
    data['Returns_3d'] = data['Close'].pct_change(3)
    data['Returns_7d'] = data['Close'].pct_change(7)
    
    # Sinais
    data['MM_Cross'] = (data['MM5'] > data['MM20']).astype(int)
    data['Price_Above_MA20'] = (data['Close'] > data['MM20']).astype(int)
    data['RSI_Signal'] = np.where(data['RSI'] < 30, 1, np.where(data['RSI'] > 70, -1, 0))
    data['Volume_Spike'] = (data['Volume'] > 1.5 * data['Volume'].rolling(20).mean()).astype(int)
    data['Low_Vol'] = (data['Volatilidade'] < data['Volatilidade'].rolling(50).quantile(0.3)).astype(int)
    
    features_basicas = [
        'MM5', 'MM20', 'RSI', 'Volatilidade', 'MM_Cross', 'Price_Above_MA20',
        'RSI_Signal', 'Returns_3d', 'Returns_7d', 'Volume_Spike', 'Low_Vol', 
        'Channel_Position', 'Volume_Norm', 'Momentum_3'
    ]
    
    print(f"✅ {len(features_basicas)} features básicas criadas")
    
    return data, features_basicas

def criar_features_avancadas(data, features_basicas):
    """
    Cria features avançadas baseadas nas básicas
    
    Args:
        data (pd.DataFrame): Dados com features básicas
        features_basicas (list): Lista de features básicas
        
    Returns:
        tuple: (data_com_features_avancadas, lista_completa_features)
    """
    print("🚀 Criando features avançadas...")
    
    # Features de interação
    data['Volume_Price_Momentum'] = data['Volume_Norm'] * data['Retorno']
    data['Momentum_Ratio'] = data['MM5'] / (data['MM20'] + 0.001)
    data['Price_Momentum'] = data['Close'] / (data['MM50'] + 0.001)
    
    # Features de volatilidade
    data['Volatilidade_Relativa'] = data['Volatilidade'] / (data['Volatilidade'].rolling(50).mean() + 0.001)
    data['Volatilidade_Tendencia'] = data['Volatilidade'].rolling(5).mean() / (data['Volatilidade'].rolling(20).mean() + 0.001)
    
    # Features de RSI avançadas
    data['RSI_Momentum'] = data['RSI'].diff()
    data['RSI_Normalized'] = (data['RSI'] - 50) / 50
    data['RSI_Volume'] = data['RSI_Normalized'] * data['Volume_Norm']
    
    # Features de canal e posição
    data['Canal_Momentum'] = data['Channel_Position'].diff()
    data['MM_Cross_Strength'] = (data['MM5'] - data['MM20']) / (data['MM20'] + 0.001)
    
    # Features de regime
    data['Bull_Market'] = (data['Close'] > data['MM50']).astype(int)
    data['High_Vol_Regime'] = (data['Volatilidade'] > data['Volatilidade'].rolling(50).quantile(0.7)).astype(int)
    data['Consolidation'] = ((data['Max_20'] - data['Min_20']) / data['Close'] < 0.05).astype(int)
    
    # Features lagged
    data['Retorno_Lag1'] = data['Retorno'].shift(1)
    data['RSI_Lag1'] = data['RSI'].shift(1)
    
    # Features de tendência
    data['Tendencia_5d'] = (data['Close'] > data['Close'].shift(5)).astype(int)
    data['Aceleracao'] = data['Retorno'] - data['Retorno'].shift(1)
    
    features_avancadas = [
        'Volume_Price_Momentum', 'Momentum_Ratio', 'Price_Momentum',
        'Volatilidade_Relativa', 'Volatilidade_Tendencia',
        'RSI_Momentum', 'RSI_Normalized', 'RSI_Volume',
        'Canal_Momentum', 'MM_Cross_Strength',
        'Bull_Market', 'High_Vol_Regime', 'Consolidation',
        'Retorno_Lag1', 'RSI_Lag1', 'Tendencia_5d', 'Aceleracao'
    ]
    
    features_completas = features_basicas + features_avancadas
    
    print(f"✅ {len(features_avancadas)} features avançadas criadas")
    print(f"📊 Total: {len(features_completas)} features")
    
    return data, features_completas

def criar_targets(data):
    """
    Cria targets para regressão e classificação
    
    Args:
        data (pd.DataFrame): Dados base
        
    Returns:
        pd.DataFrame: Dados com targets criados
    """
    print("🎯 Criando targets...")
    
    # Target de regressão: retorno do próximo dia
    data['Target_Return'] = data['Close'].pct_change().shift(-1)
    
    # Target de classificação: direção do próximo dia
    data['Target_Direction'] = (data['Target_Return'] > 0).astype(int)
    
    # Estatísticas
    returns_stats = data['Target_Return'].describe()
    direction_counts = data['Target_Direction'].value_counts()
    total = len(data['Target_Direction'].dropna())
    
    print(f"✅ Targets criados:")
    print(f"   Retorno médio: {returns_stats['mean']:.4f}")
    print(f"   Desvio padrão: {returns_stats['std']:.4f}")
    print(f"   Dias de alta: {direction_counts.get(1, 0)} ({direction_counts.get(1, 0)/total:.1%})")
    print(f"   Dias de baixa: {direction_counts.get(0, 0)} ({direction_counts.get(0, 0)/total:.1%})")
    
    return data
