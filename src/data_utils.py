"""
Utilitários para carregamento e preparação de dados do IBOVESPA
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def carregar_dados_ibovespa(anos=10):
    """
    Carrega dados históricos do IBOVESPA via Yahoo Finance
    
    Args:
        anos (int): Número de anos de dados históricos
        
    Returns:
        pd.DataFrame: Dados históricos do IBOVESPA
    """
    print("📊 Carregando dados do IBOVESPA...")
    
    # Definir período
    end_date = datetime.now()
    start_date = end_date - timedelta(days=anos * 365)
    
    # Baixar dados
    data = yf.download('^BVSP', start=start_date, end=end_date)
    
    # Limpar colunas se necessário
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    print(f"✅ {len(data)} dias carregados ({anos} anos)")
    
    return data

def preparar_dataset_temporal(data, features_lista, n_test_days=30):
    """
    Prepara dataset com divisão temporal para validação
    
    Args:
        data (pd.DataFrame): Dados com features e targets
        features_lista (list): Lista de features a usar
        n_test_days (int): Número de dias para teste
        
    Returns:
        tuple: X_treino, X_teste, y_treino, y_teste, y_treino_dir, y_teste_dir
    """
    print("🔧 Preparando dataset temporal...")
    
    # Criar dataset final
    dataset = data[features_lista + ['Target_Return', 'Target_Direction']].copy()
    dataset = dataset.dropna()
    
    # Divisão temporal
    split_index = len(dataset) - n_test_days
    
    # Separar dados
    data_treino = dataset.iloc[:split_index]
    data_teste = dataset.iloc[split_index:]
    
    # Separar features e targets
    X_treino = data_treino[features_lista]
    y_treino = data_treino['Target_Return']
    X_teste = data_teste[features_lista]
    y_teste = data_teste['Target_Return']
    
    # Direções para avaliação
    y_treino_dir = data_treino['Target_Direction']
    y_teste_dir = data_teste['Target_Direction']
    
    print(f"✅ Dataset preparado:")
    print(f"   Treino: {len(X_treino)} observações")
    print(f"   Teste: {len(X_teste)} observações")
    print(f"   Features: {len(features_lista)}")
    
    return X_treino, X_teste, y_treino, y_teste, y_treino_dir, y_teste_dir
