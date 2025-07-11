"""
Teste simples do pipeline otimizado
"""
import sys
sys.path.append('.')

try:
    print("🧪 Iniciando teste simples...")
    
    import pandas as pd
    import numpy as np
    import yfinance as yf
    print("✅ Imports básicos OK")
    
    from sklearn.model_selection import train_test_split, cross_val_score, KFold
    from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import mean_squared_error, r2_score, make_scorer, accuracy_score
    from sklearn.feature_selection import SelectKBest, f_regression
    from sklearn.ensemble import VotingRegressor
    from sklearn.model_selection import TimeSeriesSplit
    print("✅ Imports sklearn OK")
    
    # Teste básico
    data = yf.download('^BVSP', start='2023-01-01', end='2024-01-01')
    print(f"✅ Dados carregados: {len(data)} registros")
    
    print("🎉 Todos os componentes funcionando!")
    
except Exception as e:
    print(f"❌ Erro: {e}")
    import traceback
    traceback.print_exc()
