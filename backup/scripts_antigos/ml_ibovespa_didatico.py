"""
Machine Learning Aplicado ao IBOVESPA - Versão Didática Simples
Com Validação Cruzada para demonstrar os conceitos fundamentais do curso
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Imports do Scikit-Learn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def main():
    print("📚 MACHINE LEARNING - LINEAR REGRESSION NO IBOVESPA")
    print("🔬 VERSÃO DIDÁTICA COM VALIDAÇÃO CRUZADA")
    print("🎯 Métricas: R², RMSE, MAE + ACURÁCIA DE DIREÇÃO")
    print("=" * 60)
    
    # 1. CARREGAMENTO DE DADOS
    print("1. CARREGANDO DADOS DO IBOVESPA...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=10*365)  # 10 anos
    
    ibov = yf.download('^BVSP', start=start_date, end=end_date)
    print(f"✅ {len(ibov)} dias de dados carregados")
    
    # 2. CRIAÇÃO DE FEATURES BÁSICAS
    print("\n2. CRIANDO FEATURES...")
    data = pd.DataFrame()
    data['Close'] = ibov['Close']
    data['Volume'] = ibov['Volume']
    data['High'] = ibov['High']
    data['Low'] = ibov['Low']
    
    # Features técnicas básicas
    data['MM5'] = data['Close'].rolling(5).mean()
    data['MM20'] = data['Close'].rolling(20).mean()
    data['Volatilidade'] = data['Close'].pct_change().rolling(20).std()
    data['RSI'] = calculate_rsi(data['Close'])
    data['Volume_Norm'] = data['Volume'] / data['Volume'].rolling(20).mean()
    
    # 3. CRIAÇÃO DO TARGET
    print("3. CRIANDO TARGET...")
    data['Retorno_Futuro'] = data['Close'].pct_change().shift(-1)
    data['Direcao_Futura'] = (data['Retorno_Futuro'] > 0).astype(int)
    
    # 4. PREPARAÇÃO DOS DADOS
    print("4. PREPARANDO DADOS...")
    features = ['MM5', 'MM20', 'Volatilidade', 'RSI', 'Volume_Norm']
    data_clean = data[features + ['Retorno_Futuro', 'Direcao_Futura']].dropna()
    
    X = data_clean[features]
    y_reg = data_clean['Retorno_Futuro']  # Para regressão
    y_class = data_clean['Direcao_Futura']  # Para acurácia de direção
    
    print(f"✅ {len(X)} observações preparadas")
    print(f"✅ {len(features)} features utilizadas")
    
    # 5. DIVISÃO TREINO/TESTE (70/30)
    print("\n5. DIVIDINDO DADOS (70% TREINO / 30% TESTE)...")
    X_train, X_test, y_reg_train, y_reg_test, y_class_train, y_class_test = train_test_split(
        X, y_reg, y_class, test_size=0.3, random_state=42, shuffle=False
    )
    
    print(f"📊 Treino: {len(X_train)} observações")
    print(f"📊 Teste: {len(X_test)} observações")
    
    # 6. TREINAMENTO DO MODELO
    print("\n6. TREINAMENTO - LINEAR REGRESSION...")
    
    # Normalização
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Modelo
    model = LinearRegression()
    model.fit(X_train_scaled, y_reg_train)
    
    # 7. AVALIAÇÃO NO TESTE
    print("\n7. AVALIAÇÃO NO CONJUNTO DE TESTE...")
    y_pred = model.predict(X_test_scaled)
    
    # Métricas de regressão
    r2 = r2_score(y_reg_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_reg_test, y_pred))
    mae = mean_absolute_error(y_reg_test, y_pred)
    
    # Acurácia de direção
    direcao_pred = (y_pred > 0).astype(int)
    acuracia = np.mean(direcao_pred == y_class_test)
    
    print(f"📊 R²: {r2:.4f}")
    print(f"📊 RMSE: {rmse:.4f}")
    print(f"📊 MAE: {mae:.4f}")
    print(f"🎯 Acurácia de Direção: {acuracia:.1%}")
    
    # 8. VALIDAÇÃO CRUZADA
    print("\n8. VALIDAÇÃO CRUZADA (5-FOLD)...")
    
    # Para R²
    cv_scores_r2 = cross_val_score(model, X_train_scaled, y_reg_train, cv=5, scoring='r2')
    print(f"📊 R² CV: {cv_scores_r2.mean():.4f} (±{cv_scores_r2.std():.4f})")
    
    # Para acurácia de direção (manual)
    from sklearn.model_selection import KFold
    kfold = KFold(n_splits=5, shuffle=False)
    cv_accuracies = []
    
    for train_idx, val_idx in kfold.split(X_train_scaled):
        X_cv_train, X_cv_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
        y_cv_train, y_cv_val = y_reg_train.iloc[train_idx], y_reg_train.iloc[val_idx]
        y_class_cv_val = y_class_train.iloc[val_idx]
        
        model_cv = LinearRegression()
        model_cv.fit(X_cv_train, y_cv_train)
        y_cv_pred = model_cv.predict(X_cv_val)
        
        direcao_cv_pred = (y_cv_pred > 0).astype(int)
        acc_cv = np.mean(direcao_cv_pred == y_class_cv_val)
        cv_accuracies.append(acc_cv)
    
    cv_acc_mean = np.mean(cv_accuracies)
    cv_acc_std = np.std(cv_accuracies)
    
    print(f"🎯 Acurácia CV: {cv_acc_mean:.1%} (±{cv_acc_std:.1%})")
    
    # 9. RESUMO FINAL
    print("\n" + "="*50)
    print("🎯 RESUMO FINAL")
    print("="*50)
    print(f"📊 Dataset: {len(data_clean)} observações")
    print(f"📊 Features: {len(features)}")
    print(f"📊 Período: 10 anos")
    print(f"📊 Divisão: 70% treino / 30% teste")
    print("\n🧪 RESULTADOS HOLDOUT:")
    print(f"📊 R²: {r2:.4f}")
    print(f"🎯 Acurácia: {acuracia:.1%}")
    print("\n🔄 RESULTADOS VALIDAÇÃO CRUZADA:")
    print(f"📊 R² CV: {cv_scores_r2.mean():.4f} (±{cv_scores_r2.std():.4f})")
    print(f"🎯 Acurácia CV: {cv_acc_mean:.1%} (±{cv_acc_std:.1%})")
    print("\n✅ Análise concluída!")

def calculate_rsi(prices, window=14):
    """Calcula o RSI (Relative Strength Index)"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

if __name__ == "__main__":
    main()
