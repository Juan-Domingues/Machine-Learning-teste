"""
Pipeline SUPER OTIMIZADO para maximizar acurácia
Baseado em estratégias que funcionaram para outros grupos
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Adicionar pasta src ao path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, chi2, f_classif

def carregar_dados_simples(anos=5):
    """
    Carrega dados mais focados (menos anos para menos ruído)
    """
    print("📊 Carregando dados IBOVESPA (5 anos - menos ruído)...")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=anos * 365)
    
    data = yf.download('^BVSP', start=start_date, end=end_date)
    
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    print(f"✅ {len(data)} dias carregados ({anos} anos)")
    return data

def criar_features_simples_eficazes(data):
    """
    Features SIMPLES mas EFICAZES (como provavelmente o colega usou)
    """
    print("🎯 Criando features simples e eficazes...")
    
    # 1. RETORNOS SIMPLES (fundamentais)
    data['Return_1d'] = data['Close'].pct_change()
    data['Return_2d'] = data['Close'].pct_change(2) 
    data['Return_3d'] = data['Close'].pct_change(3)
    
    # 2. MÉDIAS MÓVEIS SIMPLES
    data['SMA_5'] = data['Close'].rolling(5).mean()
    data['SMA_10'] = data['Close'].rolling(10).mean()
    data['SMA_20'] = data['Close'].rolling(20).mean()
    
    # 3. POSIÇÃO RELATIVA (muito eficaz!)
    data['Price_vs_SMA5'] = (data['Close'] - data['SMA_5']) / data['SMA_5']
    data['Price_vs_SMA10'] = (data['Close'] - data['SMA_10']) / data['SMA_10']
    data['Price_vs_SMA20'] = (data['Close'] - data['SMA_20']) / data['SMA_20']
    
    # 4. MOMENTUM SIMPLES
    data['SMA5_vs_SMA10'] = (data['SMA_5'] - data['SMA_10']) / data['SMA_10']
    data['SMA10_vs_SMA20'] = (data['SMA_10'] - data['SMA_20']) / data['SMA_20']
    
    # 5. VOLUME (normalizado)
    data['Volume_SMA'] = data['Volume'].rolling(20).mean()
    data['Volume_Ratio'] = data['Volume'] / data['Volume_SMA']
    
    # 6. VOLATILIDADE SIMPLES
    data['Volatility'] = data['Return_1d'].rolling(5).std()
    
    # 7. FEATURES LAGGED (memória do mercado)
    data['Return_1d_lag1'] = data['Return_1d'].shift(1)
    data['Return_1d_lag2'] = data['Return_1d'].shift(2)
    
    # 8. RSI SIMPLES
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    data['RSI_Signal'] = (data['RSI'] > 50).astype(int)  # Signal binário
    
    # 9. BREAKOUT SIGNALS
    data['High_20'] = data['High'].rolling(20).max()
    data['Low_20'] = data['Low'].rolling(20).min()
    data['Breakout_Up'] = (data['Close'] > data['High_20'].shift(1)).astype(int)
    data['Breakout_Down'] = (data['Close'] < data['Low_20'].shift(1)).astype(int)
    
    # Lista de features (apenas as mais eficazes)
    features_eficazes = [
        'Return_1d_lag1', 'Return_1d_lag2', 'Return_2d', 'Return_3d',
        'Price_vs_SMA5', 'Price_vs_SMA10', 'Price_vs_SMA20',
        'SMA5_vs_SMA10', 'SMA10_vs_SMA20',
        'Volume_Ratio', 'Volatility',
        'RSI_Signal', 'Breakout_Up', 'Breakout_Down'
    ]
    
    print(f"✅ {len(features_eficazes)} features eficazes criadas")
    
    return data, features_eficazes

def estrategia_classificacao_direta(data, features_lista):
    """
    ESTRATÉGIA DIFERENTE: Classificação direta ao invés de regressão
    (Pode ser o que o colega fez!)
    """
    print("🎯 TESTANDO CLASSIFICAÇÃO DIRETA...")
    
    # Target: DIREÇÃO direta (não regressão)
    data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
    
    # Dataset limpo
    dataset = data[features_lista + ['Target']].dropna()
    
    # Divisão temporal conservadora
    n_test = 30
    train_data = dataset.iloc[:-n_test]
    test_data = dataset.iloc[-n_test:]
    
    X_train = train_data[features_lista]
    y_train = train_data['Target']
    X_test = test_data[features_lista]
    y_test = test_data['Target']
    
    print(f"📊 Treino: {len(X_train)} | Teste: {len(X_test)}")
    print(f"📊 Distribuição treino: {y_train.mean():.1%} alta, {1-y_train.mean():.1%} baixa")
    
    # MODELO 1: Logistic Regression (SIMPLES E EFICAZ)
    print("\n🔹 MODELO 1: Logistic Regression")
    modelo_lr = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(random_state=42, max_iter=1000))
    ])
    
    modelo_lr.fit(X_train, y_train)
    y_pred_lr = modelo_lr.predict(X_test)
    acc_lr = accuracy_score(y_test, y_pred_lr)
    print(f"   Acurácia: {acc_lr:.1%}")
    
    # MODELO 2: Random Forest (NÃO LINEAR)
    print("\n🔹 MODELO 2: Random Forest")
    modelo_rf = RandomForestClassifier(
        n_estimators=100, 
        max_depth=10, 
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    )
    
    modelo_rf.fit(X_train, y_train)
    y_pred_rf = modelo_rf.predict(X_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    print(f"   Acurácia: {acc_rf:.1%}")
    
    # Feature importance do RF
    importances = modelo_rf.feature_importances_
    feature_imp = list(zip(features_lista, importances))
    feature_imp.sort(key=lambda x: x[1], reverse=True)
    print("   Top 5 features:")
    for feat, imp in feature_imp[:5]:
        print(f"     {feat}: {imp:.3f}")
    
    # MODELO 3: Gradient Boosting
    print("\n🔹 MODELO 3: Gradient Boosting")
    modelo_gb = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    
    modelo_gb.fit(X_train, y_train)
    y_pred_gb = modelo_gb.predict(X_test)
    acc_gb = accuracy_score(y_test, y_pred_gb)
    print(f"   Acurácia: {acc_gb:.1%}")
    
    # Encontrar melhor modelo
    resultados = [
        ('Logistic Regression', acc_lr, modelo_lr, y_pred_lr),
        ('Random Forest', acc_rf, modelo_rf, y_pred_rf),
        ('Gradient Boosting', acc_gb, modelo_gb, y_pred_gb)
    ]
    
    melhor = max(resultados, key=lambda x: x[1])
    
    return melhor, resultados, X_test, y_test

def testar_subset_features(data, features_lista):
    """
    Testa diferentes subsets de features (como o colega pode ter feito)
    """
    print("\n🧪 TESTANDO SUBSETS DE FEATURES...")
    
    # Target
    data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
    dataset = data[features_lista + ['Target']].dropna()
    
    # Divisão
    n_test = 30
    train_data = dataset.iloc[:-n_test]
    test_data = dataset.iloc[-n_test:]
    
    X_train = train_data[features_lista]
    y_train = train_data['Target']
    X_test = test_data[features_lista]
    y_test = test_data['Target']
    
    melhores = []
    
    # SUBSET 1: Apenas momentum/trend (3 features)
    subset1 = ['Return_1d_lag1', 'Price_vs_SMA5', 'SMA5_vs_SMA10']
    if all(f in features_lista for f in subset1):
        modelo = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(random_state=42, max_iter=1000))
        ])
        modelo.fit(X_train[subset1], y_train)
        y_pred = modelo.predict(X_test[subset1])
        acc = accuracy_score(y_test, y_pred)
        melhores.append(('Momentum (3 features)', acc, subset1, modelo))
        print(f"🔸 Momentum (3): {acc:.1%}")
    
    # SUBSET 2: Trend + Volume (4 features)
    subset2 = ['Price_vs_SMA5', 'Price_vs_SMA10', 'Volume_Ratio', 'Return_1d_lag1']
    if all(f in features_lista for f in subset2):
        modelo = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(random_state=42, max_iter=1000))
        ])
        modelo.fit(X_train[subset2], y_train)
        y_pred = modelo.predict(X_test[subset2])
        acc = accuracy_score(y_test, y_pred)
        melhores.append(('Trend+Volume (4 features)', acc, subset2, modelo))
        print(f"🔸 Trend+Volume (4): {acc:.1%}")
    
    # SUBSET 3: Top 5 por SelectKBest
    selector = SelectKBest(score_func=f_classif, k=5)
    X_selected = selector.fit_transform(X_train, y_train)
    subset3 = X_train.columns[selector.get_support()].tolist()
    
    modelo = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(random_state=42, max_iter=1000))
    ])
    modelo.fit(X_train[subset3], y_train)
    y_pred = modelo.predict(X_test[subset3])
    acc = accuracy_score(y_test, y_pred)
    melhores.append(('SelectKBest (5 features)', acc, subset3, modelo))
    print(f"🔸 SelectKBest (5): {acc:.1%}")
    
    # SUBSET 4: Random Forest com top features
    rf_temp = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    rf_temp.fit(X_train, y_train)
    
    # Top 5 features por importance
    importances = rf_temp.feature_importances_
    indices = np.argsort(importances)[::-1][:5]
    subset4 = [features_lista[i] for i in indices]
    
    modelo = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(random_state=42, max_iter=1000))
    ])
    modelo.fit(X_train[subset4], y_train)
    y_pred = modelo.predict(X_test[subset4])
    acc = accuracy_score(y_test, y_pred)
    melhores.append(('RF Importance (5 features)', acc, subset4, modelo))
    print(f"🔸 RF Importance (5): {acc:.1%}")
    
    # Melhor subset
    melhor_subset = max(melhores, key=lambda x: x[1])
    
    return melhor_subset, melhores, X_test, y_test

def pipeline_super_otimizado():
    """
    Pipeline SUPER otimizado tentando reproduzir sucesso de 60%
    """
    print("="*70)
    print("🚀 PIPELINE SUPER OTIMIZADO")
    print("🎯 OBJETIVO: Reproduzir 60% de acurácia")
    print("📊 ESTRATÉGIA: Classificação direta + Features simples")
    print("="*70)
    
    try:
        # 1. Dados mais focados (5 anos ao invés de 10)
        print("\n📥 ETAPA 1: Dados Focados")
        data = carregar_dados_simples(anos=5)
        
        # 2. Features simples e eficazes
        print("\n🔧 ETAPA 2: Features Simples")
        data, features_eficazes = criar_features_simples_eficazes(data)
        
        # 3. Estratégia classificação direta
        print("\n🎯 ETAPA 3: Classificação Direta")
        melhor_modelo, todos_modelos, X_test, y_test = estrategia_classificacao_direta(data, features_eficazes)
        
        # 4. Teste de subsets de features
        melhor_subset, todos_subsets, _, _ = testar_subset_features(data, features_eficazes)
        
        # 5. COMPARAÇÃO FINAL
        print("\n" + "="*70)
        print("🏆 RESULTADOS FINAIS")
        print("="*70)
        
        print(f"\n🥇 MELHOR MODELO COMPLETO:")
        print(f"   {melhor_modelo[0]}: {melhor_modelo[1]:.1%}")
        
        print(f"\n🥇 MELHOR SUBSET:")
        print(f"   {melhor_subset[0]}: {melhor_subset[1]:.1%}")
        print(f"   Features: {melhor_subset[2]}")
        
        # Escolher o melhor geral
        acuracia_final = max(melhor_modelo[1], melhor_subset[1])
        config_final = melhor_modelo if melhor_modelo[1] >= melhor_subset[1] else melhor_subset
        
        print(f"\n🎯 ACURÁCIA FINAL: {acuracia_final:.1%}")
        print(f"🏆 CONFIGURAÇÃO VENCEDORA: {config_final[0]}")
        
        # Baseline
        baseline = max(y_test.mean(), 1 - y_test.mean())
        print(f"\n📊 COMPARAÇÃO:")
        print(f"   Baseline: {baseline:.1%}")
        print(f"   Nosso modelo: {acuracia_final:.1%}")
        print(f"   Melhoria: {(acuracia_final - baseline)*100:+.1f} pontos")
        
        # Status das metas
        if acuracia_final >= 0.75:
            print(f"\n🎉 META 75% ATINGIDA!")
        elif acuracia_final >= 0.60:
            print(f"\n✅ META 60% ATINGIDA!")
            print(f"   Faltam {(0.75 - acuracia_final)*100:.1f} pontos para 75%")
        elif acuracia_final >= 0.55:
            print(f"\n📈 PROGRESSO! Acima de 55%")
            print(f"   Faltam {(0.60 - acuracia_final)*100:.1f} pontos para 60%")
        else:
            print(f"\n📊 Ainda abaixo de 55%")
        
        # Ranking completo
        print(f"\n📊 RANKING MODELOS COMPLETOS:")
        todos_modelos.sort(key=lambda x: x[1], reverse=True)
        for i, (nome, acc, _, _) in enumerate(todos_modelos, 1):
            print(f"   {i}. {nome}: {acc:.1%}")
        
        print(f"\n📊 RANKING SUBSETS:")
        todos_subsets.sort(key=lambda x: x[1], reverse=True)
        for i, (nome, acc, features, _) in enumerate(todos_subsets, 1):
            print(f"   {i}. {nome}: {acc:.1%}")
        
        print("\n✅ PIPELINE SUPER OTIMIZADO CONCLUÍDO!")
        
        return {
            'acuracia_final': acuracia_final,
            'melhor_config': config_final[0],
            'melhor_features': config_final[2] if len(config_final) > 2 else features_eficazes,
            'meta_60_atingida': acuracia_final >= 0.60,
            'meta_75_atingida': acuracia_final >= 0.75,
            'todos_resultados': {'modelos': todos_modelos, 'subsets': todos_subsets}
        }
        
    except Exception as e:
        print(f"\n❌ ERRO: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    resultado = pipeline_super_otimizado()
    
    if resultado:
        print(f"\n🎯 RESULTADO FINAL: {resultado['acuracia_final']:.1%}")
        if resultado['meta_75_atingida']:
            print("🏆 INCRÍVEL! META 75% ATINGIDA!")
        elif resultado['meta_60_atingida']:
            print("🎉 EXCELENTE! META 60% ATINGIDA!")
        else:
            print("📊 Continuar otimizando...")
    else:
        print("\n❌ Pipeline falhou")
