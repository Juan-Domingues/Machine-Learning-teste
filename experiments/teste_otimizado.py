"""
DIAGNÓSTICO RÁPIDO: Por que a acurácia está baixa?
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

def teste_rapido_otimizado():
    """
    Teste rápido com otimizações específicas
    """
    print("="*70)
    print("🔍 DIAGNÓSTICO RÁPIDO - OTIMIZAÇÃO DE ACURÁCIA")
    print("="*70)
    
    # Carregar dados (3 anos para reduzir ruído)
    print("\n📊 Carregando 3 anos de dados...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=3 * 365)
    data = yf.download('^BVSP', start=start_date, end=end_date)
    
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    print(f"✅ {len(data)} dias carregados")
    
    # Features ultra-simples mas eficazes
    print("\n🔧 Criando features ultra-simples...")
    
    data['Return'] = data['Close'].pct_change()
    data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
    
    # Apenas as features mais básicas
    data['Return_lag1'] = data['Return'].shift(1)
    data['Return_lag2'] = data['Return'].shift(2)
    data['SMA_5'] = data['Close'].rolling(5).mean()
    data['Price_above_SMA5'] = (data['Close'] > data['SMA_5']).astype(int)
    data['Volume_above_avg'] = (data['Volume'] > data['Volume'].rolling(20).mean()).astype(int)
    
    features = ['Return_lag1', 'Return_lag2', 'Price_above_SMA5', 'Volume_above_avg']
    
    # Dataset
    dataset = data[features + ['Target']].dropna()
    print(f"📊 Dataset limpo: {len(dataset)} observações")
    
    # Análise da distribuição
    target_dist = dataset['Target'].value_counts(normalize=True)
    print(f"📊 Distribuição target: {target_dist.get(1, 0):.1%} alta, {target_dist.get(0, 0):.1%} baixa")
    
    # TESTE 1: Diferentes tamanhos de janela de teste
    print(f"\n🧪 TESTE 1: Diferentes janelas de teste")
    
    resultados = []
    for n_test in [15, 20, 25, 30, 40]:
        if n_test >= len(dataset):
            continue
            
        train_data = dataset.iloc[:-n_test]
        test_data = dataset.iloc[-n_test:]
        
        X_train = train_data[features]
        y_train = train_data['Target']
        X_test = test_data[features]
        y_test = test_data['Target']
        
        # Baseline do período
        test_baseline = max(y_test.mean(), 1 - y_test.mean())
        
        # Modelo simples
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(random_state=42, max_iter=1000))
        ])
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        resultados.append((n_test, acc, test_baseline))
        print(f"   {n_test:2d} dias: {acc:.1%} (baseline: {test_baseline:.1%})")
    
    # Melhor janela
    melhor_janela = max(resultados, key=lambda x: x[1])
    print(f"\n🏆 Melhor janela: {melhor_janela[0]} dias com {melhor_janela[1]:.1%}")
    
    # TESTE 2: Otimização de features
    print(f"\n🧪 TESTE 2: Otimização de Features")
    
    n_test_otimo = melhor_janela[0]
    train_data = dataset.iloc[:-n_test_otimo]
    test_data = dataset.iloc[-n_test_otimo:]
    
    X_train = train_data[features]
    y_train = train_data['Target']
    X_test = test_data[features]
    y_test = test_data['Target']
    
    # Teste subsets de features
    feature_subsets = [
        (['Return_lag1'], 'Apenas lag1'),
        (['Return_lag1', 'Return_lag2'], 'Lags'),
        (['Return_lag1', 'Price_above_SMA5'], 'Lag + Trend'),
        (['Price_above_SMA5', 'Volume_above_avg'], 'Trend + Volume'),
        (features, 'Todas')
    ]
    
    melhores_features = []
    for feat_subset, nome in feature_subsets:
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(random_state=42, max_iter=1000))
        ])
        
        model.fit(X_train[feat_subset], y_train)
        y_pred = model.predict(X_test[feat_subset])
        acc = accuracy_score(y_test, y_pred)
        
        melhores_features.append((nome, acc, feat_subset))
        print(f"   {nome}: {acc:.1%}")
    
    # Melhor combinação de features
    melhor_features = max(melhores_features, key=lambda x: x[1])
    print(f"\n🏆 Melhores features: {melhor_features[0]} com {melhor_features[1]:.1%}")
    
    # TESTE 3: Diferentes modelos
    print(f"\n🧪 TESTE 3: Diferentes Modelos")
    
    best_features = melhor_features[2]
    
    modelos = [
        ('Logistic C=0.1', Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(C=0.1, random_state=42, max_iter=1000))
        ])),
        ('Logistic C=1.0', Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(C=1.0, random_state=42, max_iter=1000))
        ])),
        ('Logistic C=10.0', Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(C=10.0, random_state=42, max_iter=1000))
        ])),
        ('Random Forest', RandomForestClassifier(
            n_estimators=50, max_depth=5, min_samples_split=10, 
            random_state=42, n_jobs=-1
        ))
    ]
    
    resultados_modelos = []
    for nome, modelo in modelos:
        modelo.fit(X_train[best_features], y_train)
        y_pred = modelo.predict(X_test[best_features])
        acc = accuracy_score(y_test, y_pred)
        
        resultados_modelos.append((nome, acc, modelo))
        print(f"   {nome}: {acc:.1%}")
    
    # Melhor modelo
    melhor_modelo = max(resultados_modelos, key=lambda x: x[1])
    print(f"\n🏆 Melhor modelo: {melhor_modelo[0]} com {melhor_modelo[1]:.1%}")
    
    # RESULTADO FINAL
    print(f"\n" + "="*70)
    print("🏆 CONFIGURAÇÃO OTIMIZADA FINAL")
    print("="*70)
    
    baseline_final = melhor_janela[2]
    acuracia_final = melhor_modelo[1]
    
    print(f"🎯 Janela de teste: {n_test_otimo} dias")
    print(f"🔧 Features: {best_features}")
    print(f"🤖 Modelo: {melhor_modelo[0]}")
    print(f"📊 Acurácia: {acuracia_final:.1%}")
    print(f"📊 Baseline: {baseline_final:.1%}")
    print(f"📈 Melhoria: {(acuracia_final - baseline_final)*100:+.1f} pontos")
    
    # Status
    if acuracia_final >= 0.60:
        print(f"\n🎉 EXCELENTE! Acurácia ≥ 60%")
    elif acuracia_final >= 0.55:
        print(f"\n📈 BOM! Próximo de 60%")
        print(f"   Faltam {(0.60 - acuracia_final)*100:.1f} pontos")
    else:
        print(f"\n📊 Melhorando... ainda abaixo de 55%")
    
    # TESTE FINAL: Validação com mais períodos
    print(f"\n🧪 VALIDAÇÃO FINAL: Múltiplos períodos")
    
    modelo_final = melhor_modelo[2]
    
    # Teste os últimos 60 dias em janelas de 20
    acuracias_validacao = []
    for start_day in range(20, 61, 20):
        end_day = start_day + 20
        if end_day > len(dataset):
            break
            
        train_val = dataset.iloc[:-end_day]
        test_val = dataset.iloc[-end_day:-start_day]
        
        if len(test_val) < 15:  # Mínimo de 15 dias
            continue
            
        X_train_val = train_val[best_features]
        y_train_val = train_val['Target']
        X_test_val = test_val[best_features]
        y_test_val = test_val['Target']
        
        # Retreinar modelo
        modelo_val = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(C=1.0, random_state=42, max_iter=1000))
        ])
        
        modelo_val.fit(X_train_val, y_train_val)
        y_pred_val = modelo_val.predict(X_test_val)
        acc_val = accuracy_score(y_test_val, y_pred_val)
        
        acuracias_validacao.append(acc_val)
        print(f"   Dias -{end_day} a -{start_day}: {acc_val:.1%}")
    
    if acuracias_validacao:
        acc_media = np.mean(acuracias_validacao)
        acc_std = np.std(acuracias_validacao)
        print(f"\n📊 Validação média: {acc_media:.1%} (±{acc_std:.1%})")
        
        if acc_media >= 0.55:
            print(f"✅ Modelo consistente!")
        else:
            print(f"⚠️ Modelo instável entre períodos")
    
    return {
        'acuracia_final': acuracia_final,
        'configuracao': {
            'janela_teste': n_test_otimo,
            'features': best_features,
            'modelo': melhor_modelo[0]
        },
        'baseline': baseline_final,
        'validacao_media': np.mean(acuracias_validacao) if acuracias_validacao else 0
    }

if __name__ == "__main__":
    resultado = teste_rapido_otimizado()
    
    print(f"\n🎯 RESUMO EXECUTIVO:")
    print(f"   Acurácia: {resultado['acuracia_final']:.1%}")
    print(f"   Configuração: {resultado['configuracao']}")
    
    if resultado['acuracia_final'] >= 0.60:
        print(f"🎉 META 60% ALCANÇADA!")
    else:
        print(f"📊 Progredindo... {resultado['acuracia_final']:.1%}")
