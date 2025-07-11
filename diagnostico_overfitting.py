"""
DIAGNÓSTICO DE OVERFITTING - REVISÃO TÉCNICA
🔍 Verificação rigorosa do pipeline de 75%
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def diagnostico_completo():
    """
    Diagnóstico completo para detectar overfitting
    """
    print("="*80)
    print("🔍 DIAGNÓSTICO DE OVERFITTING - PIPELINE 75%")
    print("="*80)
    
    # 1. CARREGAMENTO DOS DADOS
    print("\n📊 ETAPA 1: Análise dos Dados")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=int(1.5 * 365))
    data = yf.download('^BVSP', start=start_date, end=end_date)
    
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    print(f"📅 Período: {start_date.date()} a {end_date.date()}")
    print(f"📊 Total de dias: {len(data)}")
    print(f"📈 Dados faltantes: {data.isnull().sum().sum()}")
    
    # Estatísticas descritivas
    print(f"\n📊 ESTATÍSTICAS DESCRITIVAS:")
    print(f"   Close - Min: {data['Close'].min():.2f}, Max: {data['Close'].max():.2f}")
    print(f"   Volume - Min: {data['Volume'].min():,.0f}, Max: {data['Volume'].max():,.0f}")
    
    # 2. CRIAÇÃO DAS FEATURES (EXATAMENTE COMO NO MAIN)
    print(f"\n🔧 ETAPA 2: Criação das Features")
    
    data['Return'] = data['Close'].pct_change()
    data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
    
    # TRIO BASE
    data['SMA_5'] = data['Close'].rolling(5).mean()
    data['Price_above_SMA5'] = (data['Close'] > data['SMA_5']).astype(int)
    
    data['Volume_MA_20'] = data['Volume'].rolling(20).mean()
    data['Volume_above_avg'] = (data['Volume'] > data['Volume_MA_20']).astype(int)
    
    data['Return_lag1'] = data['Return'].shift(1)
    data['Positive_return_lag1'] = (data['Return_lag1'] > 0).astype(int)
    
    # FEATURES DIFERENCIAIS
    data['SMA_10'] = data['Close'].rolling(10).mean()
    data['Trend_strong'] = (data['SMA_5'] > data['SMA_10']).astype(int)
    
    data['Return_lag2'] = data['Return'].shift(2)
    data['Momentum_strong'] = ((data['Return_lag1'] > 0) & (data['Return_lag2'] > 0)).astype(int)
    
    features = [
        'Price_above_SMA5', 'Volume_above_avg', 'Positive_return_lag1',
        'Trend_strong', 'Momentum_strong'
    ]
    
    dataset = data[features + ['Target']].dropna()
    
    print(f"✅ Features criadas: {len(features)}")
    print(f"📊 Dataset final: {len(dataset)} observações")
    
    # 3. ANÁLISE EXPLORATÓRIA DAS FEATURES
    print(f"\n📈 ETAPA 3: Análise Exploratória das Features")
    
    X = dataset[features]
    y = dataset['Target']
    
    print(f"📊 Distribuição do Target:")
    target_dist = y.value_counts(normalize=True)
    print(f"   Classe 0 (Baixa): {target_dist[0]:.1%}")
    print(f"   Classe 1 (Alta): {target_dist[1]:.1%}")
    
    print(f"\n📊 Correlação Features vs Target:")
    for feature in features:
        corr = dataset[feature].corr(dataset['Target'])
        print(f"   {feature}: {corr:.3f}")
    
    print(f"\n📊 Estatísticas das Features:")
    for feature in features:
        mean_val = dataset[feature].mean()
        std_val = dataset[feature].std()
        print(f"   {feature}: Mean={mean_val:.3f}, Std={std_val:.3f}")
    
    # 4. TESTE DE MÚLTIPLAS DIVISÕES TEMPORAIS
    print(f"\n🔍 ETAPA 4: Teste de Múltiplas Divisões (Anti-Overfitting)")
    
    # Testando diferentes janelas de teste
    janelas_teste = [10, 15, 20, 25, 30]
    resultados_janelas = []
    
    for janela in janelas_teste:
        if len(dataset) > janela + 50:  # Garantir dados suficientes
            train_data = dataset.iloc[:-janela]
            test_data = dataset.iloc[-janela:]
            
            X_train = train_data[features]
            y_train = train_data['Target']
            X_test = test_data[features]
            y_test = test_data['Target']
            
            modelo = Pipeline([
                ('scaler', StandardScaler()),
                ('clf', VotingClassifier([
                    ('lr', LogisticRegression(C=1.0, random_state=42)),
                    ('rf', RandomForestClassifier(n_estimators=150, max_depth=7, random_state=42))
                ], voting='hard'))
            ])
            
            modelo.fit(X_train, y_train)
            y_pred = modelo.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            
            baseline = max(y_test.mean(), 1 - y_test.mean())
            
            resultados_janelas.append({
                'janela': janela,
                'acuracia': acc,
                'baseline': baseline,
                'n_test': len(y_test),
                'n_train': len(y_train)
            })
            
            print(f"   Janela {janela:2d} dias: {acc:.1%} (baseline: {baseline:.1%}, n_test: {len(y_test)})")
    
    # 5. VALIDAÇÃO CRUZADA TEMPORAL RIGOROSA
    print(f"\n🔍 ETAPA 5: Validação Cruzada Temporal (5 Folds)")
    
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = []
    cv_baselines = []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train_cv = X.iloc[train_idx]
        X_test_cv = X.iloc[test_idx]
        y_train_cv = y.iloc[train_idx]
        y_test_cv = y.iloc[test_idx]
        
        modelo_cv = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', VotingClassifier([
                ('lr', LogisticRegression(C=1.0, random_state=42)),
                ('rf', RandomForestClassifier(n_estimators=150, max_depth=7, random_state=42))
            ], voting='hard'))
        ])
        
        modelo_cv.fit(X_train_cv, y_train_cv)
        score = accuracy_score(y_test_cv, modelo_cv.predict(X_test_cv))
        baseline = max(y_test_cv.mean(), 1 - y_test_cv.mean())
        
        cv_scores.append(score)
        cv_baselines.append(baseline)
        
        print(f"   Fold {fold+1}: {score:.1%} (baseline: {baseline:.1%}, test_size: {len(y_test_cv)})")
    
    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores)
    baseline_mean = np.mean(cv_baselines)
    
    print(f"\n📊 CV FINAL: {cv_mean:.1%} ± {cv_std:.1%}")
    print(f"📊 BASELINE MÉDIO: {baseline_mean:.1%}")
    
    # 6. TESTE COM MODELO MAIS SIMPLES (BASELINE)
    print(f"\n🔍 ETAPA 6: Comparação com Modelos Mais Simples")
    
    # Último teste como no main
    n_test = 20
    train_data = dataset.iloc[:-n_test]
    test_data = dataset.iloc[-n_test:]
    
    X_train = train_data[features]
    y_train = train_data['Target']
    X_test = test_data[features]
    y_test = test_data['Target']
    
    # Modelo simples (apenas Logistic Regression)
    modelo_simples = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(C=1.0, random_state=42))
    ])
    
    modelo_simples.fit(X_train, y_train)
    acc_simples = accuracy_score(y_test, modelo_simples.predict(X_test))
    
    # Modelo complexo (como no main)
    modelo_complexo = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', VotingClassifier([
            ('lr', LogisticRegression(C=1.0, random_state=42)),
            ('rf', RandomForestClassifier(n_estimators=150, max_depth=7, random_state=42))
        ], voting='hard'))
    ])
    
    modelo_complexo.fit(X_train, y_train)
    acc_complexo = accuracy_score(y_test, modelo_complexo.predict(X_test))
    
    baseline_final = max(y_test.mean(), 1 - y_test.mean())
    
    print(f"   Modelo Simples (LogReg): {acc_simples:.1%}")
    print(f"   Modelo Complexo (Ensemble): {acc_complexo:.1%}")
    print(f"   Baseline: {baseline_final:.1%}")
    print(f"   Diferença: {(acc_complexo - acc_simples)*100:+.1f} pontos")
    
    # 7. ANÁLISE DE ESTABILIDADE TEMPORAL
    print(f"\n🔍 ETAPA 7: Análise de Estabilidade Temporal")
    
    # Dividindo em 3 períodos
    n_total = len(dataset)
    periodo1 = dataset.iloc[:n_total//3]
    periodo2 = dataset.iloc[n_total//3:2*n_total//3]
    periodo3 = dataset.iloc[2*n_total//3:]
    
    print(f"   Período 1: {periodo1['Target'].mean():.1%} positivos")
    print(f"   Período 2: {periodo2['Target'].mean():.1%} positivos")
    print(f"   Período 3: {periodo3['Target'].mean():.1%} positivos")
    
    # 8. DIAGNÓSTICO FINAL
    print(f"\n" + "="*80)
    print(f"🏆 DIAGNÓSTICO FINAL - DETECÇÃO DE OVERFITTING")
    print(f"="*80)
    
    # Critérios de overfitting
    overfitting_flags = []
    
    # 1. CV muito diferente do resultado final
    if abs(cv_mean - acc_complexo) > 0.15:  # 15 pontos de diferença
        overfitting_flags.append("CV vs Final muito diferente")
    
    # 2. Variabilidade alta no CV
    if cv_std > 0.20:  # 20 pontos de desvio padrão
        overfitting_flags.append("CV com alta variabilidade")
    
    # 3. Modelo complexo muito melhor que simples
    if (acc_complexo - acc_simples) > 0.20:  # 20 pontos melhor
        overfitting_flags.append("Modelo complexo suspeito")
    
    # 4. Resultado muito acima da média do CV
    if acc_complexo > (cv_mean + 2*cv_std):
        overfitting_flags.append("Resultado final suspeito")
    
    # 5. Poucos dados de teste
    if n_test < 15:
        overfitting_flags.append("Poucos dados de teste")
    
    print(f"🎯 RESULTADO PRINCIPAL: {acc_complexo:.1%}")
    print(f"📊 CV MÉDIO: {cv_mean:.1%} ± {cv_std:.1%}")
    print(f"🔢 FEATURES: {len(features)}")
    print(f"📊 DADOS DE TESTE: {n_test}")
    print(f"📈 BASELINE: {baseline_final:.1%}")
    
    print(f"\n🚨 FLAGS DE OVERFITTING:")
    if overfitting_flags:
        for flag in overfitting_flags:
            print(f"   ❌ {flag}")
        print(f"\n⚠️ ALERTA: Possível overfitting detectado!")
        print(f"🔍 RECOMENDAÇÃO: Investigar mais ou usar CV como resultado")
    else:
        print(f"   ✅ Nenhum flag crítico detectado")
        print(f"   ✅ Resultado parece legítimo")
        print(f"\n🏆 CONCLUSÃO: Pipeline aprovado!")
    
    # Resumo das métricas
    print(f"\n📊 RESUMO DAS MÉTRICAS:")
    print(f"   Acurácia Final: {acc_complexo:.1%}")
    print(f"   CV Médio: {cv_mean:.1%}")
    print(f"   Modelo Simples: {acc_simples:.1%}")
    print(f"   Baseline: {baseline_final:.1%}")
    print(f"   Estabilidade CV: {cv_std:.1%}")
    
    return {
        'acuracia_final': acc_complexo,
        'cv_mean': cv_mean,
        'cv_std': cv_std,
        'modelo_simples': acc_simples,
        'baseline': baseline_final,
        'overfitting_flags': overfitting_flags,
        'n_features': len(features),
        'n_test': n_test,
        'resultados_janelas': resultados_janelas
    }

if __name__ == "__main__":
    resultado = diagnostico_completo()
    
    print(f"\n" + "="*80)
    print(f"📋 RELATÓRIO EXECUTIVO - OVERFITTING CHECK")
    print(f"="*80)
    
    if len(resultado['overfitting_flags']) == 0:
        print(f"✅ PIPELINE APROVADO - SEM OVERFITTING DETECTADO")
        print(f"🎯 Resultado: {resultado['acuracia_final']:.1%} é confiável")
    else:
        print(f"⚠️ ATENÇÃO - POSSÍVEL OVERFITTING")
        print(f"🔍 Usar CV como referência: {resultado['cv_mean']:.1%}")
    
    print(f"\n📊 MÉTRICAS CHAVE:")
    print(f"   Final: {resultado['acuracia_final']:.1%}")
    print(f"   CV: {resultado['cv_mean']:.1%} ± {resultado['cv_std']:.1%}")
    print(f"   Simples: {resultado['modelo_simples']:.1%}")
    print(f"   Flags: {len(resultado['overfitting_flags'])}")
