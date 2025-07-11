"""
ANÁLISE E OTIMIZAÇÃO DE FEATURES
🔍 Investigar e melhorar as features para aumentar performance
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif, RFE

def analise_features_completa():
    """
    Análise completa das features para otimização
    """
    print("="*80)
    print("🔍 ANÁLISE COMPLETA DE FEATURES")
    print("="*80)
    
    # 1. CARREGAMENTO DOS DADOS
    print("\n📊 ETAPA 1: Carregamento dos Dados")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=int(1.5 * 365))
    data = yf.download('^BVSP', start=start_date, end=end_date)
    
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    print(f"📅 Período: {start_date.date()} a {end_date.date()}")
    print(f"📊 Total de dias: {len(data)}")
    
    # 2. CRIAÇÃO DE FEATURES EXPANDIDAS
    print(f"\n🔧 ETAPA 2: Criação de Features Expandidas")
    
    data['Return'] = data['Close'].pct_change()
    data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
    
    # FEATURES TÉCNICAS BÁSICAS
    print("   📈 Criando features técnicas...")
    
    # SMAs de diferentes períodos
    for periodo in [3, 5, 10, 20]:
        data[f'SMA_{periodo}'] = data['Close'].rolling(periodo).mean()
        data[f'Price_above_SMA{periodo}'] = (data['Close'] > data[f'SMA_{periodo}']).astype(int)
    
    # EMAs
    data['EMA_12'] = data['Close'].ewm(span=12).mean()
    data['EMA_26'] = data['Close'].ewm(span=26).mean()
    data['Price_above_EMA12'] = (data['Close'] > data['EMA_12']).astype(int)
    
    # RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    data['RSI_oversold'] = (data['RSI'] < 30).astype(int)
    data['RSI_overbought'] = (data['RSI'] > 70).astype(int)
    
    # MACD
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    data['MACD_signal'] = data['MACD'].ewm(span=9).mean()
    data['MACD_positive'] = (data['MACD'] > data['MACD_signal']).astype(int)
    
    # FEATURES DE VOLUME
    print("   📊 Criando features de volume...")
    for periodo in [10, 20, 30]:
        data[f'Volume_MA_{periodo}'] = data['Volume'].rolling(periodo).mean()
        data[f'Volume_above_avg_{periodo}'] = (data['Volume'] > data[f'Volume_MA_{periodo}']).astype(int)
    
    # Volume relativo
    data['Volume_ratio'] = data['Volume'] / data['Volume_MA_20']
    data['High_volume'] = (data['Volume_ratio'] > 1.5).astype(int)
    
    # FEATURES DE MOMENTUM
    print("   🚀 Criando features de momentum...")
    for lag in [1, 2, 3, 5]:
        data[f'Return_lag{lag}'] = data['Return'].shift(lag)
        data[f'Positive_return_lag{lag}'] = (data[f'Return_lag{lag}'] > 0).astype(int)
    
    # Momentum consecutivo
    data['Momentum_2days'] = ((data['Return_lag1'] > 0) & (data['Return_lag2'] > 0)).astype(int)
    data['Momentum_3days'] = ((data['Return_lag1'] > 0) & (data['Return_lag2'] > 0) & (data['Return_lag3'] > 0)).astype(int)
    
    # FEATURES DE VOLATILIDADE
    print("   📈 Criando features de volatilidade...")
    data['Volatility_10'] = data['Return'].rolling(10).std()
    data['Volatility_20'] = data['Return'].rolling(20).std()
    data['High_volatility'] = (data['Volatility_10'] > data['Volatility_20']).astype(int)
    
    # FEATURES DE PREÇO
    print("   💰 Criando features de preço...")
    data['High_vs_Close'] = (data['High'] - data['Close']) / data['Close']
    data['Low_vs_Close'] = (data['Close'] - data['Low']) / data['Close']
    data['Daily_range'] = (data['High'] - data['Low']) / data['Close']
    data['Large_range'] = (data['Daily_range'] > data['Daily_range'].rolling(20).mean()).astype(int)
    
    # FEATURES DE TENDÊNCIA
    print("   📊 Criando features de tendência...")
    for fast, slow in [(5, 10), (10, 20), (5, 20)]:
        data[f'Trend_{fast}_{slow}'] = (data[f'SMA_{fast}'] > data[f'SMA_{slow}']).astype(int)
    
    # LISTA COMPLETA DE FEATURES
    features_candidatas = [
        # Básicas (atuais)
        'Price_above_SMA5', 'Volume_above_avg_20', 'Positive_return_lag1',
        'Trend_5_10', 'Momentum_2days',
        
        # SMAs adicionais
        'Price_above_SMA3', 'Price_above_SMA10', 'Price_above_SMA20',
        
        # EMAs
        'Price_above_EMA12',
        
        # RSI
        'RSI_oversold', 'RSI_overbought',
        
        # MACD
        'MACD_positive',
        
        # Volume expandido
        'Volume_above_avg_10', 'Volume_above_avg_30', 'High_volume',
        
        # Momentum expandido
        'Positive_return_lag2', 'Positive_return_lag3', 'Positive_return_lag5',
        'Momentum_3days',
        
        # Volatilidade
        'High_volatility',
        
        # Preço
        'Large_range',
        
        # Tendências
        'Trend_10_20', 'Trend_5_20'
    ]
    
    # Remover features que não existem
    features_existentes = []
    for feature in features_candidatas:
        if feature in data.columns:
            features_existentes.append(feature)
        else:
            print(f"   ⚠️ Feature não encontrada: {feature}")
    
    print(f"✅ Total de features candidatas: {len(features_existentes)}")
    
    # 3. PREPARAÇÃO DO DATASET
    dataset = data[features_existentes + ['Target']].dropna()
    print(f"📊 Dataset final: {len(dataset)} observações")
    
    X = dataset[features_existentes]
    y = dataset['Target']
    
    # 4. ANÁLISE DE CORRELAÇÃO
    print(f"\n📊 ETAPA 3: Análise de Correlação com Target")
    
    correlacoes = []
    for feature in features_existentes:
        corr = dataset[feature].corr(dataset['Target'])
        correlacoes.append({'feature': feature, 'correlacao': abs(corr), 'correlacao_raw': corr})
    
    df_corr = pd.DataFrame(correlacoes).sort_values('correlacao', ascending=False)
    
    print(f"📈 TOP 10 FEATURES POR CORRELAÇÃO:")
    for i, row in df_corr.head(10).iterrows():
        print(f"   {i+1:2d}. {row['feature']:25s}: {row['correlacao_raw']:+.3f}")
    
    # 5. SELEÇÃO DE FEATURES
    print(f"\n🎯 ETAPA 4: Seleção de Features")
    
    # Método 1: Top K por correlação
    top_features_corr = df_corr.head(10)['feature'].tolist()
    print(f"📊 Top 10 por correlação: {top_features_corr}")
    
    # Método 2: SelectKBest
    selector = SelectKBest(score_func=f_classif, k=10)
    X_selected = selector.fit_transform(X, y)
    selected_features = [features_existentes[i] for i in selector.get_support(indices=True)]
    print(f"📊 SelectKBest (10): {selected_features}")
    
    # Método 3: RFE (Recursive Feature Elimination)
    estimator = LogisticRegression(C=1.0, random_state=42, max_iter=1000)
    rfe = RFE(estimator, n_features_to_select=10)
    rfe.fit(X, y)
    rfe_features = [features_existentes[i] for i in range(len(features_existentes)) if rfe.support_[i]]
    print(f"📊 RFE (10): {rfe_features}")
    
    # 6. TESTE DE DIFERENTES CONJUNTOS
    print(f"\n🧪 ETAPA 5: Teste de Diferentes Conjuntos de Features")
    
    conjuntos_teste = {
        'Atuais_5': ['Price_above_SMA5', 'Volume_above_avg_20', 'Positive_return_lag1', 'Trend_5_10', 'Momentum_2days'],
        'Top_Correlacao_5': top_features_corr[:5],
        'Top_Correlacao_10': top_features_corr[:10],
        'SelectKBest_5': selected_features[:5],
        'SelectKBest_10': selected_features[:10],
        'RFE_5': rfe_features[:5],
        'RFE_10': rfe_features[:10],
        'Mix_Otimizado': top_features_corr[:3] + selected_features[:2] + rfe_features[:2]  # Mix dos 3 métodos
    }
    
    # Remover duplicatas do mix
    conjuntos_teste['Mix_Otimizado'] = list(dict.fromkeys(conjuntos_teste['Mix_Otimizado']))[:7]
    
    resultados = []
    
    for nome, features_conjunto in conjuntos_teste.items():
        # Verificar se todas as features existem
        features_validas = [f for f in features_conjunto if f in dataset.columns]
        
        if len(features_validas) < 3:  # Mínimo de 3 features
            print(f"   ⚠️ {nome}: Muito poucas features válidas ({len(features_validas)})")
            continue
            
        print(f"\n   🧪 Testando {nome} ({len(features_validas)} features):")
        print(f"      Features: {features_validas}")
        
        X_test = dataset[features_validas]
        
        # Validação cruzada
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = []
        
        for train_idx, test_idx in tscv.split(X_test):
            X_train_cv = X_test.iloc[train_idx]
            X_test_cv = X_test.iloc[test_idx]
            y_train_cv = y.iloc[train_idx]
            y_test_cv = y.iloc[test_idx]
            
            modelo = Pipeline([
                ('scaler', StandardScaler()),
                ('clf', LogisticRegression(C=1.0, random_state=42, max_iter=1000))
            ])
            
            modelo.fit(X_train_cv, y_train_cv)
            score = accuracy_score(y_test_cv, modelo.predict(X_test_cv))
            cv_scores.append(score)
        
        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)
        
        resultado = {
            'nome': nome,
            'features': features_validas,
            'n_features': len(features_validas),
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'cv_scores': cv_scores
        }
        
        resultados.append(resultado)
        
        print(f"      CV: {cv_mean:.1%} ± {cv_std:.1%}")
        print(f"      Folds: {[f'{s:.1%}' for s in cv_scores]}")
    
    # 7. RELATÓRIO FINAL
    print(f"\n" + "="*80)
    print(f"🏆 RELATÓRIO FINAL - OTIMIZAÇÃO DE FEATURES")
    print(f"="*80)
    
    # Ordenar por performance
    resultados.sort(key=lambda x: x['cv_mean'], reverse=True)
    
    print(f"📊 RANKING DE PERFORMANCE:")
    for i, resultado in enumerate(resultados):
        status = "🏆" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "📊"
        print(f"   {status} {i+1:2d}. {resultado['nome']:20s}: {resultado['cv_mean']:.1%} ± {resultado['cv_std']:.1%} ({resultado['n_features']} features)")
    
    # Melhor resultado
    melhor = resultados[0]
    print(f"\n🎯 MELHOR CONJUNTO: {melhor['nome']}")
    print(f"   📊 Performance: {melhor['cv_mean']:.1%} ± {melhor['cv_std']:.1%}")
    print(f"   🔧 Features ({melhor['n_features']}): {melhor['features']}")
    
    # Comparação com atual
    resultado_atual = next((r for r in resultados if r['nome'] == 'Atuais_5'), None)
    if resultado_atual:
        melhoria = (melhor['cv_mean'] - resultado_atual['cv_mean']) * 100
        print(f"   📈 Melhoria vs atual: {melhoria:+.1f} pontos percentuais")
    
    return {
        'melhor_conjunto': melhor,
        'todos_resultados': resultados,
        'correlacoes': df_corr,
        'dataset': dataset
    }

if __name__ == "__main__":
    resultado = analise_features_completa()
    
    melhor = resultado['melhor_conjunto']
    
    print(f"\n" + "="*80)
    print(f"📋 RECOMENDAÇÃO FINAL")
    print(f"="*80)
    
    print(f"🎯 USAR CONJUNTO: {melhor['nome']}")
    print(f"📊 PERFORMANCE ESPERADA: {melhor['cv_mean']:.1%} ± {melhor['cv_std']:.1%}")
    print(f"🔧 FEATURES RECOMENDADAS:")
    for i, feature in enumerate(melhor['features'], 1):
        print(f"   {i}. {feature}")
    
    print(f"\n✅ PRÓXIMO PASSO: Atualizar main.py com essas features!")
