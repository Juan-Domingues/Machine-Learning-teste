"""
ANÁLISE DIAGNÓSTICA: Por que a acurácia está baixa?
Vamos investigar o que pode estar acontecendo
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def analise_diagnostica():
    """
    Análise diagnóstica profunda dos dados
    """
    print("="*70)
    print("🔍 ANÁLISE DIAGNÓSTICA - POR QUE ACURÁCIA ESTÁ BAIXA?")
    print("="*70)
    
    # Carregar dados
    print("\n📊 Carregando dados para análise...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5 * 365)
    data = yf.download('^BVSP', start=start_date, end=end_date)
    
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    print(f"✅ {len(data)} dias carregados")
    
    # ANÁLISE 1: Distribuição dos retornos
    print("\n🔍 ANÁLISE 1: Distribuição dos Retornos")
    data['Return'] = data['Close'].pct_change()
    data['Direction'] = (data['Return'] > 0).astype(int)
    
    stats = data['Return'].describe()
    print(f"📊 Estatísticas dos retornos diários:")
    print(f"   Média: {stats['mean']:.4f} ({stats['mean']*252:.1%} anualizado)")
    print(f"   Desvio: {stats['std']:.4f} ({stats['std']*np.sqrt(252):.1%} vol anual)")
    print(f"   Min: {stats['min']:.1%} | Max: {stats['max']:.1%}")
    
    # Distribuição de direções
    dir_counts = data['Direction'].value_counts(normalize=True)
    print(f"📊 Distribuição das direções:")
    print(f"   Alta (1): {dir_counts.get(1, 0):.1%}")
    print(f"   Baixa (0): {dir_counts.get(0, 0):.1%}")
    
    # ANÁLISE 2: Autocorrelação
    print(f"\n🔍 ANÁLISE 2: Autocorrelação dos Retornos")
    autocorr_1 = data['Return'].autocorr(lag=1)
    autocorr_2 = data['Return'].autocorr(lag=2)
    autocorr_3 = data['Return'].autocorr(lag=3)
    
    print(f"📊 Autocorrelação dos retornos:")
    print(f"   Lag 1: {autocorr_1:.4f}")
    print(f"   Lag 2: {autocorr_2:.4f}")
    print(f"   Lag 3: {autocorr_3:.4f}")\n    \n    # Autocorrelação das direções\n    data['Direction_lag1'] = data['Direction'].shift(1)\n    direction_corr = data[['Direction', 'Direction_lag1']].corr().iloc[0, 1]\n    print(f\"📊 Autocorrelação das direções: {direction_corr:.4f}\")\n    \n    # ANÁLISE 3: Análise por períodos\n    print(f\"\\n🔍 ANÁLISE 3: Performance por Períodos\")\n    \n    # Últimos 30 dias (período de teste)\n    data_recent = data.tail(30)\n    recent_stats = data_recent['Direction'].value_counts(normalize=True)\n    print(f\"📊 Últimos 30 dias:\")\n    print(f\"   Alta: {recent_stats.get(1, 0):.1%} | Baixa: {recent_stats.get(0, 0):.1%}\")\n    print(f\"   Baseline: {max(recent_stats.get(1, 0), recent_stats.get(0, 0)):.1%}\")\n    \n    # Por ano\n    data['Year'] = data.index.year\n    yearly_stats = data.groupby('Year')['Direction'].agg(['mean', 'count'])\n    print(f\"📊 Performance por ano:\")\n    for year, row in yearly_stats.iterrows():\n        print(f\"   {year}: {row['mean']:.1%} alta ({row['count']} dias)\")\n    \n    # ANÁLISE 4: Teste de features simples\n    print(f\"\\n🔍 ANÁLISE 4: Teste de Features Simples\")\n    \n    # Feature mais simples: retorno anterior\n    data['Target'] = data['Direction'].shift(-1)\n    data['Feature_Return_Lag1'] = data['Return'].shift(1)\n    \n    # Limpar dados\n    clean_data = data[['Feature_Return_Lag1', 'Target']].dropna()\n    \n    # Correlação básica\n    corr = clean_data.corr().iloc[0, 1]\n    print(f\"📊 Correlação Return_lag1 vs Target: {corr:.4f}\")\n    \n    # Teste com threshold simples\n    threshold_results = []\n    for threshold in [-0.005, 0.0, 0.005]:\n        predictions = (clean_data['Feature_Return_Lag1'] > threshold).astype(int)\n        accuracy = accuracy_score(clean_data['Target'], predictions)\n        threshold_results.append((threshold, accuracy))\n        print(f\"   Threshold {threshold:+.3f}: {accuracy:.1%}\")\n    \n    # ANÁLISE 5: Teste com 30 dias específicos\n    print(f\"\\n🔍 ANÁLISE 5: Análise dos Últimos 30 Dias (Teste)\")\n    \n    # Preparar dados para últimos 30 dias\n    n_test = 30\n    train_data = clean_data.iloc[:-n_test]\n    test_data = clean_data.iloc[-n_test:]\n    \n    print(f\"📊 Dados de teste ({len(test_data)} dias):\")\n    test_direction_dist = test_data['Target'].value_counts(normalize=True)\n    print(f\"   Alta: {test_direction_dist.get(1, 0):.1%}\")\n    print(f\"   Baixa: {test_direction_dist.get(0, 0):.1%}\")\n    \n    # Baseline no período de teste\n    test_baseline = max(test_direction_dist.get(1, 0), test_direction_dist.get(0, 0))\n    print(f\"   Baseline teste: {test_baseline:.1%}\")\n    \n    # Modelo simples no teste\n    X_train = train_data[['Feature_Return_Lag1']]\n    y_train = train_data['Target']\n    X_test = test_data[['Feature_Return_Lag1']]\n    y_test = test_data['Target']\n    \n    # Logistic Regression simples\n    model_simple = LogisticRegression(random_state=42)\n    model_simple.fit(X_train, y_train)\n    y_pred_simple = model_simple.predict(X_test)\n    acc_simple = accuracy_score(y_test, y_pred_simple)\n    \n    print(f\"📊 Modelo simples (1 feature):\")\n    print(f\"   Acurácia: {acc_simple:.1%}\")\n    print(f\"   Coeficiente: {model_simple.coef_[0][0]:.3f}\")\n    \n    # Confusion matrix\n    cm = confusion_matrix(y_test, y_pred_simple)\n    print(f\"   Confusion Matrix:\")\n    print(f\"     Pred\\\\Real   0    1\")\n    print(f\"          0    {cm[0,0]:2d}   {cm[0,1]:2d}\")\n    print(f\"          1    {cm[1,0]:2d}   {cm[1,1]:2d}\")\n    \n    # ANÁLISE 6: O que pode estar errado?\n    print(f\"\\n🔍 ANÁLISE 6: Possíveis Problemas\")\n    \n    # Problema 1: Período muito volátil?\n    recent_vol = data_recent['Return'].std()\n    overall_vol = data['Return'].std()\n    print(f\"📊 Volatilidade:\")\n    print(f\"   Geral: {overall_vol:.4f}\")\n    print(f\"   Últimos 30 dias: {recent_vol:.4f}\")\n    print(f\"   Ratio: {recent_vol/overall_vol:.2f}x\")\n    \n    # Problema 2: Tendência forte?\n    recent_trend = data_recent['Return'].mean()\n    overall_trend = data['Return'].mean()\n    print(f\"📊 Tendência (retorno médio):\")\n    print(f\"   Geral: {overall_trend:.4f}\")\n    print(f\"   Últimos 30 dias: {recent_trend:.4f}\")\n    \n    # Problema 3: Outliers?\n    outliers_test = np.abs(test_data['Feature_Return_Lag1']) > 2 * train_data['Feature_Return_Lag1'].std()\n    n_outliers = outliers_test.sum()\n    print(f\"📊 Outliers no teste: {n_outliers}/{len(test_data)} ({n_outliers/len(test_data):.1%})\")\n    \n    # ANÁLISE 7: Sugestões\n    print(f\"\\n💡 SUGESTÕES PARA MELHORAR:\")\n    \n    if test_baseline > 0.6:\n        print(f\"   ✅ Baseline alto ({test_baseline:.1%}) - período favorável\")\n    else:\n        print(f\"   ⚠️ Baseline baixo ({test_baseline:.1%}) - período difícil\")\n    \n    if abs(corr) > 0.05:\n        print(f\"   ✅ Correlação detectável ({corr:.4f})\")\n    else:\n        print(f\"   ⚠️ Correlação muito baixa ({corr:.4f})\")\n    \n    if recent_vol/overall_vol > 1.5:\n        print(f\"   ⚠️ Período de teste muito volátil ({recent_vol/overall_vol:.1f}x)\")\n        print(f\"      Sugestão: Usar período mais estável\")\n    \n    if n_outliers/len(test_data) > 0.2:\n        print(f\"   ⚠️ Muitos outliers no teste ({n_outliers/len(test_data):.1%})\")\n        print(f\"      Sugestão: Filtrar outliers ou usar período diferente\")\n    \n    # Top insights\n    print(f\"\\n🎯 TOP INSIGHTS:\")\n    print(f\"   1. Autocorrelação direção: {direction_corr:.4f}\")\n    print(f\"   2. Baseline período teste: {test_baseline:.1%}\")\n    print(f\"   3. Melhor threshold: {max(threshold_results, key=lambda x: x[1])[1]:.1%}\")\n    print(f\"   4. Modelo simples: {acc_simple:.1%}\")\n    \n    return {\n        'autocorr_direction': direction_corr,\n        'test_baseline': test_baseline,\n        'simple_model_acc': acc_simple,\n        'volatility_ratio': recent_vol/overall_vol,\n        'correlation': corr,\n        'test_data': test_data,\n        'outliers_pct': n_outliers/len(test_data)\n    }\n\ndef estrategia_melhorada(diagnostico):\n    \"\"\"\n    Estratégia melhorada baseada no diagnóstico\n    \"\"\"\n    print(f\"\\n\" + \"=\"*70)\n    print(f\"🚀 ESTRATÉGIA MELHORADA BASEADA NO DIAGNÓSTICO\")\n    print(f\"=\"*70)\n    \n    # Carregar dados novamente\n    end_date = datetime.now()\n    start_date = end_date - timedelta(days=5 * 365)\n    data = yf.download('^BVSP', start=start_date, end=end_date)\n    \n    if isinstance(data.columns, pd.MultiIndex):\n        data.columns = data.columns.get_level_values(0)\n    \n    # Features melhoradas baseadas no diagnóstico\n    print(f\"\\n🔧 Criando features baseadas no diagnóstico...\")\n    \n    data['Return'] = data['Close'].pct_change()\n    data['Return_2d'] = data['Close'].pct_change(2)\n    data['Return_3d'] = data['Close'].pct_change(3)\n    \n    # Médias móveis\n    data['SMA_5'] = data['Close'].rolling(5).mean()\n    data['SMA_10'] = data['Close'].rolling(10).mean()\n    \n    # Features mais sofisticadas\n    data['Price_vs_SMA5'] = (data['Close'] - data['SMA_5']) / data['SMA_5']\n    data['Price_vs_SMA10'] = (data['Close'] - data['SMA_10']) / data['SMA_10']\n    data['SMA5_vs_SMA10'] = (data['SMA_5'] - data['SMA_10']) / data['SMA_10']\n    \n    # Volume\n    data['Volume_SMA'] = data['Volume'].rolling(20).mean()\n    data['Volume_Ratio'] = data['Volume'] / data['Volume_SMA']\n    \n    # Volatilidade\n    data['Vol_5d'] = data['Return'].rolling(5).std()\n    data['Vol_20d'] = data['Return'].rolling(20).mean()\n    data['Vol_Ratio'] = data['Vol_5d'] / data['Vol_20d']\n    \n    # Features lagged\n    data['Return_lag1'] = data['Return'].shift(1)\n    data['Return_lag2'] = data['Return'].shift(2)\n    \n    # Target\n    data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)\n    \n    # Lista de features\n    features = [\n        'Return_lag1', 'Return_lag2', 'Return_2d', 'Return_3d',\n        'Price_vs_SMA5', 'Price_vs_SMA10', 'SMA5_vs_SMA10',\n        'Volume_Ratio', 'Vol_Ratio'\n    ]\n    \n    # Dataset limpo\n    dataset = data[features + ['Target']].dropna()\n    \n    # Divisão específica\n    n_test = 30\n    train_data = dataset.iloc[:-n_test]\n    test_data = dataset.iloc[-n_test:]\n    \n    X_train = train_data[features]\n    y_train = train_data['Target']\n    X_test = test_data[features]\n    y_test = test_data['Target']\n    \n    print(f\"📊 Dataset: {len(X_train)} treino, {len(X_test)} teste\")\n    \n    # ESTRATÉGIA 1: Modelo adaptado ao diagnóstico\n    print(f\"\\n🔹 ESTRATÉGIA 1: Logistic com regularização otimizada\")\n    \n    # Teste diferentes valores de C (regularização)\n    best_c = None\n    best_acc = 0\n    \n    for C in [0.01, 0.1, 1.0, 10.0, 100.0]:\n        model = Pipeline([\n            ('scaler', StandardScaler()),\n            ('clf', LogisticRegression(C=C, random_state=42, max_iter=1000))\n        ])\n        \n        # Cross-validation temporal rápida\n        tscv = TimeSeriesSplit(n_splits=3)\n        scores = []\n        \n        for train_idx, val_idx in tscv.split(X_train):\n            X_tr = X_train.iloc[train_idx]\n            X_val = X_train.iloc[val_idx]\n            y_tr = y_train.iloc[train_idx]\n            y_val = y_train.iloc[val_idx]\n            \n            model.fit(X_tr, y_tr)\n            y_pred_val = model.predict(X_val)\n            score = accuracy_score(y_val, y_pred_val)\n            scores.append(score)\n        \n        avg_score = np.mean(scores)\n        print(f\"   C={C:6.2f}: CV={avg_score:.1%}\")\n        \n        if avg_score > best_acc:\n            best_acc = avg_score\n            best_c = C\n    \n    # Modelo final com melhor C\n    final_model = Pipeline([\n        ('scaler', StandardScaler()),\n        ('clf', LogisticRegression(C=best_c, random_state=42, max_iter=1000))\n    ])\n    \n    final_model.fit(X_train, y_train)\n    y_pred_final = final_model.predict(X_test)\n    acc_final = accuracy_score(y_test, y_pred_final)\n    \n    print(f\"\\n🏆 RESULTADO FINAL:\")\n    print(f\"   Melhor C: {best_c}\")\n    print(f\"   CV Score: {best_acc:.1%}\")\n    print(f\"   Teste Score: {acc_final:.1%}\")\n    \n    # Feature importance\n    feature_coefs = list(zip(features, final_model.named_steps['clf'].coef_[0]))\n    feature_coefs.sort(key=lambda x: abs(x[1]), reverse=True)\n    \n    print(f\"\\n📊 Top 5 Features (coeficientes):\")\n    for feat, coef in feature_coefs[:5]:\n        print(f\"   {feat}: {coef:+.3f}\")\n    \n    return acc_final, final_model, features\n\nif __name__ == \"__main__\":\n    # Executar diagnóstico\n    diagnostico = analise_diagnostica()\n    \n    # Executar estratégia melhorada\n    acc_final, modelo, features = estrategia_melhorada(diagnostico)\n    \n    print(f\"\\n\" + \"=\"*70)\n    print(f\"📊 RESUMO FINAL\")\n    print(f\"=\"*70)\n    print(f\"🎯 Acurácia Final: {acc_final:.1%}\")\n    print(f\"📊 Baseline Teste: {diagnostico['test_baseline']:.1%}\")\n    print(f\"📈 Melhoria: {(acc_final - diagnostico['test_baseline'])*100:+.1f} pontos\")\n    \n    if acc_final >= 0.60:\n        print(f\"🎉 META 60% ATINGIDA!\")\n    elif acc_final >= 0.55:\n        print(f\"📈 Próximo de 60%! Faltam {(0.60 - acc_final)*100:.1f} pontos\")\n    else:\n        print(f\"📊 Ainda trabalhando para 60%...\")
