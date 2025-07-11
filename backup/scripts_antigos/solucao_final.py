"""
SOLUÇÃO FINAL - Machine Learning IBOVESPA
Baseado na descoberta: previsão de alta volatilidade (93.5% acurácia)
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif

def carregar_dados_ibovespa():
    """Carrega dados históricos do IBOVESPA"""
    print("📈 SOLUÇÃO FINAL - Previsão de Volatilidade IBOVESPA")
    print("="*60)
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=15 * 365)
    
    # IBOVESPA + contexto macroeconômico
    ibov = yf.download('^BVSP', start=start_date, end=end_date)
    if isinstance(ibov.columns, pd.MultiIndex):
        ibov.columns = ibov.columns.get_level_values(0)
    
    sp500 = yf.download('^GSPC', start=start_date, end=end_date)['Close']
    dollar = yf.download('USDBRL=X', start=start_date, end=end_date)['Close']
    
    data = ibov.copy()
    data['SP500'] = sp500
    data['USD_BRL'] = dollar
    
    print(f"✓ {len(data)} dias carregados")
    return data

def criar_features_volatilidade(data):
    """Features específicas para previsão de volatilidade"""
    print("🔧 Criando features para previsão de volatilidade...")
    
    # Retornos básicos
    data['Retorno'] = data['Close'].pct_change()
    data['Retorno_Abs'] = abs(data['Retorno'])
    
    # Features de reversão (sinal negativo detectado)
    data['Retorno_Lag1'] = data['Retorno'].shift(1)
    data['Retorno_Lag2'] = data['Retorno'].shift(2)
    data['Retorno_Lag3'] = data['Retorno'].shift(3)
    
    # Volatilidade multi-escala (principal driver)
    for window in [5, 10, 20]:
        data[f'Vol_{window}d'] = data['Retorno'].rolling(window).std()
        data[f'Vol_Rank_{window}d'] = data[f'Vol_{window}d'].rolling(252).rank(pct=True)
    
    # Momentum features
    for window in [5, 10, 20]:
        data[f'Mom_{window}d'] = data['Close'].pct_change(window)
    
    # Contexto macroeconômico
    if 'SP500' in data.columns:
        data['SP500_Ret'] = data['SP500'].pct_change()
        data['SP500_Vol'] = data['SP500_Ret'].rolling(20).std()
    
    if 'USD_BRL' in data.columns:
        data['USD_Ret'] = data['USD_BRL'].pct_change()
        data['USD_Vol'] = data['USD_Ret'].rolling(20).std()
    
    # Features de regime
    data['High_Vol_Recent'] = (data['Vol_5d'] > data['Vol_5d'].rolling(20).quantile(0.8)).astype(int)
    data['Trend_Change'] = ((data['Close'] > data['Close'].shift(1)) != 
                           (data['Close'].shift(1) > data['Close'].shift(2))).astype(int)
    
    # Volume features
    if 'Volume' in data.columns:
        data['Volume_MA'] = data['Volume'].rolling(20).mean()
        data['Volume_Spike'] = (data['Volume'] > 1.5 * data['Volume_MA']).astype(int)
    
    # RSI e extremos
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    data['RSI'] = 100 - (100 / (1 + gain / loss))
    data['RSI_Extreme'] = ((data['RSI'] < 30) | (data['RSI'] > 70)).astype(int)
    
    # Features de sequência
    data['Big_Move_Yesterday'] = (abs(data['Retorno_Lag1']) > 0.015).astype(int)
    data['Consecutive_Vol'] = ((data['Vol_5d'] > data['Vol_5d'].shift(1)) & 
                              (data['Vol_5d'].shift(1) > data['Vol_5d'].shift(2))).astype(int)
    
    features_list = [
        'Retorno_Lag1', 'Retorno_Lag2', 'Retorno_Lag3',
        'Vol_5d', 'Vol_10d', 'Vol_20d',
        'Vol_Rank_5d', 'Vol_Rank_10d', 'Vol_Rank_20d',
        'Mom_5d', 'Mom_10d', 'Mom_20d',
        'High_Vol_Recent', 'Trend_Change', 'RSI_Extreme',
        'Big_Move_Yesterday', 'Consecutive_Vol'
    ]
    
    # Adicionar features condicionais
    if 'SP500_Vol' in data.columns:
        features_list.extend(['SP500_Ret', 'SP500_Vol'])
    if 'USD_Vol' in data.columns:
        features_list.extend(['USD_Ret', 'USD_Vol'])
    if 'Volume_Spike' in data.columns:
        features_list.append('Volume_Spike')
    
    print(f"✓ {len(features_list)} features criadas para volatilidade")
    return data, features_list

def criar_target_volatilidade(data):
    """Target: previsão de alta volatilidade (descoberta do diagnóstico)"""
    print("🎯 Criando target de alta volatilidade...")
    
    # Usar volatilidade rolling do retorno futuro
    vol_futuro = data['Retorno'].shift(-1).abs()  # Simplificar: valor absoluto do retorno
    
    # Threshold: top 25% dos dias mais voláteis (janela móvel)
    threshold = vol_futuro.rolling(60, min_periods=30).quantile(0.75)
    
    # Target: 1 = alta volatilidade esperada, 0 = volatilidade normal/baixa  
    data['Target_HighVol'] = (vol_futuro > threshold).astype(int)
    
    # Remover NaN
    data['Target_HighVol'] = data['Target_HighVol'].fillna(0)
    
    # Estatísticas
    target_stats = data['Target_HighVol'].value_counts()
    total = len(data['Target_HighVol'])
    
    print(f"✓ Target criado:")
    print(f"  Alta volatilidade: {target_stats.get(1, 0)} dias ({target_stats.get(1, 0)/total:.1%})")
    print(f"  Volatilidade normal: {target_stats.get(0, 0)} dias ({target_stats.get(0, 0)/total:.1%})")
    
    return data

def treinar_modelo_volatilidade(X, y):
    """Treina modelo otimizado para previsão de volatilidade"""
    print("🤖 Treinando modelo de previsão de volatilidade...")
    
    # Usar Time Series Split para validação temporal
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Modelo descoberto como melhor no diagnóstico
    modelo = LogisticRegression(random_state=42, max_iter=1000)
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('selector', SelectKBest(f_classif, k=min(15, X.shape[1]))),
        ('classifier', modelo)
    ])
    
    # Cross validation temporal
    scores = cross_val_score(pipeline, X, y, cv=tscv, scoring='accuracy')
    
    print(f"✓ Cross-validation: {scores.mean():.1%} (±{scores.std():.1%})")
    
    # Baseline
    baseline = max(y.mean(), 1 - y.mean())
    improvement = scores.mean() - baseline
    
    print(f"✓ Baseline: {baseline:.1%}")
    print(f"✓ Melhoria: {improvement*100:+.1f} pontos percentuais")
    
    return pipeline, scores

def teste_final_volatilidade(pipeline, X, y):
    """Teste final com hold-out"""
    print("📊 Teste final com dados de hold-out...")
    
    # Split temporal: 80% treino, 20% teste
    split_point = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
    y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]
    
    # Treinar
    pipeline.fit(X_train, y_train)
    
    # Predizer
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    
    # Métricas
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"✓ Acurácia final: {accuracy:.1%}")
    print(f"✓ Período de teste: {len(X_test)} dias")
    
    # Relatório detalhado
    print("\n📋 Relatório de classificação:")
    print(classification_report(y_test, y_pred, target_names=['Vol Normal', 'Vol Alta']))
    
    # Feature importance
    if hasattr(pipeline.named_steps['classifier'], 'coef_'):
        # Obter features selecionadas
        selector = pipeline.named_steps['selector']
        selected_features = X.columns[selector.get_support()]
        coef = pipeline.named_steps['classifier'].coef_[0]
        
        feature_importance = pd.DataFrame({
            'feature': selected_features,
            'coefficient': coef,
            'abs_coef': abs(coef)
        }).sort_values('abs_coef', ascending=False)
        
        print(f"\n🔍 Features mais importantes:")
        for _, row in feature_importance.head(8).iterrows():
            print(f"  {row['feature']}: {row['coefficient']:+.3f}")
    
    return accuracy, y_test, y_pred, y_pred_proba

def interpretacao_resultados(accuracy, baseline_accuracy=0.74):
    """Interpreta os resultados do modelo"""
    print("\n" + "="*60)
    print("INTERPRETAÇÃO DOS RESULTADOS")
    print("="*60)
    
    if accuracy > 0.90:
        print("🎉 EXCELENTE! Modelo muito preciso para volatilidade")
        print("💡 Aplicações práticas:")
        print("   - Ajuste de estratégias de risco")
        print("   - Timing para hedge de posições")
        print("   - Precificação de opções")
        print("   - Alocação dinâmica de ativos")
        
    elif accuracy > 0.80:
        print("✅ MUITO BOM! Modelo útil para gestão de risco")
        print("💡 Pode ser usado para:")
        print("   - Alerta de períodos de alta volatilidade")
        print("   - Ajuste de stop-loss dinâmico")
        
    elif accuracy > baseline_accuracy:
        print(f"📈 BOM! Modelo supera baseline ({baseline_accuracy:.1%})")
        print("💡 Tem valor preditivo, mas com cautela")
        
    else:
        print("📊 Modelo não supera baseline significativamente")
    
    print(f"\n🎯 DESCOBERTA PRINCIPAL:")
    print(f"   Embora seja difícil prever DIREÇÃO do mercado,")
    print(f"   conseguimos prever VOLATILIDADE com {accuracy:.1%} de acurácia!")

def main():
    """Pipeline principal da solução final"""
    try:
        # 1. Carregar dados
        data = carregar_dados_ibovespa()
        
        # 2. Features para volatilidade
        data, features = criar_features_volatilidade(data)
        
        # 3. Target de volatilidade
        data = criar_target_volatilidade(data)
        
        # 4. Preparar dataset limpo
        dataset = data[features + ['Target_HighVol']].dropna()
        X = dataset[features]
        y = dataset['Target_HighVol']
        
        print(f"\n✓ Dataset final: {len(X)} observações válidas")
        
        # 5. Treinar modelo
        pipeline, cv_scores = treinar_modelo_volatilidade(X, y)
        
        # 6. Teste final
        final_accuracy, y_test, y_pred, y_pred_proba = teste_final_volatilidade(pipeline, X, y)
        
        # 7. Interpretação
        interpretacao_resultados(final_accuracy)
        
        print(f"\n✅ PROJETO CONCLUÍDO COM SUCESSO!")
        print(f"   Acurácia final: {final_accuracy:.1%}")
        print(f"   Cross-validation: {cv_scores.mean():.1%} (±{cv_scores.std():.1%})")
        
        return {
            'modelo': pipeline,
            'acuracia': final_accuracy,
            'cv_scores': cv_scores,
            'features': features
        }
        
    except Exception as e:
        print(f"❌ Erro na execução: {e}")
        return None

if __name__ == "__main__":
    resultados = main()
    
    if resultados and resultados['acuracia'] > 0.85:
        print(f"\n🏆 MISSÃO CUMPRIDA!")
        print(f"   Modelo de volatilidade com {resultados['acuracia']:.1%} de precisão!")
    elif resultados:
        print(f"\n📊 Projeto concluído. Acurácia: {resultados['acuracia']:.1%}")
    else:
        print(f"\n❌ Falha na execução do projeto.")
