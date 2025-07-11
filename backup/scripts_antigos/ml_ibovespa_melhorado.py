"""
Machine Learning - Previsão IBOVESPA (VERSÃO MELHORADA)
Baseado no diagnóstico completo do projeto original
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LogisticRegression  # Mudança para classificação
from sklearn.ensemble import RandomForestClassifier  # Modelo mais robusto
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, make_scorer

def carregar_dados_ibovespa(anos=10):
    """Carrega dados históricos do IBOVESPA"""
    print("ETAPA 1: Carregando dados...")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=anos * 365)
    
    data = yf.download('^BVSP', start=start_date, end=end_date)
    
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    print(f"✓ {len(data)} dias carregados")
    return data

def criar_features_melhoradas(data):
    """Cria features mais robustas baseadas no diagnóstico"""
    print("ETAPA 2: Criando features melhoradas...")
    
    # Retornos
    data['Retorno'] = data['Close'].pct_change()
    
    # Features baseadas em reversão à média (autocorr negativa detectada)
    data['Retorno_Lag1'] = data['Retorno'].shift(1)  # Usar reversão
    data['Retorno_Lag2'] = data['Retorno'].shift(2)
    
    # Médias móveis mais simples
    data['MM10'] = data['Close'].rolling(10).mean()
    data['MM30'] = data['Close'].rolling(30).mean()
    
    # Sinais binários mais claros
    data['Acima_MM10'] = (data['Close'] > data['MM10']).astype(int)
    data['Acima_MM30'] = (data['Close'] > data['MM30']).astype(int)
    data['MM_Trend'] = (data['MM10'] > data['MM30']).astype(int)
    
    # Volatilidade regime
    data['Vol'] = data['Retorno'].rolling(20).std()
    data['Vol_High'] = (data['Vol'] > data['Vol'].rolling(60).quantile(0.7)).astype(int)
    
    # Volume features simplificadas
    data['Vol_MA'] = data['Volume'].rolling(20).mean()
    data['Vol_High_Flag'] = (data['Volume'] > 1.2 * data['Vol_MA']).astype(int)
    
    # RSI simples
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    data['RSI'] = 100 - (100 / (1 + gain / loss))
    data['RSI_Oversold'] = (data['RSI'] < 35).astype(int)
    data['RSI_Overbought'] = (data['RSI'] > 65).astype(int)
    
    # Features lag para capturar reversão
    data['Big_Move_Yesterday'] = (abs(data['Retorno'].shift(1)) > 0.02).astype(int)
    data['Consecutive_Up'] = ((data['Retorno'].shift(1) > 0) & 
                             (data['Retorno'].shift(2) > 0)).astype(int)
    data['Consecutive_Down'] = ((data['Retorno'].shift(1) < 0) & 
                               (data['Retorno'].shift(2) < 0)).astype(int)
    
    features_lista = [
        'Retorno_Lag1', 'Retorno_Lag2', 'Acima_MM10', 'Acima_MM30', 'MM_Trend',
        'Vol_High', 'Vol_High_Flag', 'RSI_Oversold', 'RSI_Overbought',
        'Big_Move_Yesterday', 'Consecutive_Up', 'Consecutive_Down'
    ]
    
    print(f"✓ {len(features_lista)} features criadas (foco em reversão à média)")
    return data, features_lista

def criar_target_classificacao(data, threshold=0.002):
    """Cria target de classificação com threshold para reduzir ruído"""
    print("ETAPA 3: Criando target de classificação...")
    
    # Target com threshold para reduzir ruído
    retorno_futuro = data['Retorno'].shift(-1)
    
    # Classificação: 1 = alta significativa, 0 = baixa/lateral
    data['Target'] = (retorno_futuro > threshold).astype(int)
    
    # Estatísticas
    target_counts = data['Target'].value_counts()
    total = len(data['Target'].dropna())
    
    print(f"✓ Target criado com threshold {threshold*100:.1f}%")
    print(f"  Altas significativas: {target_counts[1]/total:.1%}")
    print(f"  Baixas/laterais: {target_counts[0]/total:.1%}")
    
    return data

def testar_modelos_melhorados(X, y):
    """Testa diferentes modelos de classificação"""
    print("ETAPA 4: Testando modelos melhorados...")
    
    modelos = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42, max_depth=5)
    }
    
    resultados = {}
    cv = KFold(n_splits=5, shuffle=False)
    
    for nome, modelo in modelos.items():
        # Pipeline com normalização
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', modelo)
        ])
        
        # Cross validation
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')
        
        resultados[nome] = {
            'mean': scores.mean(),
            'std': scores.std(),
            'scores': scores
        }
        
        print(f"  {nome}: {scores.mean():.1%} (±{scores.std():.1%})")
    
    return resultados

def main_melhorado():
    """Pipeline melhorado baseado no diagnóstico"""
    print("="*60)
    print("MACHINE LEARNING MELHORADO - IBOVESPA")
    print("Soluções baseadas no diagnóstico completo")
    print("="*60)
    
    try:
        # 1. Carregar dados
        data = carregar_dados_ibovespa(anos=10)
        
        # 2. Features melhoradas
        data, features_lista = criar_features_melhoradas(data)
        
        # 3. Target de classificação com threshold
        data = criar_target_classificacao(data, threshold=0.002)
        
        # 4. Preparar dataset
        dataset = data[features_lista + ['Target']].dropna()
        X = dataset[features_lista]
        y = dataset['Target']
        
        print(f"\n✓ Dataset preparado: {len(X)} observações")
        
        # 5. Testar modelos
        resultados_cv = testar_modelos_melhorados(X, y)
        
        # 6. Teste final com melhor modelo
        print("\nETAPA 5: Teste final...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, shuffle=False
        )
        
        # Usar Random Forest (geralmente melhor para dados financeiros)
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(n_estimators=50, random_state=42, max_depth=5))
        ])
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        acuracia_final = accuracy_score(y_test, y_pred)
        
        print(f"✓ Acurácia final: {acuracia_final:.1%}")
        
        # Baseline comparison
        baseline = max(y.mean(), 1 - y.mean())
        print(f"✓ Baseline: {baseline:.1%}")
        print(f"✓ Melhoria: {(acuracia_final - baseline)*100:+.1f} pontos")
        
        # Feature importance
        if hasattr(pipeline.named_steps['classifier'], 'feature_importances_'):
            importances = pipeline.named_steps['classifier'].feature_importances_
            feature_importance = pd.DataFrame({
                'feature': features_lista,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            print(f"\n📊 Features mais importantes:")
            for _, row in feature_importance.head(5).iterrows():
                print(f"  {row['feature']}: {row['importance']:.3f}")
        
        print("\n✓ PROJETO MELHORADO CONCLUÍDO!")
        
        return {
            'acuracia_final': acuracia_final,
            'baseline': baseline,
            'melhoria': acuracia_final - baseline
        }
        
    except Exception as e:
        print(f"Erro: {e}")
        return None

if __name__ == "__main__":
    resultados = main_melhorado()
    
    if resultados and resultados['melhoria'] > 0:
        print(f"\n🎉 SUCESSO! Modelo melhorado em {resultados['melhoria']*100:.1f} pontos!")
    elif resultados:
        print(f"\n📊 Modelo ainda abaixo do baseline. Mercado muito eficiente.")
    else:
        print(f"\n❌ Falha na execução.")
