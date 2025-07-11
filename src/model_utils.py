"""
Utilitários para criação e avaliação de modelos
"""

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import VotingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, make_scorer

def criar_modelo_otimizado():
    """
    Cria modelo ensemble otimizado
    
    Returns:
        Pipeline: Pipeline com modelo ensemble
    """
    print("🤖 Criando modelo ensemble otimizado...")
    
    # Ensemble de modelos de regressão
    modelos = [
        ('linear', LinearRegression()),
        ('ridge', Ridge(alpha=1.0, random_state=42)),
        ('lasso', Lasso(alpha=0.01, random_state=42, max_iter=2000)),
        ('elastic', ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42, max_iter=2000))
    ]
    
    ensemble = VotingRegressor(estimators=modelos)
    
    # Pipeline completo
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', ensemble)
    ])
    
    print("✅ Modelo ensemble criado com 4 algoritmos")
    
    return pipeline

def acuracia_direcao_scorer(y_true, y_pred):
    """
    Scorer personalizado para calcular acurácia de direção
    """
    direcao_real = (y_true > 0).astype(int)
    direcao_pred = (y_pred > 0).astype(int)
    return np.mean(direcao_real == direcao_pred)

def validacao_temporal(modelo, X_treino, y_treino):
    """
    Realiza validação cruzada temporal
    
    Args:
        modelo: Modelo a validar
        X_treino: Features de treino
        y_treino: Target de treino
        
    Returns:
        tuple: (acuracia_media, r2_medio)
    """
    print("⏰ Validação cruzada temporal...")
    
    # TimeSeriesSplit para dados temporais
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Scorer personalizado
    scorer_acuracia = make_scorer(acuracia_direcao_scorer)
    
    # Validação cruzada
    scores_acuracia = cross_val_score(modelo, X_treino, y_treino, cv=tscv, scoring=scorer_acuracia)
    scores_r2 = cross_val_score(modelo, X_treino, y_treino, cv=tscv, scoring='r2')
    
    acuracia_media = scores_acuracia.mean()
    r2_medio = scores_r2.mean()
    
    print(f"   📊 Acurácia CV: {acuracia_media:.1%} (±{scores_acuracia.std():.1%})")
    print(f"   📈 R² CV: {r2_medio:.4f}")
    
    return acuracia_media, r2_medio

def avaliar_modelo(modelo, X_treino, X_teste, y_treino, y_teste, y_treino_dir, y_teste_dir):
    """
    Avalia modelo completo
    
    Args:
        modelo: Modelo a avaliar
        X_treino, X_teste: Features
        y_treino, y_teste: Targets de regressão
        y_treino_dir, y_teste_dir: Targets de direção
        
    Returns:
        dict: Resultados da avaliação
    """
    print("📈 Avaliando modelo...")
    
    # Validação cruzada
    acuracia_cv, r2_cv = validacao_temporal(modelo, X_treino, y_treino)
    
    # Treinar modelo final
    modelo.fit(X_treino, y_treino)
    
    # Previsões no teste
    y_pred_returns = modelo.predict(X_teste)
    y_pred_direction = (y_pred_returns > 0).astype(int)
    
    # Métricas de regressão
    mse_teste = mean_squared_error(y_teste, y_pred_returns)
    r2_teste = r2_score(y_teste, y_pred_returns)
    
    # Métrica de classificação (direção)
    acuracia_teste = accuracy_score(y_teste_dir, y_pred_direction)
    
    # Baseline
    baseline = max(y_teste_dir.mean(), 1 - y_teste_dir.mean())
    
    print(f"✅ Avaliação concluída:")
    print(f"   Acurácia teste: {acuracia_teste:.1%}")
    print(f"   R² teste: {r2_teste:.4f}")
    print(f"   Baseline: {baseline:.1%}")
    
    return {
        'acuracia_cv': acuracia_cv,
        'r2_cv': r2_cv,
        'acuracia_teste': acuracia_teste,
        'r2_teste': r2_teste,
        'mse_teste': mse_teste,
        'baseline': baseline,
        'predicoes_retorno': y_pred_returns,
        'predicoes_direcao': y_pred_direction
    }
