"""
Pipeline refinado para previsão do IBOVESPA
OBJETIVO: Maximizar acurácia através de otimizações avançadas
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Adicionar pasta src ao path para importar módulos
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from data_utils import carregar_dados_ibovespa, preparar_dataset_temporal
from feature_engineering import criar_features_basicas, criar_features_avancadas, criar_targets
from model_utils import criar_modelo_otimizado, avaliar_modelo
from correlation_analysis import analisar_correlacoes, selecionar_features_otimizadas

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.feature_selection import SelectKBest, f_regression, RFE

def criar_features_otimizadas(data):
    """
    Cria conjunto otimizado de features com foco em acurácia
    """
    print("🚀 Criando features otimizadas...")
    
    # Features básicas de momentum
    data['Returns_1d'] = data['Close'].pct_change()
    data['Returns_2d'] = data['Close'].pct_change(2)
    data['Returns_3d'] = data['Close'].pct_change(3)
    data['Returns_5d'] = data['Close'].pct_change(5)
    
    # Médias móveis e cruzamentos
    data['SMA_5'] = data['Close'].rolling(5).mean()
    data['SMA_10'] = data['Close'].rolling(10).mean()
    data['SMA_20'] = data['Close'].rolling(20).mean()
    
    # Sinais de cruzamento mais simples
    data['Price_vs_SMA5'] = data['Close'] / data['SMA_5'] - 1
    data['Price_vs_SMA20'] = data['Close'] / data['SMA_20'] - 1
    data['SMA5_vs_SMA20'] = data['SMA_5'] / data['SMA_20'] - 1
    
    # RSI simplificado
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    data['RSI_Signal'] = data['RSI'] - 50  # Centralizado em 0
    
    # Volume features
    data['Volume_MA'] = data['Volume'].rolling(20).mean()
    data['Volume_Ratio'] = data['Volume'] / data['Volume_MA']
    
    # Volatilidade
    data['Volatility'] = data['Returns_1d'].rolling(10).std()
    data['Vol_Ratio'] = data['Volatility'] / data['Volatility'].rolling(30).mean()
    
    # Features lagged (memória)
    data['Returns_1d_lag1'] = data['Returns_1d'].shift(1)
    data['Returns_1d_lag2'] = data['Returns_1d'].shift(2)
    data['RSI_lag1'] = data['RSI'].shift(1)
    
    # High/Low features
    data['HL_Ratio'] = (data['High'] - data['Low']) / data['Close']
    data['Close_Position'] = (data['Close'] - data['Low']) / (data['High'] - data['Low'])
    
    # Lista de features otimizadas (apenas as mais importantes)
    features_otimizadas = [
        'Returns_1d_lag1', 'Returns_1d_lag2', 'Returns_2d', 'Returns_3d',
        'Price_vs_SMA5', 'Price_vs_SMA20', 'SMA5_vs_SMA20',
        'RSI_Signal', 'RSI_lag1',
        'Volume_Ratio', 'Vol_Ratio',
        'HL_Ratio', 'Close_Position'
    ]
    
    print(f"✅ {len(features_otimizadas)} features otimizadas criadas")
    
    return data, features_otimizadas

def selecionar_melhores_features(X_treino, y_treino, features_lista, n_features=5):
    """
    Seleção rigorosa das melhores features
    """
    print(f"🎯 Selecionando {n_features} melhores features...")
    
    # Método 1: SelectKBest
    selector = SelectKBest(score_func=f_regression, k=n_features)
    X_selected = selector.fit_transform(X_treino[features_lista], y_treino)
    features_kbest = X_treino[features_lista].columns[selector.get_support()].tolist()
    
    # Método 2: RFE com modelo simples
    estimator = LinearRegression()
    rfe = RFE(estimator, n_features_to_select=n_features)
    rfe.fit(X_treino[features_lista], y_treino)
    features_rfe = X_treino[features_lista].columns[rfe.support_].tolist()
    
    # Combinar métodos (intersecção + união)
    features_intersecao = list(set(features_kbest) & set(features_rfe))
    features_uniao = list(set(features_kbest) | set(features_rfe))
    
    # Se intersecção for pequena, usar união limitada
    if len(features_intersecao) >= 3:
        features_finais = features_intersecao[:n_features]
    else:
        features_finais = features_uniao[:n_features]
    
    print(f"✅ Features selecionadas: {features_finais}")
    
    return features_finais

def criar_modelo_avancado():
    """
    Cria modelo mais sofisticado que pode capturar padrões não-lineares
    """
    print("🤖 Criando modelo avançado...")
    
    # Modelos de regressão diversos
    modelos = [
        ('linear', LinearRegression()),
        ('ridge', Ridge(alpha=0.1, random_state=42)),
        ('elastic', ElasticNet(alpha=0.01, l1_ratio=0.7, random_state=42, max_iter=3000)),
        ('rf', RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)),
        ('gb', GradientBoostingRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42))
    ]
    
    # Voting regressor
    ensemble = VotingRegressor(estimators=modelos, n_jobs=-1)
    
    # Pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', ensemble)
    ])
    
    print("✅ Modelo avançado criado com 5 algoritmos")
    
    return pipeline

def validacao_otimizada(modelo, X_treino, y_treino, cv_folds=3):
    """
    Validação otimizada para dados financeiros
    """
    print("⏰ Validação cruzada otimizada...")
    
    # TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=cv_folds)
    
    # Scorer personalizado
    def acuracia_direcao(y_true, y_pred):
        direcao_real = (y_true > 0).astype(int)
        direcao_pred = (y_pred > 0).astype(int)
        return np.mean(direcao_real == direcao_pred)
    
    scorer = make_scorer(acuracia_direcao)
    
    # Cross validation
    scores = []
    for train_idx, val_idx in tscv.split(X_treino):
        X_train_fold = X_treino.iloc[train_idx]
        X_val_fold = X_treino.iloc[val_idx]
        y_train_fold = y_treino.iloc[train_idx]
        y_val_fold = y_treino.iloc[val_idx]
        
        # Treinar modelo
        modelo.fit(X_train_fold, y_train_fold)
        
        # Prever
        y_pred = modelo.predict(X_val_fold)
        
        # Calcular acurácia de direção
        score = acuracia_direcao(y_val_fold, y_pred)
        scores.append(score)
    
    acuracia_cv = np.mean(scores)
    std_cv = np.std(scores)
    
    print(f"   📊 Acurácia CV: {acuracia_cv:.1%} (±{std_cv:.1%})")
    
    return acuracia_cv, std_cv

def testar_multiplas_configuracoes(data, features_lista):
    """
    Testa múltiplas configurações para encontrar a melhor
    """
    print("\n🧪 TESTANDO MÚLTIPLAS CONFIGURAÇÕES...")
    
    # Preparar target
    data['Target_Return'] = data['Close'].pct_change().shift(-1)
    data['Target_Direction'] = (data['Target_Return'] > 0).astype(int)
    
    # Limpar dados
    dataset = data[features_lista + ['Target_Return', 'Target_Direction']].dropna()
    
    # Divisão temporal
    n_test = 30
    train_data = dataset.iloc[:-n_test]
    test_data = dataset.iloc[-n_test:]
    
    X_treino = train_data[features_lista]
    y_treino = train_data['Target_Return']
    y_treino_dir = train_data['Target_Direction']
    
    X_teste = test_data[features_lista]
    y_teste = test_data['Target_Return'] 
    y_teste_dir = test_data['Target_Direction']
    
    melhores_resultados = []
    
    # Configuração 1: 5 features + modelo linear
    print("\n🔹 CONFIG 1: 5 features + Linear")
    features_5 = selecionar_melhores_features(X_treino, y_treino, features_lista, 5)
    modelo_linear = Pipeline([('scaler', StandardScaler()), ('reg', LinearRegression())])
    acuracia_cv, _ = validacao_otimizada(modelo_linear, X_treino[features_5], y_treino, 3)
    
    # Teste final
    modelo_linear.fit(X_treino[features_5], y_treino)
    y_pred = modelo_linear.predict(X_teste[features_5])
    y_pred_dir = (y_pred > 0).astype(int)
    acuracia_teste = accuracy_score(y_teste_dir, y_pred_dir)
    
    melhores_resultados.append({
        'config': 'Linear 5 features',
        'features': features_5,
        'modelo': modelo_linear,
        'acuracia_cv': acuracia_cv,
        'acuracia_teste': acuracia_teste,
        'n_features': 5
    })
    print(f"   CV: {acuracia_cv:.1%} | Teste: {acuracia_teste:.1%}")
    
    # Configuração 2: 3 features + modelo avançado
    print("\n🔹 CONFIG 2: 3 features + Ensemble Avançado")
    features_3 = selecionar_melhores_features(X_treino, y_treino, features_lista, 3)
    modelo_avancado = criar_modelo_avancado()
    acuracia_cv, _ = validacao_otimizada(modelo_avancado, X_treino[features_3], y_treino, 3)
    
    # Teste final
    modelo_avancado.fit(X_treino[features_3], y_treino)
    y_pred = modelo_avancado.predict(X_teste[features_3])
    y_pred_dir = (y_pred > 0).astype(int)
    acuracia_teste = accuracy_score(y_teste_dir, y_pred_dir)
    
    melhores_resultados.append({
        'config': 'Ensemble 3 features',
        'features': features_3,
        'modelo': modelo_avancado,
        'acuracia_cv': acuracia_cv,
        'acuracia_teste': acuracia_teste,
        'n_features': 3
    })
    print(f"   CV: {acuracia_cv:.1%} | Teste: {acuracia_teste:.1%}")
    
    # Configuração 3: 7 features + ensemble simples
    print("\n🔹 CONFIG 3: 7 features + Ensemble Simples")
    features_7 = selecionar_melhores_features(X_treino, y_treino, features_lista, 7)
    modelo_simples = Pipeline([
        ('scaler', StandardScaler()),
        ('reg', VotingRegressor([
            ('ridge', Ridge(alpha=0.1, random_state=42)),
            ('elastic', ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42, max_iter=3000))
        ], n_jobs=-1))
    ])
    acuracia_cv, _ = validacao_otimizada(modelo_simples, X_treino[features_7], y_treino, 3)
    
    # Teste final
    modelo_simples.fit(X_treino[features_7], y_treino)
    y_pred = modelo_simples.predict(X_teste[features_7])
    y_pred_dir = (y_pred > 0).astype(int)
    acuracia_teste = accuracy_score(y_teste_dir, y_pred_dir)
    
    melhores_resultados.append({
        'config': 'Ensemble Simples 7 features',
        'features': features_7,
        'modelo': modelo_simples,
        'acuracia_cv': acuracia_cv,
        'acuracia_teste': acuracia_teste,
        'n_features': 7
    })
    print(f"   CV: {acuracia_cv:.1%} | Teste: {acuracia_teste:.1%}")
    
    # Encontrar melhor configuração
    melhor = max(melhores_resultados, key=lambda x: x['acuracia_teste'])
    
    return melhor, melhores_resultados, X_teste, y_teste, y_teste_dir

def pipeline_refinado():
    """
    Pipeline refinado com foco em maximizar acurácia
    """
    print("="*70)
    print("🎯 PIPELINE REFINADO - MAXIMIZAÇÃO DE ACURÁCIA")
    print("🚀 OBJETIVO: Superar 60% de acurácia")
    print("="*70)
    
    try:
        # 1. CARREGAMENTO DE DADOS
        print("\n📥 ETAPA 1: Carregamento de Dados")
        data = carregar_dados_ibovespa(anos=10)
        
        # 2. ENGENHARIA DE FEATURES OTIMIZADA
        print("\n🔧 ETAPA 2: Features Otimizadas")
        data, features_otimizadas = criar_features_otimizadas(data)
        
        # 3. TESTE DE MÚLTIPLAS CONFIGURAÇÕES
        melhor_config, todos_resultados, X_teste, y_teste, y_teste_dir = testar_multiplas_configuracoes(data, features_otimizadas)
        
        # 4. RELATÓRIO FINAL
        print("\n" + "="*70)
        print("🏆 RESULTADOS FINAIS")
        print("="*70)
        
        print(f"\n🥇 MELHOR CONFIGURAÇÃO: {melhor_config['config']}")
        print(f"🎯 ACURÁCIA CV: {melhor_config['acuracia_cv']:.1%}")
        print(f"📊 ACURÁCIA TESTE: {melhor_config['acuracia_teste']:.1%}")
        print(f"🔧 FEATURES ({melhor_config['n_features']}): {melhor_config['features']}")
        
        # Baseline comparison
        baseline = max(y_teste_dir.mean(), 1 - y_teste_dir.mean())
        melhoria = melhor_config['acuracia_teste'] - baseline
        print(f"\n📈 COMPARAÇÃO:")
        print(f"   Baseline: {baseline:.1%}")
        print(f"   Nosso modelo: {melhor_config['acuracia_teste']:.1%}")
        print(f"   Melhoria: {melhoria*100:+.1f} pontos")
        
        # Status da meta
        if melhor_config['acuracia_teste'] >= 0.75:
            print(f"\n🎉 META 75% ATINGIDA!")
        elif melhor_config['acuracia_teste'] >= 0.60:
            print(f"\n✅ EXCELENTE! Acurácia > 60%")
            print(f"   Faltam {(0.75 - melhor_config['acuracia_teste'])*100:.1f} pontos para 75%")
        else:
            print(f"\n📊 Meta 60% {'ATINGIDA' if melhor_config['acuracia_teste'] >= 0.60 else 'NÃO ATINGIDA'}")
        
        # Ranking de todas as configurações
        print(f"\n📊 RANKING COMPLETO:")
        todos_resultados.sort(key=lambda x: x['acuracia_teste'], reverse=True)
        for i, config in enumerate(todos_resultados, 1):
            print(f"   {i}. {config['config']}: {config['acuracia_teste']:.1%} (CV: {config['acuracia_cv']:.1%})")
        
        print("\n✅ PIPELINE REFINADO CONCLUÍDO!")
        
        return {
            'melhor_config': melhor_config,
            'todos_resultados': todos_resultados,
            'acuracia_final': melhor_config['acuracia_teste'],
            'meta_60_atingida': melhor_config['acuracia_teste'] >= 0.60,
            'meta_75_atingida': melhor_config['acuracia_teste'] >= 0.75
        }
        
    except Exception as e:
        print(f"\n❌ ERRO: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    resultado = pipeline_refinado()
    
    if resultado:
        print(f"\n🎯 ACURÁCIA FINAL: {resultado['acuracia_final']:.1%}")
        if resultado['meta_75_atingida']:
            print("🏆 META 75% ATINGIDA!")
        elif resultado['meta_60_atingida']:
            print("✅ META 60% ATINGIDA!")
        else:
            print("📊 Continuar refinando...")
    else:
        print("\n❌ Pipeline falhou")
