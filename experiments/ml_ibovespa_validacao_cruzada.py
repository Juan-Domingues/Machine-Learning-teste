"""
Machine Learning - Previsão IBOVESPA
Objetivo: Usar REGRESSÃO para prever retorno do próximo dia e converter em direção
Meta: Acurácia mínima de 75% no conjunto de teste (últimos 30 dias)
Abordagem: Regressão Linear + conversão para classificação binária
"""

# IMPORTS NECESSÁRIOS
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Scikit-Learn
from sklearn.model_selection import train_test_split, cross_val_score, KFold, TimeSeriesSplit
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, make_scorer, accuracy_score
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.ensemble import VotingRegressor
import matplotlib.pyplot as plt
import seaborn as sns

def carregar_dados_ibovespa(anos=10):
    """
    ETAPA 1: CARREGAMENTO DE DADOS
    Baixa dados históricos do IBOVESPA via Yahoo Finance
    """
    print("ETAPA 1: Carregando dados do IBOVESPA...")
    
    # Definir período
    end_date = datetime.now()
    start_date = end_date - timedelta(days=anos * 365)
    
    # Baixar dados
    data = yf.download('^BVSP', start=start_date, end=end_date)
    
    # Limpar colunas se necessário
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    print(f"✓ {len(data)} dias carregados ({anos} anos)")
    
    return data

def criar_features(data):
    """
    ETAPA 2: CRIAÇÃO DE FEATURES
    Cria indicadores técnicos para o modelo
    """
    print("ETAPA 2: Criando features técnicas...")
    
    # Calcular retornos
    data['Retorno'] = data['Close'].pct_change()
    
    # Médias móveis
    data['MM5'] = data['Close'].rolling(5).mean()
    data['MM20'] = data['Close'].rolling(20).mean()
    data['MM50'] = data['Close'].rolling(50).mean()
    
    # RSI (Índice de Força Relativa)
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # Volatilidade
    data['Volatilidade'] = data['Retorno'].rolling(20).std()
    
    # Volume normalizado
    data['Volume_Norm'] = data['Volume'] / data['Volume'].rolling(20).mean()
    
    # Posição no canal
    data['Max_20'] = data['High'].rolling(20).max()
    data['Min_20'] = data['Low'].rolling(20).min()
    data['Posicao_Canal'] = (data['Close'] - data['Min_20']) / (data['Max_20'] - data['Min_20'])
    
    # Indicadores de momentum
    data['Momentum_3'] = data['Close'] / data['Close'].shift(3) - 1
    data['Momentum_7'] = data['Close'] / data['Close'].shift(7) - 1
    data['Retorno_Acum_3'] = data['Retorno'].rolling(3).sum()
    
    # Sinais de cruzamento
    data['MM_Cross'] = (data['MM5'] > data['MM20']).astype(int)
    data['Price_Above_MA20'] = (data['Close'] > data['MM20']).astype(int)
    
    # Sinais de RSI
    data['RSI_Signal'] = np.where(data['RSI'] < 30, 1, np.where(data['RSI'] > 70, -1, 0))
    
    # Retornos de diferentes períodos
    data['Returns_3d'] = data['Close'].pct_change(3)
    data['Returns_7d'] = data['Close'].pct_change(7)
    
    # Análise de volume
    data['Volume_Spike'] = (data['Volume'] > 1.5 * data['Volume'].rolling(20).mean()).astype(int)
    
    # Regime de volatilidade
    data['Low_Vol'] = (data['Volatilidade'] < data['Volatilidade'].rolling(50).quantile(0.3)).astype(int)
    
    # Posição no canal de preços
    data['Max_20'] = data['High'].rolling(20).max()
    data['Min_20'] = data['Low'].rolling(20).min()
    data['Channel_Position'] = (data['Close'] - data['Min_20']) / (data['Max_20'] - data['Min_20'])
    
    features_lista = [
        'MM5', 'MM20', 'RSI', 'Volatilidade',
        'MM_Cross', 'Price_Above_MA20', 'RSI_Signal',
        'Returns_3d', 'Returns_7d', 'Volume_Spike', 'Low_Vol', 'Channel_Position'
    ]
    
    print(f"✓ {len(features_lista)} features criadas")
    
    return data, features_lista

def criar_target(data):
    """
    ETAPA 3: CRIAÇÃO DO TARGET PARA REGRESSÃO
    Target: retorno percentual do próximo dia (para regressão)
    """
    print("ETAPA 3: Criando target para regressão...")
    
    # Target: retorno do próximo dia (em percentual)
    data['Target_Return'] = data['Close'].pct_change().shift(-1)
    
    # Para análise também criar a direção (mas não usar no treino)
    data['Target_Direction'] = (data['Target_Return'] > 0).astype(int)
    
    # Estatísticas básicas
    returns_stats = data['Target_Return'].describe()
    direction_counts = data['Target_Direction'].value_counts()
    total = len(data['Target_Direction'].dropna())
    
    print(f"✓ Target de REGRESSÃO criado:")
    print(f"  Retorno médio: {returns_stats['mean']:.4f}")
    print(f"  Desvio padrão: {returns_stats['std']:.4f}")
    print(f"  Min: {returns_stats['min']:.4f} | Max: {returns_stats['max']:.4f}")
    print(f"  Dias de alta: {direction_counts.get(1, 0)} ({direction_counts.get(1, 0)/total:.1%})")
    print(f"  Dias de baixa: {direction_counts.get(0, 0)} ({direction_counts.get(0, 0)/total:.1%})")
    
    return data

def preparar_dataset(data, features_lista):
    """
    ETAPA 4: PREPARAÇÃO DO DATASET PARA REGRESSÃO
    Limpa os dados e separa features do target - Últimos 30 dias para teste
    """
    print("ETAPA 4: Preparando dataset para regressão...")
    
    # Criar dataset final usando target de regressão
    dataset = data[features_lista + ['Target_Return', 'Target_Direction']].copy()
    
    # Remover valores nulos
    dataset = dataset.dropna()
    
    # DIVISÃO ESPECÍFICA: Últimos 30 dias para teste, resto para treino
    n_test_days = 30
    split_index = len(dataset) - n_test_days
    
    # Separar dados de treino e teste
    data_treino = dataset.iloc[:split_index]
    data_teste = dataset.iloc[split_index:]
    
    # Separar X (features) e y (target de regressão)
    X_treino = data_treino[features_lista]
    y_treino = data_treino['Target_Return']  # Target de regressão
    X_teste = data_teste[features_lista]
    y_teste = data_teste['Target_Return']     # Target de regressão
    
    # Também separar as direções reais para avaliar acurácia
    y_treino_dir = data_treino['Target_Direction']
    y_teste_dir = data_teste['Target_Direction']
    
    print(f"✓ Dataset preparado para regressão:")
    print(f"  Total de observações: {len(dataset)}")
    print(f"  Dados de treino: {len(X_treino)} observações")
    print(f"  Dados de teste (últimos 30 dias): {len(X_teste)} observações")
    print(f"  Features: {len(features_lista)}")
    
    return X_treino, X_teste, y_treino, y_teste, y_treino_dir, y_teste_dir

def validacao_cruzada_regressao(X_treino, y_treino, y_treino_dir):
    """
    ETAPA 5: VALIDAÇÃO CRUZADA COM REGRESSÃO
    Testa múltiplos modelos de regressão e converte previsões em direção
    """
    print("ETAPA 5: Validação cruzada com modelos de REGRESSÃO...")
    
    # Modelos de regressão para testar
    modelos = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0, random_state=42),
        'Lasso Regression': Lasso(alpha=0.1, random_state=42),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
    }
    
    # Função personalizada para calcular acurácia de direção a partir de regressão
    def acuracia_direcao_regressao(y_true, y_pred):
        """Converte previsões de regressão em direção e calcula acurácia"""
        direcao_real = (y_true > 0).astype(int)
        direcao_pred = (y_pred > 0).astype(int)
        return np.mean(direcao_real == direcao_pred)
    
    # Criar scorer personalizado
    scorer_acuracia = make_scorer(acuracia_direcao_regressao)
    
    resultados = {}
    
    # Testar cada modelo
    for nome, modelo_base in modelos.items():
        # Criar pipeline com normalização
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', modelo_base)
        ])
        
        # Validação cruzada (5 folds) sem shuffle para manter ordem temporal
        cv = KFold(n_splits=5, shuffle=False)
        
        # Calcular acurácia de direção usando regressão
        scores_acuracia = cross_val_score(
            pipeline, X_treino, y_treino, cv=cv, scoring=scorer_acuracia, n_jobs=-1
        )
        
        # Também calcular R² para avaliar qualidade da regressão
        scores_r2 = cross_val_score(
            pipeline, X_treino, y_treino, cv=cv, scoring='r2', n_jobs=-1
        )
        
        # Armazenar resultados
        resultados[nome] = {
            'acuracia_mean': scores_acuracia.mean(),
            'acuracia_std': scores_acuracia.std(),
            'r2_mean': scores_r2.mean(),
            'r2_std': scores_r2.std(),
            'scores': scores_acuracia
        }
        
        # Mostrar resultados
        status = "✅" if scores_acuracia.mean() >= 0.75 else "⚠️" if scores_acuracia.mean() >= 0.60 else "❌"
        print(f"  {status} {nome}:")
        print(f"     Acurácia direção: {scores_acuracia.mean():.1%} (±{scores_acuracia.std():.1%})")
        print(f"     R² regressão: {scores_r2.mean():.4f} (±{scores_r2.std():.4f})")
    
    # Encontrar melhor modelo
    melhor_modelo_nome = max(resultados.keys(), key=lambda k: resultados[k]['acuracia_mean'])
    melhor_score = resultados[melhor_modelo_nome]['acuracia_mean']
    
    print(f"\n🏆 Melhor modelo: {melhor_modelo_nome}")
    print(f"   Acurácia: {melhor_score:.1%}")
    print(f"   R²: {resultados[melhor_modelo_nome]['r2_mean']:.4f}")
    
    if melhor_score >= 0.75:
        print("✅ META ATINGIDA: Acurácia >= 75% na validação cruzada!")
    else:
        print(f"⚠️ META NÃO ATINGIDA: Acurácia {melhor_score:.1%} < 75%")
        print(f"   Faltam {(0.75 - melhor_score)*100:.1f} pontos percentuais")
    
    return resultados, melhor_modelo_nome


def teste_final_30_dias(X_treino, X_teste, y_treino, y_teste, y_teste_dir, modelo_nome):
    """
    ETAPA 6: TESTE FINAL NOS ÚLTIMOS 30 DIAS COM REGRESSÃO
    Avalia o melhor modelo de regressão nos últimos 30 dias
    """
    print("ETAPA 6: Teste final nos últimos 30 dias com REGRESSÃO...")
    
    # Mapear nome do modelo para classe (REGRESSÃO)
    if modelo_nome == "Ensemble":
        modelo = criar_ensemble_robusto()
    else:
        modelos_map = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0, random_state=42),
            'Lasso Regression': Lasso(alpha=0.1, random_state=42),
            'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
        }
        modelo = modelos_map.get(modelo_nome, LinearRegression())
    
    # Criar pipeline com o modelo
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', modelo)
    ])
    
    # Treinar no conjunto de treino (usando retornos)
    pipeline.fit(X_treino, y_treino)
    
    # Fazer previsões no conjunto de teste (últimos 30 dias)
    y_pred_returns = pipeline.predict(X_teste)
    
    # Converter previsões de regressão para direção
    y_pred_direction = (y_pred_returns > 0).astype(int)
    
    # Calcular acurácia usando as direções
    acuracia_final = accuracy_score(y_teste_dir, y_pred_direction)
    
    print(f"✓ Modelo usado: {modelo_nome}")
    print(f"✓ Acurácia nos últimos 30 dias: {acuracia_final:.1%}")
    
    # Verificar se atingiu a meta
    if acuracia_final >= 0.75:
        print("🎉 META ATINGIDA: Acurácia >= 75% no conjunto de teste!")
        status = "SUCESSO"
    else:
        print("❌ META NÃO ATINGIDA: Acurácia < 75% no conjunto de teste")
        status = "FALHA"
    
    # Relatório detalhado
    print(f"\n📊 Relatório detalhado do teste:")
    print(f"  Previsões corretas: {(y_pred_direction == y_teste_dir).sum()}/{len(y_teste_dir)}")
    print(f"  Baseline (sempre classe majoritária): {max(y_teste_dir.mean(), 1-y_teste_dir.mean()):.1%}")
    
    # Distribuição das previsões
    pred_counts = pd.Series(y_pred_direction).value_counts()
    real_counts = y_teste_dir.value_counts()
    print(f"  Previsões ALTA: {pred_counts.get(1, 0)} | Real ALTA: {real_counts.get(1, 0)}")
    print(f"  Previsões BAIXA: {pred_counts.get(0, 0)} | Real BAIXA: {real_counts.get(0, 0)}")
    
    # Estatísticas da regressão
    mse = mean_squared_error(y_teste, y_pred_returns)
    r2 = r2_score(y_teste, y_pred_returns)
    print(f"\n📈 Qualidade da regressão:")
    print(f"  MSE: {mse:.6f}")
    print(f"  R²: {r2:.4f}")
    print(f"  Retorno real médio: {y_teste.mean():.4f}")
    print(f"  Retorno previsto médio: {y_pred_returns.mean():.4f}")
    
    return {
        'acuracia': acuracia_final,
        'status': status,
        'modelo': modelo_nome,
        'predicoes_retorno': y_pred_returns,
        'predicoes_direcao': y_pred_direction,
        'real_retorno': y_teste,
        'real_direcao': y_teste_dir,
        'mse': mse,
        'r2': r2
    }


def analisar_correlacoes_features(data, features_lista, target_col='Target_Return'):
    """
    ANÁLISE DETALHADA DE CORRELAÇÕES
    Analisa correlações entre features e target para identificar as mais relevantes
    """
    print("="*60)
    print("ANÁLISE DETALHADA DE CORRELAÇÕES")
    print("="*60)
    
    # Criar dataset completo para análise
    dataset = data[features_lista + [target_col, 'Target_Direction']].copy()
    dataset = dataset.dropna()
    
    print(f"📊 Analisando {len(features_lista)} features com {len(dataset)} observações")
    
    # 1. CORRELAÇÃO COM TARGET DE REGRESSÃO
    print(f"\n🎯 CORRELAÇÃO COM RETORNO (Target de Regressão):")
    correlacoes_retorno = dataset[features_lista].corrwith(dataset[target_col]).abs().sort_values(ascending=False)
    print("Top 10 features mais correlacionadas com o retorno:")
    for i, (feature, corr) in enumerate(correlacoes_retorno.head(10).items(), 1):
        status = "🟢" if corr > 0.05 else "🟡" if corr > 0.02 else "🔴"
        print(f"  {i:2d}. {status} {feature:<20}: {corr:.4f}")
    
    # 2. CORRELAÇÃO COM TARGET DE DIREÇÃO
    print(f"\n🎯 CORRELAÇÃO COM DIREÇÃO (Target de Classificação):")
    correlacoes_direcao = dataset[features_lista].corrwith(dataset['Target_Direction']).abs().sort_values(ascending=False)
    print("Top 10 features mais correlacionadas com a direção:")
    for i, (feature, corr) in enumerate(correlacoes_direcao.head(10).items(), 1):
        status = "🟢" if corr > 0.05 else "🟡" if corr > 0.02 else "🔴"
        print(f"  {i:2d}. {status} {feature:<20}: {corr:.4f}")
    
    # 3. MATRIZ DE CORRELAÇÃO ENTRE FEATURES
    print(f"\n🔗 CORRELAÇÃO ENTRE FEATURES (detectar multicolinearidade):")
    corr_matrix = dataset[features_lista].corr().abs()
    
    # Encontrar pares altamente correlacionados
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > 0.8:
                high_corr_pairs.append((
                    corr_matrix.columns[i], 
                    corr_matrix.columns[j], 
                    corr_matrix.iloc[i, j]
                ))
    
    if high_corr_pairs:
        print("⚠️ Features altamente correlacionadas (>0.8) - possível multicolinearidade:")
        for feat1, feat2, corr in high_corr_pairs:
            print(f"   {feat1} <-> {feat2}: {corr:.3f}")
    else:
        print("✅ Não há multicolinearidade significativa entre features")
    
    # 4. SELEÇÃO DE MELHORES FEATURES
    print(f"\n🎯 SELEÇÃO DE FEATURES BASEADA EM CORRELAÇÃO:")
    
    # Combinar correlações (média ponderada)
    correlacao_combinada = (correlacoes_retorno * 0.6 + correlacoes_direcao * 0.4).sort_values(ascending=False)
    
    # Selecionar top features
    top_5_features = correlacao_combinada.head(5).index.tolist()
    top_8_features = correlacao_combinada.head(8).index.tolist()
    top_all_features = correlacao_combinada.index.tolist()
    
    print(f"🥇 TOP 5 Features (correlação combinada):")
    for i, feature in enumerate(top_5_features, 1):
        corr_comb = correlacao_combinada[feature]
        corr_ret = correlacoes_retorno[feature]
        corr_dir = correlacoes_direcao[feature]
        print(f"  {i}. {feature:<20}: {corr_comb:.4f} (ret:{corr_ret:.4f}, dir:{corr_dir:.4f})")
    
    print(f"\n🥈 TOP 8 Features (correlação combinada):")
    for i, feature in enumerate(top_8_features, 1):
        corr_comb = correlacao_combinada[feature]
        print(f"  {i}. {feature:<20}: {corr_comb:.4f}")
    
    # 5. ESTATÍSTICAS DESCRITIVAS DAS MELHORES FEATURES
    print(f"\n📈 ESTATÍSTICAS DAS TOP 5 FEATURES:")
    top_5_stats = dataset[top_5_features].describe()
    print(top_5_stats.round(4))
    
    return {
        'correlacoes_retorno': correlacoes_retorno,
        'correlacoes_direcao': correlacoes_direcao,
        'correlacao_combinada': correlacao_combinada,
        'top_5_features': top_5_features,
        'top_8_features': top_8_features,
        'high_corr_pairs': high_corr_pairs,
        'corr_matrix': corr_matrix,
        'dataset_analise': dataset
    }

def criar_features_avancadas(data):
    """
    CRIAÇÃO DE FEATURES AVANÇADAS
    Baseado na análise de correlação, criar features mais sofisticadas
    """
    print("\n" + "="*60)
    print("CRIAÇÃO DE FEATURES AVANÇADAS")
    print("="*60)
    
    # Features originais já criadas
    data, features_originais = criar_features(data)
    
    print("🚀 Criando features avançadas baseadas em análise de correlação...")
    
    # 1. FEATURES DE MOMENTUM AVANÇADAS
    data['Momentum_Ratio'] = data['MM5'] / data['MM20']  # Razão entre médias
    data['Price_Momentum'] = data['Close'] / data['MM50']  # Posição relativa à MM50
    data['Volume_Price_Momentum'] = data['Volume_Norm'] * data['Retorno']  # Volume ponderado pelo retorno
    
    # 2. FEATURES DE VOLATILIDADE AVANÇADAS
    data['Volatilidade_Relativa'] = data['Volatilidade'] / data['Volatilidade'].rolling(50).mean()
    data['Volatilidade_Tendencia'] = data['Volatilidade'].rolling(5).mean() / data['Volatilidade'].rolling(20).mean()
    
    # 3. FEATURES DE RSI AVANÇADAS
    data['RSI_Momentum'] = data['RSI'].diff()  # Momentum do RSI
    data['RSI_Volatilidade'] = data['RSI'].rolling(10).std()  # Volatilidade do RSI
    data['RSI_Normalized'] = (data['RSI'] - 50) / 50  # RSI normalizado entre -1 e 1
    
    # 4. FEATURES DE CANAL/POSIÇÃO AVANÇADAS
    data['Canal_Momentum'] = data['Channel_Position'].diff()  # Movimento no canal
    data['Breakout_Signal'] = ((data['Close'] > data['Max_20']) | (data['Close'] < data['Min_20'])).astype(int)
    
    # 5. FEATURES COMBINADAS (INTERAÇÕES)
    data['RSI_Volume'] = data['RSI_Normalized'] * data['Volume_Norm']  # RSI ponderado por volume
    data['Momentum_Volatilidade'] = data['Momentum_3'] / (data['Volatilidade'] + 0.001)  # Momentum ajustado por volatilidade
    data['MM_Cross_Strength'] = (data['MM5'] - data['MM20']) / data['MM20']  # Força do cruzamento
    
    # 6. FEATURES DE REGIME DE MERCADO
    data['Bull_Market'] = (data['Close'] > data['MM50']).astype(int)  # Mercado em alta
    data['High_Vol_Regime'] = (data['Volatilidade'] > data['Volatilidade'].rolling(50).quantile(0.7)).astype(int)
    data['Consolidation'] = ((data['Max_20'] - data['Min_20']) / data['Close'] < 0.05).astype(int)  # Consolidação
    
    # 7. FEATURES LAGGED (DEFASADAS)
    data['Retorno_Lag1'] = data['Retorno'].shift(1)
    data['Retorno_Lag2'] = data['Retorno'].shift(2)
    data['RSI_Lag1'] = data['RSI'].shift(1)
    data['Volume_Norm_Lag1'] = data['Volume_Norm'].shift(1)
    
    # 8. FEATURES DE TENDÊNCIA
    data['Tendencia_5d'] = (data['Close'] > data['Close'].shift(5)).astype(int)
    data['Tendencia_10d'] = (data['Close'] > data['Close'].shift(10)).astype(int)
    data['Aceleracao'] = data['Retorno'] - data['Retorno'].shift(1)  # Aceleração do retorno
    
    # Lista completa de features
    features_avancadas = [
        # Features originais
        'MM5', 'MM20', 'RSI', 'Volatilidade', 'MM_Cross', 'Price_Above_MA20', 
        'RSI_Signal', 'Returns_3d', 'Returns_7d', 'Volume_Spike', 'Low_Vol', 'Channel_Position',
        
        # Features avançadas
        'Momentum_Ratio', 'Price_Momentum', 'Volume_Price_Momentum',
        'Volatilidade_Relativa', 'Volatilidade_Tendencia',
        'RSI_Momentum', 'RSI_Volatilidade', 'RSI_Normalized',
        'Canal_Momentum', 'Breakout_Signal',
        'RSI_Volume', 'Momentum_Volatilidade', 'MM_Cross_Strength',
        'Bull_Market', 'High_Vol_Regime', 'Consolidation',
        'Retorno_Lag1', 'Retorno_Lag2', 'RSI_Lag1', 'Volume_Norm_Lag1',
        'Tendencia_5d', 'Tendencia_10d', 'Aceleracao'
    ]
    
    print(f"✅ {len(features_avancadas)} features criadas ({len(features_avancadas) - len(features_originais)} novas)")
    print(f"   Features originais: {len(features_originais)}")
    print(f"   Features adicionadas: {len(features_avancadas) - len(features_originais)}")
    
    return data, features_avancadas

def main():
    """
    FUNÇÃO PRINCIPAL
    Executa pipeline completo com análise de correlação e seleção otimizada de features
    """
    print("="*70)
    print("MACHINE LEARNING - PREVISÃO IBOVESPA")
    print("ABORDAGEM: REGRESSÃO + ANÁLISE DE CORRELAÇÃO")
    print("META: 75% de acurácia nos últimos 30 dias")
    print("="*70)
    
    try:
        # 1. Carregar dados
        data = carregar_dados_ibovespa(anos=10)
        
        # 2. Criar features avançadas (incluindo as originais)
        data, features_completas = criar_features_avancadas(data)
        
        # 3. Criar target
        data = criar_target(data)
        
        # 4. ANÁLISE DE CORRELAÇÃO DETALHADA
        print("\n🔍 Executando análise detalhada de correlações...")
        analise_correlacao = analisar_correlacoes_features(data, features_completas)
        
        # 5. Preparar dataset com divisão específica
        X_treino, X_teste, y_treino, y_teste, y_treino_dir, y_teste_dir = preparar_dataset(data, features_completas)
        
        # 6. CORREÇÃO DE MULTICOLINEARIDADE
        print("\n🔧 Aplicando correções avançadas...")
        features_sem_multicolinearidade = remover_multicolinearidade(X_treino, features_completas, threshold=0.85)
        
        # 7. SELEÇÃO ESTATÍSTICA DE FEATURES  
        features_estatisticas = selecao_features_estatistica(
            X_treino[features_sem_multicolinearidade], y_treino, k=min(15, len(features_sem_multicolinearidade))
        )
        
        # 8. ENSEMBLE ROBUSTO
        ensemble_modelo = criar_ensemble_robusto()
        
        # 9. VALIDAÇÃO TEMPORAL ROBUSTA
        acuracia_ensemble, std_ensemble, r2_ensemble = validacao_temporal_robusta(
            X_treino, y_treino, y_treino_dir, ensemble_modelo, features_estatisticas
        )
        
        print(f"\n🎯 MODELO ENSEMBLE OTIMIZADO:")
        print(f"   Features utilizadas: {len(features_estatisticas)}")
        print(f"   Acurácia temporal: {acuracia_ensemble:.1%} (±{std_ensemble:.1%})")
        print(f"   R² temporal: {r2_ensemble:.4f}")
        
        # 10. COMPARAÇÃO: Modelo simples vs Ensemble
        print(f"\n📊 COMPARAÇÃO DE ABORDAGENS:")
        melhores_features_orig, melhor_modelo_orig, melhor_acuracia_orig = testar_selecao_features(
            X_treino, y_treino, y_treino_dir, analise_correlacao
        )
        
        print(f"   🔹 Abordagem Original: {melhor_acuracia_orig:.1%}")
        print(f"   🔹 Ensemble Otimizado: {acuracia_ensemble:.1%}")
        
        # Escolher melhor abordagem
        if acuracia_ensemble > melhor_acuracia_orig:
            print(f"   ✅ Usando ENSEMBLE (melhor performance)")
            modelo_final = ensemble_modelo
            features_finais = features_estatisticas
            acuracia_cv_final = acuracia_ensemble
        else:
            print(f"   ✅ Usando abordagem ORIGINAL (melhor performance)")
            modelo_final = melhor_modelo_orig
            features_finais = melhores_features_orig
            acuracia_cv_final = melhor_acuracia_orig
        
        # 11. Treinar modelo final com as melhores configurações
        print(f"\n🎯 TREINAMENTO FINAL COM CONFIGURAÇÃO OTIMIZADA")
        print(f"Features selecionadas: {len(features_finais)}")
        print(f"Modelo selecionado: {type(modelo_final).__name__}")
        print(f"Acurácia CV esperada: {acuracia_cv_final:.1%}")
        
        # Preparar dados com features selecionadas
        X_treino_otimizado = X_treino[features_finais]
        X_teste_otimizado = X_teste[features_finais]
        
        # 12. Teste final nos últimos 30 dias
        if isinstance(modelo_final, VotingRegressor):
            modelo_nome_final = "Ensemble"
        else:
            modelo_nome_final = modelo_final
            
        resultado_final = teste_final_30_dias(
            X_treino_otimizado, X_teste_otimizado, y_treino, y_teste, y_teste_dir, modelo_nome_final
        )
        
        # RESUMO FINAL COM FOCO NA META
        print("\n" + "="*70)
        print("RESUMO FINAL - AVALIAÇÃO DA META")
        print("="*70)
        
        print(f"🎯 META: 75% de acurácia nos últimos 30 dias")
        print(f"📊 RESULTADO: {resultado_final['acuracia']:.1%}")
        print(f"🏆 MODELO USADO: {resultado_final['modelo']}")
        print(f"📈 STATUS: {resultado_final['status']}")
        print(f"🔧 FEATURES USADAS: {len(features_finais)} selecionadas")
        
        # Comparação com validação cruzada
        diferenca = resultado_final['acuracia'] - acuracia_cv_final
        print(f"\n🔍 ANÁLISE DE CONSISTÊNCIA:")
        print(f"  Validação Cruzada: {acuracia_cv_final:.1%}")
        print(f"  Teste Final: {resultado_final['acuracia']:.1%}")
        print(f"  Diferença: {diferenca:+.1%}")
        
        if abs(diferenca) < 0.05:
            print("✅ Modelo consistente entre CV e teste final")
        else:
            print("⚠️ Grande diferença entre CV e teste - verificar overfitting")
        
        # Baseline comparison
        baseline = max(y_teste_dir.mean(), 1 - y_teste_dir.mean())
        melhoria = resultado_final['acuracia'] - baseline
        print(f"\n📊 COMPARAÇÃO COM BASELINE:")
        print(f"  Baseline (classe majoritária): {baseline:.1%}")
        print(f"  Nosso modelo: {resultado_final['acuracia']:.1%}")
        print(f"  Melhoria: {melhoria*100:+.1f} pontos percentuais")
        
        # Análise da regressão
        print(f"\n📈 QUALIDADE DA REGRESSÃO:")
        print(f"  R² no teste: {resultado_final['r2']:.4f}")
        print(f"  MSE no teste: {resultado_final['mse']:.6f}")
        print(f"  Correlação retorno real vs previsto: {np.corrcoef(resultado_final['real_retorno'], resultado_final['predicoes_retorno'])[0,1]:.4f}")
        
        # Análise das features selecionadas
        print(f"\n🎯 FEATURES FINAIS SELECIONADAS:")
        for i, feature in enumerate(features_finais[:10], 1):  # Top 10
            corr_ret = analise_correlacao['correlacoes_retorno'].get(feature, 0)
            corr_dir = analise_correlacao['correlacoes_direcao'].get(feature, 0)
            print(f"  {i:2d}. {feature:<20}: ret={corr_ret:.4f}, dir={corr_dir:.4f}")
        
        if len(features_finais) > 10:
            print(f"     ... e mais {len(features_finais) - 10} features")
        
        if resultado_final['status'] == 'SUCESSO':
            print("\n🎉 MISSÃO CUMPRIDA!")
            print("   Modelo atingiu a meta de 75% de acurácia!")
            print("   A análise de correlação foi fundamental para o sucesso!")
        else:
            print("\n📊 META NÃO ATINGIDA")
            print(f"   Melhor resultado alcançado: {resultado_final['acuracia']:.1%}")
            print(f"   Distância da meta: {(0.75 - resultado_final['acuracia'])*100:.1f} pontos")
            print("\n💡 Estratégias adicionais para considerar:")
            print("   - Features de sentimento de mercado")
            print("   - Dados macroeconômicos (taxa SELIC, câmbio)")
            print("   - Modelos não-lineares (SVM, Random Forest)")
            print("   - Dados de maior frequência (intraday)")
            print("   - Ensemble de múltiplos modelos")
        
        print("\n✓ PROJETO CONCLUÍDO!")
        
        return {
            'teste_final': resultado_final,
            'features_selecionadas': features_finais,
            'modelo_selecionado': type(modelo_final).__name__,
            'analise_correlacao': analise_correlacao,
            'meta_atingida': resultado_final['status'] == 'SUCESSO',
            'acuracia_cv': acuracia_cv_final
        }
        
    except Exception as e:
        print(f"Erro: {e}")
        import traceback
        traceback.print_exc()
        return None


def testar_selecao_features(X_treino, y_treino, y_treino_dir, analise_correlacao):
    """
    TESTE DE SELEÇÃO DE FEATURES
    Testa diferentes conjuntos de features para maximizar a acurácia
    """
    print("\n" + "="*60)
    print("TESTE DE SELEÇÃO DE FEATURES")
    print("="*60)
    
    # Diferentes conjuntos de features para testar
    conjuntos_features = {
        'TOP_5': analise_correlacao['top_5_features'],
        'TOP_8': analise_correlacao['top_8_features'],
        'TODAS_ORIGINAIS': [col for col in X_treino.columns if col in [
            'MM5', 'MM20', 'RSI', 'Volatilidade', 'MM_Cross', 'Price_Above_MA20', 
            'RSI_Signal', 'Returns_3d', 'Returns_7d', 'Volume_Spike', 'Low_Vol', 'Channel_Position'
        ]],
        'TODAS_DISPONIVEIS': list(X_treino.columns)
    }
    
    # Modelos de regressão para testar
    modelos = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0, random_state=42),
        'Lasso Regression': Lasso(alpha=0.1, random_state=42),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
    }
    
    # Função personalizada para calcular acurácia de direção
    def acuracia_direcao_regressao(y_true, y_pred):
        direcao_real = (y_true > 0).astype(int)
        direcao_pred = (y_pred > 0).astype(int)
        return np.mean(direcao_real == direcao_pred)
    
    scorer_acuracia = make_scorer(acuracia_direcao_regressao)
    
    resultados_completos = {}
    
    print("🧪 Testando diferentes combinações de features e modelos...")
    
    # Testar cada conjunto de features
    for nome_conjunto, features_selecionadas in conjuntos_features.items():
        print(f"\n📊 Testando conjunto: {nome_conjunto} ({len(features_selecionadas)} features)")
        
        # Verificar se todas as features existem
        features_disponiveis = [f for f in features_selecionadas if f in X_treino.columns]
        if len(features_disponiveis) != len(features_selecionadas):
            print(f"   ⚠️ Algumas features não disponíveis. Usando {len(features_disponiveis)} de {len(features_selecionadas)}")
        
        if len(features_disponiveis) == 0:
            print(f"   ❌ Nenhuma feature disponível para {nome_conjunto}")
            continue
            
        X_treino_subset = X_treino[features_disponiveis]
        
        resultados_conjunto = {}
        
        # Testar cada modelo
        for nome_modelo, modelo_base in modelos.items():
            try:
                # Criar pipeline
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('regressor', modelo_base)
                ])
                
                # Validação cruzada
                cv = KFold(n_splits=5, shuffle=False)
                scores_acuracia = cross_val_score(
                    pipeline, X_treino_subset, y_treino, cv=cv, 
                    scoring=scorer_acuracia, n_jobs=-1
                )
                
                scores_r2 = cross_val_score(
                    pipeline, X_treino_subset, y_treino, cv=cv, 
                    scoring='r2', n_jobs=-1
                )
                
                # Armazenar resultados
                acuracia_media = scores_acuracia.mean()
                resultados_conjunto[nome_modelo] = {
                    'acuracia_mean': acuracia_media,
                    'acuracia_std': scores_acuracia.std(),
                    'r2_mean': scores_r2.mean(),
                    'r2_std': scores_r2.std()
                }
                
                # Mostrar resultado
                status = "✅" if acuracia_media >= 0.75 else "⚠️" if acuracia_media >= 0.60 else "❌"
                print(f"     {status} {nome_modelo:<15}: {acuracia_media:.1%} (±{scores_acuracia.std():.1%})")
                
            except Exception as e:
                print(f"     ❌ {nome_modelo:<15}: Erro - {str(e)[:50]}")
                continue
        
        # Encontrar melhor modelo para este conjunto
        if resultados_conjunto:
            melhor_modelo_conjunto = max(resultados_conjunto.keys(), 
                                       key=lambda k: resultados_conjunto[k]['acuracia_mean'])
            melhor_acuracia_conjunto = resultados_conjunto[melhor_modelo_conjunto]['acuracia_mean']
            
            resultados_completos[nome_conjunto] = {
                'features': features_disponiveis,
                'n_features': len(features_disponiveis),
                'melhor_modelo': melhor_modelo_conjunto,
                'melhor_acuracia': melhor_acuracia_conjunto,
                'resultados_modelos': resultados_conjunto
            }
            
            print(f"   🏆 Melhor: {melhor_modelo_conjunto} ({melhor_acuracia_conjunto:.1%})")
    
    # RANKING GERAL
    print(f"\n" + "="*60)
    print("🏆 RANKING GERAL - MELHORES COMBINAÇÕES")
    print("="*60)
    
    # Ordenar por acurácia
    ranking = sorted(resultados_completos.items(), 
                    key=lambda x: x[1]['melhor_acuracia'], reverse=True)
    
    for i, (nome_conjunto, resultado) in enumerate(ranking, 1):
        acuracia = resultado['melhor_acuracia']
        modelo = resultado['melhor_modelo']
        n_features = resultado['n_features']
        
        status = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "🔸"
        meta_status = "✅" if acuracia >= 0.75 else "⚠️" if acuracia >= 0.60 else "❌"
        
        print(f"{status} {i}. {nome_conjunto:<18}: {acuracia:.1%} {meta_status} | {modelo} | {n_features} features")
    
    # MELHOR COMBINAÇÃO GERAL
    if ranking:
        melhor_geral = ranking[0]
        nome_melhor = melhor_geral[0]
        dados_melhor = melhor_geral[1]
        
        print(f"\n🎯 MELHOR COMBINAÇÃO ENCONTRADA:")
        print(f"   Conjunto: {nome_melhor}")
        print(f"   Modelo: {dados_melhor['melhor_modelo']}")
        print(f"   Acurácia: {dados_melhor['melhor_acuracia']:.1%}")
        print(f"   Features ({dados_melhor['n_features']}): {', '.join(dados_melhor['features'][:5])}{'...' if len(dados_melhor['features']) > 5 else ''}")
        
        if dados_melhor['melhor_acuracia'] >= 0.75:
            print("   🎉 META DE 75% ATINGIDA!")
        else:
            print(f"   📊 Faltam {(0.75 - dados_melhor['melhor_acuracia'])*100:.1f} pontos para a meta")
        
        return dados_melhor['features'], dados_melhor['melhor_modelo'], dados_melhor['melhor_acuracia']
    
    return None, None, 0.0

def remover_multicolinearidade(X_treino, features_lista, threshold=0.9):
    """
    Remove features com alta multicolinearidade
    """
    print(f"🔧 Removendo multicolinearidade (threshold: {threshold})...")
    
    # Calcular matriz de correlação
    corr_matrix = X_treino[features_lista].corr().abs()
    
    # Encontrar features para remover
    features_to_remove = set()
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > threshold:
                # Remover a feature com menor correlação com o target
                colname = corr_matrix.columns[j]
                features_to_remove.add(colname)
    
    # Features finais
    features_finais = [f for f in features_lista if f not in features_to_remove]
    
    print(f"   📊 Features removidas: {len(features_to_remove)}")
    print(f"   ✅ Features mantidas: {len(features_finais)}")
    
    if len(features_to_remove) > 0:
        print(f"   🗑️ Removidas: {list(features_to_remove)[:5]}{'...' if len(features_to_remove) > 5 else ''}")
    
    return features_finais

def selecao_features_estatistica(X_treino, y_treino, k=10):
    """
    Seleção de features usando métodos estatísticos
    """
    print(f"📊 Seleção estatística de features (k={k})...")
    
    # SelectKBest com f_regression
    selector = SelectKBest(score_func=f_regression, k=k)
    X_selected = selector.fit_transform(X_treino, y_treino)
    
    # Obter features selecionadas
    feature_mask = selector.get_support()
    features_selecionadas = X_treino.columns[feature_mask].tolist()
    
    # Mostrar scores
    scores = selector.scores_
    feature_scores = list(zip(X_treino.columns, scores))
    feature_scores.sort(key=lambda x: x[1], reverse=True)
    
    print(f"   ✅ {len(features_selecionadas)} features selecionadas")
    print(f"   🏆 Top 5 scores:")
    for i, (feature, score) in enumerate(feature_scores[:5], 1):
        print(f"     {i}. {feature:<20}: {score:.2f}")
    
    return features_selecionadas

def criar_ensemble_robusto():
    """
    Cria ensemble de modelos para maior robustez
    """
    print("🤖 Criando ensemble de modelos...")
    
    # Modelos com diferentes características
    modelos = [
        ('linear', LinearRegression()),
        ('ridge', Ridge(alpha=1.0, random_state=42)),
        ('lasso', Lasso(alpha=0.01, random_state=42)),
        ('elastic', ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42))
    ]
    
    # Ensemble com votação
    ensemble = VotingRegressor(estimators=modelos)
    
    print(f"   ✅ Ensemble criado com {len(modelos)} modelos")
    
    return ensemble

def validacao_temporal_robusta(X_treino, y_treino, y_treino_dir, modelo, features_selecionadas):
    """
    Validação cruzada temporal mais robusta
    """
    print("⏰ Validação cruzada temporal robusta...")
    
    # Usar apenas features selecionadas
    X_treino_sel = X_treino[features_selecionadas]
    
    # TimeSeriesSplit para dados temporais
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Pipeline com ensemble
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', modelo)
    ])
    
    # Função para calcular acurácia de direção
    def acuracia_direcao_regressao(y_true, y_pred):
        direcao_real = (y_true > 0).astype(int)
        direcao_pred = (y_pred > 0).astype(int)
        return np.mean(direcao_real == direcao_pred)
    
    scorer_acuracia = make_scorer(acuracia_direcao_regressao)
    
    # Validação cruzada temporal
    scores_acuracia = cross_val_score(
        pipeline, X_treino_sel, y_treino, cv=tscv, 
        scoring=scorer_acuracia, n_jobs=-1
    )
    
    scores_r2 = cross_val_score(
        pipeline, X_treino_sel, y_treino, cv=tscv, 
        scoring='r2', n_jobs=-1
    )
    
    acuracia_media = scores_acuracia.mean()
    acuracia_std = scores_acuracia.std()
    r2_medio = scores_r2.mean()
    
    print(f"   📊 Acurácia temporal: {acuracia_media:.1%} (±{acuracia_std:.1%})")
    print(f"   📈 R² temporal: {r2_medio:.4f}")
    
    return acuracia_media, acuracia_std, r2_medio
