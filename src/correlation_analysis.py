"""
Análise de correlação e seleção de features
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression

def analisar_correlacoes(data, features_lista, target_col='Target_Return'):
    """
    Analisa correlações entre features e targets
    
    Args:
        data (pd.DataFrame): Dados completos
        features_lista (list): Lista de features
        target_col (str): Nome da coluna target
        
    Returns:
        dict: Resultados da análise de correlação
    """
    print("🔍 Analisando correlações...")
    
    # Dataset para análise
    dataset = data[features_lista + [target_col, 'Target_Direction']].copy()
    dataset = dataset.dropna()
    
    # Correlações com target de regressão
    correlacoes_retorno = dataset[features_lista].corrwith(dataset[target_col]).abs().sort_values(ascending=False)
    
    # Correlações com target de direção
    correlacoes_direcao = dataset[features_lista].corrwith(dataset['Target_Direction']).abs().sort_values(ascending=False)
    
    # Correlação combinada (60% retorno + 40% direção)
    correlacao_combinada = (correlacoes_retorno * 0.6 + correlacoes_direcao * 0.4).sort_values(ascending=False)
    
    # Features top
    top_5_features = correlacao_combinada.head(5).index.tolist()
    top_10_features = correlacao_combinada.head(10).index.tolist()
    
    # Multicolinearidade
    corr_matrix = dataset[features_lista].corr().abs()
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > 0.85:
                high_corr_pairs.append((
                    corr_matrix.columns[i], 
                    corr_matrix.columns[j], 
                    corr_matrix.iloc[i, j]
                ))
    
    print(f"✅ Análise de correlação concluída")
    print(f"   Top 5 features: {top_5_features}")
    print(f"   Pares multicolineares: {len(high_corr_pairs)}")
    
    return {
        'correlacoes_retorno': correlacoes_retorno,
        'correlacoes_direcao': correlacoes_direcao,
        'correlacao_combinada': correlacao_combinada,
        'top_5_features': top_5_features,
        'top_10_features': top_10_features,
        'high_corr_pairs': high_corr_pairs,
        'dataset': dataset
    }

def remover_multicolinearidade(features_lista, corr_matrix, threshold=0.85):
    """
    Remove features com alta correlação
    
    Args:
        features_lista (list): Lista de features
        corr_matrix (pd.DataFrame): Matriz de correlação
        threshold (float): Limite para remoção
        
    Returns:
        list: Features sem multicolinearidade
    """
    features_to_remove = set()
    
    for i in range(len(features_lista)):
        for j in range(i+1, len(features_lista)):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                # Remove a segunda feature do par
                features_to_remove.add(features_lista[j])
    
    features_finais = [f for f in features_lista if f not in features_to_remove]
    
    print(f"🔧 Multicolinearidade removida:")
    print(f"   Features removidas: {len(features_to_remove)}")
    print(f"   Features mantidas: {len(features_finais)}")
    
    return features_finais

def selecionar_features_otimizadas(analise, features_completas, k=12):
    """
    Seleciona features otimizadas baseado na análise
    
    Args:
        analise (dict): Resultados da análise de correlação
        features_completas (list): Lista completa de features
        k (int): Número de features a selecionar
        
    Returns:
        list: Features selecionadas
    """
    print(f"⚡ Selecionando {k} features otimizadas...")
    
    # Remover multicolinearidade primeiro
    dataset = analise['dataset']
    corr_matrix = dataset[features_completas].corr().abs()
    features_sem_multicolinear = remover_multicolinearidade(features_completas, corr_matrix)
    
    # Selecionar top features sem multicolinearidade
    correlacao_combinada = analise['correlacao_combinada']
    features_ordenadas = [f for f in correlacao_combinada.index if f in features_sem_multicolinear]
    
    # Tomar as k melhores
    features_selecionadas = features_ordenadas[:min(k, len(features_ordenadas))]
    
    print(f"✅ Features selecionadas: {len(features_selecionadas)}")
    for i, feature in enumerate(features_selecionadas[:5], 1):
        corr = correlacao_combinada[feature]
        print(f"   {i}. {feature}: {corr:.4f}")
    
    return features_selecionadas
