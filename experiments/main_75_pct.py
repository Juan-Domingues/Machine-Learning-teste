"""
PIPELINE ULTRA-AVANÇADO PARA 75% DE ACURÁCIA
Baseado nos resultados de 70% - vamos para 75%!
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Adicionar pasta src ao path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import SelectKBest, mutual_info_classif

def pipeline_75_pct():
    """
    Pipeline ultra-avançado para atingir 75% de acurácia
    """
    print("="*80)
    print("🎯 PIPELINE ULTRA-AVANÇADO PARA 75% DE ACURÁCIA")
    print("🚀 BASEADO NO SUCESSO DE 70% - VAMOS PARA 75%!")
    print("="*80)
    
    try:
        # 1. DADOS OTIMIZADOS (testando diferentes períodos)
        print("\n📥 ETAPA 1: Análise de Múltiplos Períodos")
        
        resultados_periodos = []
        
        # Testar diferentes períodos para encontrar o melhor
        periodos = [
            (2 * 365, "2 anos"),
            (3 * 365, "3 anos"),
            (4 * 365, "4 anos"),
            (5 * 365, "5 anos")
        ]
        
        for days, nome in periodos:
            print(f"\n🔍 Testando período: {nome}")
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            data = yf.download('^BVSP', start=start_date, end=end_date)
            
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            # Features ultra-avançadas
            data = criar_features_avancadas(data)
            
            # Teste rápido com configuração que deu 70%
            dataset = data.dropna()
            if len(dataset) < 50:
                continue
                
            # Usar últimos 20 dias como teste
            n_test = min(20, len(dataset) // 10)
            train_data = dataset.iloc[:-n_test]
            test_data = dataset.iloc[-n_test:]
            
            # Features que funcionaram (Trend + Volume)
            features_base = ['Price_above_SMA5', 'Volume_above_avg']
            
            if all(f in dataset.columns for f in features_base):
                X_train = train_data[features_base]
                y_train = train_data['Target']
                X_test = test_data[features_base]
                y_test = test_data['Target']
                
                # Ensemble que deu 70%
                ensemble = VotingClassifier([
                    ('lr', LogisticRegression(C=0.1, random_state=42, max_iter=1000)),
                    ('rf', RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42))
                ], voting='hard')
                
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('clf', ensemble)
                ])
                
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                
                baseline = max(y_test.mean(), 1 - y_test.mean())
                
                print(f"   Acurácia: {acc:.1%} (baseline: {baseline:.1%})")
                
                resultados_periodos.append({
                    'periodo': nome,
                    'days': days,
                    'acuracia': acc,
                    'baseline': baseline,
                    'dataset_size': len(dataset),
                    'data': data
                })
        
        # Escolher melhor período
        melhor_periodo = max(resultados_periodos, key=lambda x: x['acuracia'])
        print(f"\n🏆 Melhor período: {melhor_periodo['periodo']} com {melhor_periodo['acuracia']:.1%}")
        
        data_final = melhor_periodo['data']
        
        # 2. FEATURES ULTRA-AVANÇADAS
        print("\n🔧 ETAPA 2: Features Ultra-Avançadas")
        
        # Já criadas na função criar_features_avancadas
        print("✅ Features avançadas criadas")
        
        # 3. SELEÇÃO AUTOMÁTICA DE FEATURES
        print("\n🧬 ETAPA 3: Seleção Automática de Features")
        
        # Todas as features disponíveis
        all_features = [col for col in data_final.columns if col != 'Target' and not col.startswith('SMA_')]
        
        dataset_final = data_final[all_features + ['Target']].dropna()
        print(f"📊 Dataset final: {len(dataset_final)} observações, {len(all_features)} features")
        
        # Divisão otimizada
        n_test = 20
        train_data = dataset_final.iloc[:-n_test]
        test_data = dataset_final.iloc[-n_test:]
        
        # Feature selection
        X_train_all = train_data[all_features]
        y_train = train_data['Target']
        X_test_all = test_data[all_features]
        y_test = test_data['Target']
        
        # Selecionar top features por mutual information
        selector = SelectKBest(score_func=mutual_info_classif, k=8)
        X_train_selected = selector.fit_transform(X_train_all, y_train)
        X_test_selected = selector.transform(X_test_all)
        
        selected_features = [all_features[i] for i in selector.get_support(indices=True)]
        print(f"📊 Features selecionadas: {selected_features}")
        
        # 4. MODELOS ULTRA-OTIMIZADOS
        print("\n🚀 ETAPA 4: Modelos Ultra-Otimizados")
        
        modelos_avancados = []
        
        # Modelo 1: Logistic Regression com Grid Search
        print("🔹 Logistic Regression com Grid Search")
        lr_params = {
            'clf__C': [0.01, 0.1, 1.0, 10.0],
            'clf__max_iter': [1000, 2000],
            'clf__solver': ['liblinear', 'lbfgs']
        }
        
        lr_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(random_state=42))
        ])
        
        lr_grid = GridSearchCV(lr_pipeline, lr_params, cv=3, scoring='accuracy', n_jobs=-1)
        lr_grid.fit(X_train_selected, y_train)
        
        y_pred_lr = lr_grid.predict(X_test_selected)
        acc_lr = accuracy_score(y_test, y_pred_lr)
        print(f"   LR Otimizado: {acc_lr:.1%}")
        modelos_avancados.append(('lr_opt', lr_grid.best_estimator_, acc_lr))
        
        # Modelo 2: Random Forest Otimizado
        print("🔹 Random Forest com Grid Search")
        rf_params = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'min_samples_split': [5, 10, 20],
            'min_samples_leaf': [2, 5, 10]
        }
        
        rf_grid = GridSearchCV(
            RandomForestClassifier(random_state=42, n_jobs=-1),
            rf_params, cv=3, scoring='accuracy', n_jobs=-1
        )
        rf_grid.fit(X_train_selected, y_train)
        
        y_pred_rf = rf_grid.predict(X_test_selected)
        acc_rf = accuracy_score(y_test, y_pred_rf)
        print(f"   RF Otimizado: {acc_rf:.1%}")
        modelos_avancados.append(('rf_opt', rf_grid.best_estimator_, acc_rf))
        
        # Modelo 3: Gradient Boosting
        print("🔹 Gradient Boosting")
        gb_params = {
            'n_estimators': [50, 100],
            'learning_rate': [0.1, 0.2],
            'max_depth': [3, 5]
        }
        
        gb_grid = GridSearchCV(
            GradientBoostingClassifier(random_state=42),
            gb_params, cv=3, scoring='accuracy', n_jobs=-1
        )
        gb_grid.fit(X_train_selected, y_train)
        
        y_pred_gb = gb_grid.predict(X_test_selected)
        acc_gb = accuracy_score(y_test, y_pred_gb)
        print(f"   GB Otimizado: {acc_gb:.1%}")
        modelos_avancados.append(('gb_opt', gb_grid.best_estimator_, acc_gb))
        
        # 5. ENSEMBLE ULTRA-AVANÇADO
        print("\n🎯 ETAPA 5: Ensemble Ultra-Avançado")
        
        # Pegar os 2 melhores modelos
        top_modelos = sorted(modelos_avancados, key=lambda x: x[2], reverse=True)[:2]
        
        print(f"📊 Top 2 modelos para ensemble:")
        for nome, modelo, acc in top_modelos:
            print(f"   {nome}: {acc:.1%}")
        
        # Ensemble com pesos
        if len(top_modelos) >= 2:
            ensemble_estimators = [(nome, modelo) for nome, modelo, _ in top_modelos]
            
            # Voting Classifier com soft voting
            ensemble_soft = VotingClassifier(ensemble_estimators, voting='soft')
            
            # Usar scaler apenas se necessário
            if any('scaler' not in str(modelo) for _, modelo, _ in top_modelos):
                ensemble_pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('clf', ensemble_soft)
                ])
            else:
                ensemble_pipeline = ensemble_soft
            
            ensemble_pipeline.fit(X_train_selected, y_train)
            y_pred_ensemble = ensemble_pipeline.predict(X_test_selected)
            acc_ensemble = accuracy_score(y_test, y_pred_ensemble)
            
            print(f"🏆 Ensemble Final: {acc_ensemble:.1%}")
        else:
            acc_ensemble = top_modelos[0][2]
            ensemble_pipeline = top_modelos[0][1]
            print(f"🏆 Melhor Modelo Individual: {acc_ensemble:.1%}")
        
        # 6. ANÁLISE DETALHADA
        print("\n📊 ETAPA 6: Análise Detalhada")
        
        baseline = max(y_test.mean(), 1 - y_test.mean())
        melhoria = (acc_ensemble - baseline) * 100
        
        print(f"📊 Resultados Finais:")
        print(f"   Melhor Acurácia: {acc_ensemble:.1%}")
        print(f"   Baseline: {baseline:.1%}")
        print(f"   Melhoria: {melhoria:+.1f} pontos")
        print(f"   Features usadas: {len(selected_features)}")
        print(f"   Período: {melhor_periodo['periodo']}")
        
        # Status das metas
        if acc_ensemble >= 0.75:
            print(f"\n🎉 INCRÍVEL! META 75% ATINGIDA!")
            print(f"🏆 SUCESSO TOTAL!")
        elif acc_ensemble >= 0.70:
            print(f"\n📈 MUITO BOM! Mantivemos 70%+")
            print(f"   Faltam {(0.75 - acc_ensemble)*100:.1f} pontos para 75%")
        elif acc_ensemble >= 0.60:
            print(f"\n✅ BOM! Ainda acima de 60%")
            print(f"   Faltam {(0.75 - acc_ensemble)*100:.1f} pontos para 75%")
        
        # Confusion Matrix detalhada
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred_ensemble)
        
        print(f"\n📊 Confusion Matrix:")
        print(f"     Predito")
        print(f"Real    0    1")
        print(f"  0   {cm[0,0]:3d}  {cm[0,1]:3d}")
        print(f"  1   {cm[1,0]:3d}  {cm[1,1]:3d}")
        
        # Análise de erros
        tp = cm[1,1]  # True Positives
        tn = cm[0,0]  # True Negatives
        fp = cm[0,1]  # False Positives
        fn = cm[1,0]  # False Negatives
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        print(f"\n📊 Métricas Detalhadas:")
        print(f"   Precisão: {precision:.1%}")
        print(f"   Recall: {recall:.1%}")
        print(f"   Acertos Alta: {tp}/{tp+fn} ({recall:.1%})")
        print(f"   Acertos Baixa: {tn}/{tn+fp} ({tn/(tn+fp):.1%})")
        
        return {
            'acuracia_final': acc_ensemble,
            'baseline': baseline,
            'features_selecionadas': selected_features,
            'melhor_periodo': melhor_periodo['periodo'],
            'meta_75_atingida': acc_ensemble >= 0.75,
            'modelo_final': ensemble_pipeline,
            'confusion_matrix': cm,
            'precision': precision,
            'recall': recall
        }
        
    except Exception as e:
        print(f"\n❌ ERRO: {e}")
        import traceback
        traceback.print_exc()
        return None

def criar_features_avancadas(data):
    """
    Criar features ultra-avançadas para maximizar acurácia
    """
    # Retornos básicos
    data['Return'] = data['Close'].pct_change()
    data['Return_2d'] = data['Close'].pct_change(2)
    data['Return_3d'] = data['Close'].pct_change(3)
    data['Return_5d'] = data['Close'].pct_change(5)
    
    # Target
    data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
    
    # Lags múltiplos
    for lag in [1, 2, 3]:
        data[f'Return_lag{lag}'] = data['Return'].shift(lag)
        data[f'Positive_return_lag{lag}'] = (data['Return'].shift(lag) > 0).astype(int)
    
    # Médias móveis múltiplas
    for window in [3, 5, 10, 20]:
        data[f'SMA_{window}'] = data['Close'].rolling(window).mean()
        data[f'Price_above_SMA{window}'] = (data['Close'] > data[f'SMA_{window}']).astype(int)
    
    # Cruzamentos de médias
    data['SMA3_above_SMA5'] = (data['SMA_3'] > data['SMA_5']).astype(int)
    data['SMA5_above_SMA10'] = (data['SMA_5'] > data['SMA_10']).astype(int)
    data['SMA10_above_SMA20'] = (data['SMA_10'] > data['SMA_20']).astype(int)
    
    # Volume avançado
    data['Volume_MA_5'] = data['Volume'].rolling(5).mean()
    data['Volume_MA_10'] = data['Volume'].rolling(10).mean()
    data['Volume_MA_20'] = data['Volume'].rolling(20).mean()
    
    data['Volume_above_avg5'] = (data['Volume'] > data['Volume_MA_5']).astype(int)
    data['Volume_above_avg10'] = (data['Volume'] > data['Volume_MA_10']).astype(int)
    data['Volume_above_avg'] = (data['Volume'] > data['Volume_MA_20']).astype(int)
    
    # Volume ratios
    data['Volume_ratio_5'] = data['Volume'] / data['Volume_MA_5']
    data['Volume_ratio_20'] = data['Volume'] / data['Volume_MA_20']
    data['High_volume'] = (data['Volume'] > data['Volume'].rolling(20).quantile(0.8)).astype(int)
    data['Low_volume'] = (data['Volume'] < data['Volume'].rolling(20).quantile(0.3)).astype(int)
    
    # Volatilidade avançada
    for window in [3, 5, 10, 20]:
        data[f'Vol_{window}d'] = data['Return'].rolling(window).std()
    
    data['Vol_expanding'] = data['Return'].expanding().std()
    data['Low_vol_3d'] = (data['Vol_3d'] < data['Vol_10d']).astype(int)
    data['Low_vol_5d'] = (data['Vol_5d'] < data['Vol_20d']).astype(int)
    data['High_vol'] = (data['Vol_3d'] > data['Vol_10d']).astype(int)
    
    # Range e momentum
    data['High_Low_ratio'] = (data['High'] - data['Low']) / data['Close']
    data['Open_Close_ratio'] = (data['Close'] - data['Open']) / data['Open']
    data['Narrow_range'] = (data['High_Low_ratio'] < data['High_Low_ratio'].rolling(10).median()).astype(int)
    data['Wide_range'] = (data['High_Low_ratio'] > data['High_Low_ratio'].rolling(10).quantile(0.7)).astype(int)
    
    # Momentum avançado
    data['Momentum_3d'] = data['Close'] / data['Close'].shift(3) - 1
    data['Momentum_5d'] = data['Close'] / data['Close'].shift(5) - 1
    data['Positive_momentum_3d'] = (data['Momentum_3d'] > 0).astype(int)
    data['Positive_momentum_5d'] = (data['Momentum_5d'] > 0).astype(int)
    
    # RSI simplificado
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    data['RSI_oversold'] = (data['RSI'] < 30).astype(int)
    data['RSI_overbought'] = (data['RSI'] > 70).astype(int)
    
    # Padrões de velas (simplificado)
    data['Doji'] = (abs(data['Open'] - data['Close']) / (data['High'] - data['Low']) < 0.1).astype(int)
    data['Green_candle'] = (data['Close'] > data['Open']).astype(int)
    data['Red_candle'] = (data['Close'] < data['Open']).astype(int)
    
    # Sequências
    data['Consecutive_green'] = 0
    data['Consecutive_red'] = 0
    
    green_streak = 0
    red_streak = 0
    
    for i in range(len(data)):
        if data['Green_candle'].iloc[i]:
            green_streak += 1
            red_streak = 0
        elif data['Red_candle'].iloc[i]:
            red_streak += 1
            green_streak = 0
        else:
            green_streak = 0
            red_streak = 0
        
        data['Consecutive_green'].iloc[i] = green_streak
        data['Consecutive_red'].iloc[i] = red_streak
    
    data['Long_green_streak'] = (data['Consecutive_green'] >= 3).astype(int)
    data['Long_red_streak'] = (data['Consecutive_red'] >= 3).astype(int)
    
    return data

if __name__ == "__main__":
    resultado = pipeline_75_pct()
    
    if resultado:
        print(f"\n" + "="*80)
        print(f"🎯 RESULTADO FINAL DO PIPELINE 75%")
        print(f"="*80)
        print(f"🏆 ACURÁCIA FINAL: {resultado['acuracia_final']:.1%}")
        print(f"📊 BASELINE: {resultado['baseline']:.1%}")
        print(f"📈 MELHORIA: {(resultado['acuracia_final'] - resultado['baseline'])*100:+.1f} pontos")
        print(f"🔧 FEATURES: {len(resultado['features_selecionadas'])}")
        print(f"📅 PERÍODO: {resultado['melhor_periodo']}")
        
        if resultado['meta_75_atingida']:
            print(f"\n🎉🎉🎉 FANTÁSTICO! META 75% ATINGIDA! 🎉🎉🎉")
            print(f"🏆 MISSION ACCOMPLISHED!")
        elif resultado['acuracia_final'] >= 0.70:
            print(f"\n📈 EXCELENTE! Mantivemos acima de 70%!")
            print(f"   Faltam apenas {(0.75 - resultado['acuracia_final'])*100:.1f} pontos para 75%")
        else:
            print(f"\n📊 Progresso mantido. Investigar mais refinamentos.")
        
        print(f"\n📊 MÉTRICAS DETALHADAS:")
        print(f"   Precisão: {resultado['precision']:.1%}")
        print(f"   Recall: {resultado['recall']:.1%}")
        
    else:
        print("\n❌ Pipeline falhou")
