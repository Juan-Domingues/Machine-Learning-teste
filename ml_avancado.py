import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

# ML Avançado
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import RFE, mutual_info_classif, SelectKBest

def feature_engineering_avancado(data):
    """
    Feature Engineering Avançado com indicadores técnicos sofisticados
    """
    print("🔧 Feature Engineering Avançado...")
    
    # Features básicas
    data['Return'] = data['Close'].pct_change()
    data['Log_Return'] = np.log(data['Close']/data['Close'].shift(1))
    
    # 1. MOVING AVERAGES AVANÇADAS
    for period in [3, 5, 7, 10, 15, 20, 50]:
        data[f'SMA_{period}'] = data['Close'].rolling(period).mean()
        data[f'EMA_{period}'] = data['Close'].ewm(span=period).mean()
        data[f'Price_above_SMA{period}'] = (data['Close'] > data[f'SMA_{period}']).astype(int)
        data[f'Price_above_EMA{period}'] = (data['Close'] > data[f'EMA_{period}']).astype(int)
        
        # Distância normalizada das médias
        data[f'SMA_{period}_dist'] = (data['Close'] - data[f'SMA_{period}']) / data[f'SMA_{period}']
        data[f'EMA_{period}_dist'] = (data['Close'] - data[f'EMA_{period}']) / data[f'EMA_{period}']
    
    # 2. INDICADORES TÉCNICOS AVANÇADOS (Implementação Manual)
    print("   Calculando indicadores técnicos...")
    
    # RSI em múltiplos períodos
    for period in [7, 14, 21]:
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        data[f'RSI_{period}'] = 100 - (100 / (1 + rs))
        data[f'RSI_{period}_overbought'] = (data[f'RSI_{period}'] > 70).astype(int)
        data[f'RSI_{period}_oversold'] = (data[f'RSI_{period}'] < 30).astype(int)
    
    # MACD (Manual)
    data['EMA_12_MACD'] = data['Close'].ewm(span=12).mean()
    data['EMA_26_MACD'] = data['Close'].ewm(span=26).mean()
    data['MACD'] = data['EMA_12_MACD'] - data['EMA_26_MACD']
    data['MACD_signal'] = data['MACD'].ewm(span=9).mean()
    data['MACD_hist'] = data['MACD'] - data['MACD_signal']
    data['MACD_bullish'] = (data['MACD'] > data['MACD_signal']).astype(int)
    
    # Bollinger Bands (Manual)
    for period in [20]:
        data[f'BB_middle_{period}'] = data['Close'].rolling(period).mean()
        std = data['Close'].rolling(period).std()
        data[f'BB_upper_{period}'] = data[f'BB_middle_{period}'] + (std * 2)
        data[f'BB_lower_{period}'] = data[f'BB_middle_{period}'] - (std * 2)
        data[f'BB_position_{period}'] = (data['Close'] - data[f'BB_lower_{period}']) / (data[f'BB_upper_{period}'] - data[f'BB_lower_{period}'])
        data[f'BB_squeeze_{period}'] = ((data[f'BB_upper_{period}'] - data[f'BB_lower_{period}']) / data[f'BB_middle_{period}'] < 0.1).astype(int)
        data[f'BB_above_upper_{period}'] = (data['Close'] > data[f'BB_upper_{period}']).astype(int)
        data[f'BB_below_lower_{period}'] = (data['Close'] < data[f'BB_lower_{period}']).astype(int)
    
    # Stochastic (Manual)
    for period in [14]:
        data[f'Lowest_Low_{period}'] = data['Low'].rolling(period).min()
        data[f'Highest_High_{period}'] = data['High'].rolling(period).max()
        data[f'STOCH_K_{period}'] = 100 * (data['Close'] - data[f'Lowest_Low_{period}']) / (data[f'Highest_High_{period}'] - data[f'Lowest_Low_{period}'])
        data[f'STOCH_D_{period}'] = data[f'STOCH_K_{period}'].rolling(3).mean()
        data[f'STOCH_oversold_{period}'] = (data[f'STOCH_K_{period}'] < 20).astype(int)
        data[f'STOCH_overbought_{period}'] = (data[f'STOCH_K_{period}'] > 80).astype(int)
    
    # 3. VOLUME FEATURES AVANÇADAS
    for period in [5, 10, 20]:
        data[f'Volume_MA_{period}'] = data['Volume'].rolling(period).mean()
        data[f'Volume_ratio_{period}'] = data['Volume'] / data[f'Volume_MA_{period}']
        data[f'Volume_above_avg_{period}'] = (data['Volume'] > data[f'Volume_MA_{period}']).astype(int)
    
    # Volume Price Trend
    data['VPT'] = (data['Volume'] * data['Return']).cumsum()
    data['VPT_signal'] = data['VPT'].rolling(10).mean()
    data['VPT_bullish'] = (data['VPT'] > data['VPT_signal']).astype(int)
    
    # 4. MOMENTUM E VOLATILIDADE
    for period in [1, 3, 5, 10]:
        data[f'Return_{period}d'] = data['Close'].pct_change(period)
        data[f'Momentum_{period}d'] = (data[f'Return_{period}d'] > 0).astype(int)
    
    # Volatilidade
    for period in [5, 10, 20]:
        data[f'Volatility_{period}d'] = data['Return'].rolling(period).std()
        data[f'High_volatility_{period}d'] = (data[f'Volatility_{period}d'] > 
                                              data[f'Volatility_{period}d'].rolling(50).mean()).astype(int)
    
    # 5. FEATURES DE TENDÊNCIA
    # Slope das médias móveis
    for period in [10, 20]:
        data[f'SMA_{period}_slope'] = data[f'SMA_{period}'].diff(5) / data[f'SMA_{period}'].shift(5)
        data[f'SMA_{period}_uptrend'] = (data[f'SMA_{period}_slope'] > 0).astype(int)
    
    # Cross de médias
    data['SMA_5_20_cross'] = (data['SMA_5'] > data['SMA_20']).astype(int)
    data['SMA_10_50_cross'] = (data['SMA_10'] > data['SMA_50']).astype(int)
    data['EMA_5_20_cross'] = (data['EMA_5'] > data['EMA_20']).astype(int)
    
    # 6. FEATURES DE PADRÕES
    # Gaps
    data['Gap_up'] = (data['Open'] > data['Close'].shift(1) * 1.005).astype(int)
    data['Gap_down'] = (data['Open'] < data['Close'].shift(1) * 0.995).astype(int)
    
    # Reversão
    data['Reversal_signal'] = ((data['Low'] < data['Low'].rolling(5).min()) & 
                               (data['Close'] > data['Open'])).astype(int)
    
    # 7. FEATURES ECONÔMICAS
    # Dia da semana (efeito calendar)
    data['DayOfWeek'] = data.index.dayofweek
    data['Monday'] = (data['DayOfWeek'] == 0).astype(int)
    data['Friday'] = (data['DayOfWeek'] == 4).astype(int)
    
    # Mês
    data['Month'] = data.index.month
    data['December'] = (data['Month'] == 12).astype(int)
    data['January'] = (data['Month'] == 1).astype(int)
    
    return data

def selecao_features_avancada(X, y, n_features=20):
    """
    Seleção de features usando múltiplos métodos
    """
    print(f"🎯 Seleção de Features Avançada (top {n_features})...")
    
    # 1. Mutual Information
    mi_scores = mutual_info_classif(X, y, random_state=42)
    mi_features = X.columns[np.argsort(mi_scores)[-n_features:]].tolist()
    
    # 2. RFE com RandomForest
    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    rfe = RFE(rf, n_features_to_select=n_features)
    rfe.fit(X, y)
    rfe_features = X.columns[rfe.support_].tolist()
    
    # 3. Feature Importance do RandomForest
    rf.fit(X, y)
    importance_scores = rf.feature_importances_
    importance_features = X.columns[np.argsort(importance_scores)[-n_features:]].tolist()
    
    # Combinar features (união dos 3 métodos)
    all_selected = list(set(mi_features + rfe_features + importance_features))
    
    print(f"✅ Features selecionadas: {len(all_selected)}")
    print(f"   - Mutual Info: {len(mi_features)}")
    print(f"   - RFE: {len(rfe_features)}")
    print(f"   - Importance: {len(importance_features)}")
    
    return all_selected[:n_features]  # Limitar ao número desejado

def criar_ensemble_avancado():
    """
    Cria ensemble de modelos com hiperparâmetros otimizados
    """
    
    # 1. Logistic Regression
    lr = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(random_state=42, max_iter=2000))
    ])
    
    # 2. Random Forest
    rf = Pipeline([
        ('scaler', RobustScaler()),
        ('clf', RandomForestClassifier(random_state=42))
    ])
    
    # 3. Gradient Boosting
    gb = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', GradientBoostingClassifier(random_state=42))
    ])
    
    # Ensemble Voting
    ensemble = VotingClassifier([
        ('lr', lr),
        ('rf', rf),
        ('gb', gb)
    ], voting='soft')
    
    return ensemble

def otimizar_hiperparametros(X_train, y_train):
    """
    Otimização de hiperparâmetros com GridSearch
    """
    print("🔍 Otimização de Hiperparâmetros...")
    
    # Grid de parâmetros otimizado
    param_grid = {
        'lr__clf__C': [0.1, 1.0, 10.0],
        'rf__clf__n_estimators': [50, 100],
        'rf__clf__max_depth': [5, 10, None],
        'rf__clf__min_samples_split': [2, 5],
        'gb__clf__n_estimators': [50, 100],
        'gb__clf__learning_rate': [0.05, 0.1],
        'gb__clf__max_depth': [3, 5]
    }
    
    ensemble = criar_ensemble_avancado()
    
    # GridSearch com validação temporal
    tscv = TimeSeriesSplit(n_splits=3)  # Reduzido para performance
    
    grid_search = GridSearchCV(
        ensemble, 
        param_grid, 
        cv=tscv, 
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    print("   Executando GridSearch...")
    grid_search.fit(X_train, y_train)
    
    print(f"✅ Melhores parâmetros encontrados!")
    print(f"   Score: {grid_search.best_score_:.1%}")
    
    return grid_search.best_estimator_

def pipeline_ml_avancado():
    """
    Pipeline completo de Machine Learning Avançado
    """
    print("="*80)
    print("🚀 MACHINE LEARNING AVANÇADO - PIPELINE COMPLETO")
    print("🎯 Ensemble + Feature Engineering + Hyperparameter Tuning")
    print("="*80)
    
    try:
        # 1. CARREGAMENTO DE DADOS (2 anos para mais dados)
        print("\n📥 ETAPA 1: Carregamento de Dados Estendido")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=2*365)  # 2 anos
        data = yf.download('^BVSP', start=start_date, end=end_date)
        
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        print(f"✅ {len(data)} dias carregados (2 anos)")
        
        # 2. FEATURE ENGINEERING AVANÇADO
        print("\n🔧 ETAPA 2: Feature Engineering Avançado")
        data = feature_engineering_avancado(data)
        
        # Target
        data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
        
        # Remover colunas não-features
        feature_cols = [col for col in data.columns if col not in 
                       ['Open', 'High', 'Low', 'Close', 'Volume', 'Target', 'Adj Close']]
        
        print(f"✅ {len(feature_cols)} features criadas")
        
        # 3. PREPARAÇÃO DO DATASET
        print("\n📊 ETAPA 3: Preparação do Dataset")
        dataset = data[feature_cols + ['Target']].dropna()
        print(f"📊 Dataset: {len(dataset)} observações")
        
        # Verificar se temos dados suficientes
        if len(dataset) < 100:
            print("❌ Dados insuficientes após feature engineering")
            return None
        
        # Divisão temporal
        n_test = 30  # Teste maior para mais confiabilidade
        train_data = dataset.iloc[:-n_test]
        test_data = dataset.iloc[-n_test:]
        
        X_train = train_data[feature_cols]
        y_train = train_data['Target']
        X_test = test_data[feature_cols]
        y_test = test_data['Target']
        
        # 4. SELEÇÃO DE FEATURES AVANÇADA
        print("\n🎯 ETAPA 4: Seleção de Features Avançada")
        features_selecionadas = selecao_features_avancada(X_train, y_train, n_features=15)
        
        X_train_selected = X_train[features_selecionadas]
        X_test_selected = X_test[features_selecionadas]
        
        # 5. OTIMIZAÇÃO DE HIPERPARÂMETROS
        print("\n🔍 ETAPA 5: Otimização de Hiperparâmetros")
        modelo_otimizado = otimizar_hiperparametros(X_train_selected, y_train)
        
        # 6. VALIDAÇÃO CRUZADA RIGOROSA
        print("\n🔍 ETAPA 6: Validação Cruzada Rigorosa")
        X_all = dataset[features_selecionadas]
        y_all = dataset['Target']
        
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X_all)):
            X_train_cv = X_all.iloc[train_idx]
            X_test_cv = X_all.iloc[test_idx]
            y_train_cv = y_all.iloc[train_idx]
            y_test_cv = y_all.iloc[test_idx]
            
            # Usar modelo otimizado
            modelo_cv = criar_ensemble_avancado()
            modelo_cv.fit(X_train_cv, y_train_cv)
            score = accuracy_score(y_test_cv, modelo_cv.predict(X_test_cv))
            cv_scores.append(score)
            print(f"   Fold {fold+1}: {score:.1%}")
        
        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)
        
        # 7. TESTE FINAL
        print("\n🎯 ETAPA 7: Teste Final")
        y_pred = modelo_otimizado.predict(X_test_selected)
        acuracia_holdout = accuracy_score(y_test, y_pred)
        
        baseline = max(y_test.mean(), 1 - y_test.mean())
        
        # 8. RELATÓRIO FINAL
        print(f"\n" + "="*80)
        print(f"🏆 RELATÓRIO FINAL - MACHINE LEARNING AVANÇADO")
        print(f"="*80)
        
        print(f"🎯 ACURÁCIA CV: {cv_mean:.1%} ± {cv_std:.1%}")
        print(f"🎯 ACURÁCIA HOLDOUT: {acuracia_holdout:.1%}")
        print(f"📊 BASELINE: {baseline:.1%}")
        print(f"🔧 FEATURES SELECIONADAS: {len(features_selecionadas)}")
        print(f"🤖 MODELO: Ensemble (LR + RF + GB)")
        
        # Usar CV como métrica principal
        acuracia_final = cv_mean
        
        if baseline < acuracia_final:
            melhoria = (acuracia_final - baseline) * 100
            print(f"📈 MELHORIA: +{melhoria:.1f} pontos percentuais")
        
        # Status das metas
        print(f"\n🎯 STATUS DAS METAS:")
        print(f"   Meta 75%: {'✅ ATINGIDA!' if acuracia_final >= 0.75 else f'❌ Faltam {(0.75 - acuracia_final)*100:.1f} pontos'}")
        print(f"   Meta 70%: {'✅ ATINGIDA!' if acuracia_final >= 0.70 else f'❌ Faltam {(0.70 - acuracia_final)*100:.1f} pontos'}")
        print(f"   Meta 60%: {'✅ ATINGIDA!' if acuracia_final >= 0.60 else f'❌ Faltam {(0.60 - acuracia_final)*100:.1f} pontos'}")
        
        # Features mais importantes
        print(f"\n🔧 TOP FEATURES SELECIONADAS:")
        for i, feature in enumerate(features_selecionadas[:10], 1):
            print(f"   {i:2d}. {feature}")
        
        print(f"\n💡 TÉCNICAS APLICADAS:")
        print(f"   ✅ Feature Engineering Avançado ({len(feature_cols)} → {len(features_selecionadas)})")
        print(f"   ✅ Ensemble Learning (3 modelos)")
        print(f"   ✅ Hyperparameter Tuning (GridSearch)")
        print(f"   ✅ Feature Selection (MI + RFE + Importance)")
        print(f"   ✅ Validação Temporal Rigorosa")
        
        if acuracia_final >= 0.60:
            print(f"\n🏆 EXCELENTE! Pipeline avançado com resultado sólido!")
        elif acuracia_final >= 0.55:
            print(f"\n📊 BOM! Base sólida para refinamentos futuros")
        else:
            print(f"\n📊 Resultado obtido: {acuracia_final:.1%}")
        
        return {
            'acuracia_final': acuracia_final,
            'cv_score': cv_mean,
            'cv_std': cv_std,
            'holdout_score': acuracia_holdout,
            'features': features_selecionadas,
            'baseline': baseline,
            'modelo': modelo_otimizado,
            'meta_75': acuracia_final >= 0.75,
            'meta_70': acuracia_final >= 0.70,
            'meta_60': acuracia_final >= 0.60
        }
        
    except Exception as e:
        print(f"\n❌ ERRO: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("🚀 MACHINE LEARNING AVANÇADO")
    print("="*80)
    
    resultado = pipeline_ml_avancado()
    
    if resultado:
        print(f"\n" + "="*80)
        print(f"🎯 RESUMO EXECUTIVO - VERSÃO AVANÇADA")
        print(f"="*80)
        print(f"📊 ACURÁCIA: {resultado['acuracia_final']:.1%}")
        print(f"🔧 FEATURES: {len(resultado['features'])}")
        print(f"📈 DESVIO: ±{resultado['cv_std']:.1%}")
        
        if resultado['acuracia_final'] >= 0.65:
            print(f"🏆 STATUS: ✅ RESULTADO EXCELENTE!")
        elif resultado['acuracia_final'] >= 0.60:
            print(f"🏆 STATUS: ✅ RESULTADO SÓLIDO!")
        elif resultado['acuracia_final'] >= 0.55:
            print(f"🏆 STATUS: 📊 RESULTADO PROMISSOR")
        else:
            print(f"📊 STATUS: CONTINUAR REFINANDO")
        
        print(f"\n✅ PIPELINE AVANÇADO CONCLUÍDO!")
    else:
        print("\n❌ Falha no pipeline avançado")
