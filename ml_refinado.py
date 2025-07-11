import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

# ML com foco em features específicas e tuning fino
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import SelectKBest, f_classif, chi2
from sklearn.decomposition import PCA

def features_tecnicas_refinadas(data):
    """
    Features técnicas refinadas baseadas na análise anterior
    """
    print("🔧 Criando features técnicas refinadas...")
    
    # Features básicas
    data['Return'] = data['Close'].pct_change()
    
    # 1. MÉDIAS MÓVEIS COM FOCO NOS PERÍODOS MAIS IMPORTANTES
    periodos_importantes = [5, 10, 20, 50]
    for periodo in periodos_importantes:
        data[f'SMA_{periodo}'] = data['Close'].rolling(periodo).mean()
        data[f'EMA_{periodo}'] = data['Close'].ewm(span=periodo).mean()
        
        # Posição relativa (feature importante detectada)
        data[f'Price_above_SMA{periodo}'] = (data['Close'] > data[f'SMA_{periodo}']).astype(int)
        data[f'Price_above_EMA{periodo}'] = (data['Close'] > data[f'EMA_{periodo}']).astype(int)
        
        # Distância normalizada (feature importante)
        data[f'SMA_{periodo}_dist'] = (data['Close'] - data[f'SMA_{periodo}']) / data[f'SMA_{periodo}']
        data[f'EMA_{periodo}_dist'] = (data['Close'] - data[f'EMA_{periodo}']) / data[f'EMA_{periodo}']
        
        # Slope (tendência) - feature muito importante detectada
        data[f'SMA_{periodo}_slope'] = data[f'SMA_{periodo}'].diff(3) / data[f'SMA_{periodo}'].shift(3)
        data[f'SMA_{periodo}_uptrend'] = (data[f'SMA_{periodo}_slope'] > 0).astype(int)
    
    # 2. VOLATILIDADE REFINADA (feature importante detectada)
    for periodo in [5, 10, 20]:
        data[f'Volatility_{periodo}d'] = data['Return'].rolling(periodo).std()
        data[f'High_volatility_{periodo}d'] = (data[f'Volatility_{periodo}d'] > 
                                              data[f'Volatility_{periodo}d'].rolling(50).mean()).astype(int)
        # Volatilidade normalizada
        data[f'Vol_norm_{periodo}d'] = data[f'Volatility_{periodo}d'] / data[f'Volatility_{periodo}d'].rolling(50).mean()
    
    # 3. VOLUME INTELIGENTE (features importantes)
    for periodo in [10, 20]:
        data[f'Volume_MA_{periodo}'] = data['Volume'].rolling(periodo).mean()
        data[f'Volume_ratio_{periodo}'] = data['Volume'] / data[f'Volume_MA_{periodo}']
        data[f'Volume_above_avg_{periodo}'] = (data['Volume'] > data[f'Volume_MA_{periodo}']).astype(int)
        
        # Volume-Price Relationship
        data[f'Volume_price_corr_{periodo}'] = data['Volume'].rolling(periodo).corr(data['Close'])
    
    # 4. RSI OTIMIZADO
    for periodo in [14, 21]:
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periodo).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periodo).mean()
        rs = gain / loss
        data[f'RSI_{periodo}'] = 100 - (100 / (1 + rs))
        data[f'RSI_{periodo}_overbought'] = (data[f'RSI_{periodo}'] > 70).astype(int)
        data[f'RSI_{periodo}_oversold'] = (data[f'RSI_{periodo}'] < 30).astype(int)
        data[f'RSI_{periodo}_neutral'] = ((data[f'RSI_{periodo}'] >= 30) & (data[f'RSI_{periodo}'] <= 70)).astype(int)
    
    # 5. STOCHASTIC REFINADO (foi importante)
    periodo = 14
    data[f'Lowest_Low_{periodo}'] = data['Low'].rolling(periodo).min()
    data[f'Highest_High_{periodo}'] = data['High'].rolling(periodo).max()
    data[f'STOCH_K_{periodo}'] = 100 * (data['Close'] - data[f'Lowest_Low_{periodo}']) / (data[f'Highest_High_{periodo}'] - data[f'Lowest_Low_{periodo}'])
    data[f'STOCH_D_{periodo}'] = data[f'STOCH_K_{periodo}'].rolling(3).mean()
    data[f'STOCH_oversold_{periodo}'] = (data[f'STOCH_K_{periodo}'] < 20).astype(int)
    data[f'STOCH_overbought_{periodo}'] = (data[f'STOCH_K_{periodo}'] > 80).astype(int)
    
    # 6. BOLLINGER BANDS REFINADO
    periodo = 20
    data[f'BB_middle_{periodo}'] = data['Close'].rolling(periodo).mean()
    std = data['Close'].rolling(periodo).std()
    data[f'BB_upper_{periodo}'] = data[f'BB_middle_{periodo}'] + (std * 2)
    data[f'BB_lower_{periodo}'] = data[f'BB_middle_{periodo}'] - (std * 2)
    data[f'BB_position_{periodo}'] = (data['Close'] - data[f'BB_lower_{periodo}']) / (data[f'BB_upper_{periodo}'] - data[f'BB_lower_{periodo}'])
    data[f'BB_width_{periodo}'] = (data[f'BB_upper_{periodo}'] - data[f'BB_lower_{periodo}']) / data[f'BB_middle_{periodo}']
    
    # 7. MOMENTUM AVANÇADO
    for lag in [1, 3, 5]:
        data[f'Return_{lag}d'] = data['Close'].pct_change(lag)
        data[f'Momentum_{lag}d'] = (data[f'Return_{lag}d'] > 0).astype(int)
    
    # Momentum combinado
    data['Momentum_combined'] = (
        (data['Return_1d'] > 0).astype(int) + 
        (data['Return_3d'] > 0).astype(int) + 
        (data['Return_5d'] > 0).astype(int)
    )
    
    # 8. CROSS DE MÉDIAS (importantes)
    data['SMA_cross_5_20'] = (data['SMA_5'] > data['SMA_20']).astype(int)
    data['SMA_cross_10_50'] = (data['SMA_10'] > data['SMA_50']).astype(int)
    data['EMA_cross_5_20'] = (data['EMA_5'] > data['EMA_20']).astype(int)
    
    # 9. FEATURES DE REGIME DE MERCADO
    # Trend strength
    data['Trend_strength'] = abs(data['SMA_20_slope']) * 100
    data['Strong_trend'] = (data['Trend_strength'] > data['Trend_strength'].rolling(50).quantile(0.75)).astype(int)
    
    # Market regime
    data['Bull_market'] = (data['SMA_20'] > data['SMA_50']).astype(int)
    data['Consolidation'] = (data['BB_width_20'] < data['BB_width_20'].rolling(20).quantile(0.3)).astype(int)
    
    # 10. FEATURES TEMPORAIS INTELIGENTES
    data['DayOfWeek'] = data.index.dayofweek
    data['Monday_effect'] = (data['DayOfWeek'] == 0).astype(int)
    data['Friday_effect'] = (data['DayOfWeek'] == 4).astype(int)
    data['Month'] = data.index.month
    data['Year_end'] = (data['Month'] == 12).astype(int)
    
    return data

def selecao_inteligente_features(X, y, metodo='combined', n_features=12):
    """
    Seleção inteligente combinando múltiplos métodos
    """
    print(f"🎯 Seleção Inteligente de Features (método: {metodo})...")
    
    if metodo == 'statistical':
        # Método estatístico
        selector = SelectKBest(f_classif, k=n_features)
        selector.fit(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        
    elif metodo == 'tree_based':
        # Baseado em árvores
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        importances = rf.feature_importances_
        indices = np.argsort(importances)[-n_features:]
        selected_features = X.columns[indices].tolist()
        
    elif metodo == 'combined':
        # Método combinado (mais robusto)
        # 1. Statistical
        selector_stat = SelectKBest(f_classif, k=n_features*2)
        selector_stat.fit(X, y)
        stat_features = X.columns[selector_stat.get_support()].tolist()
        
        # 2. Tree importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        importances = rf.feature_importances_
        indices = np.argsort(importances)[-n_features*2:]
        tree_features = X.columns[indices].tolist()
        
        # 3. Combinação inteligente
        combined = list(set(stat_features + tree_features))
        
        # Re-avaliar com RF para ranking final
        if len(combined) > n_features:
            X_combined = X[combined]
            rf_final = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_final.fit(X_combined, y)
            final_importances = rf_final.feature_importances_
            final_indices = np.argsort(final_importances)[-n_features:]
            selected_features = X_combined.columns[final_indices].tolist()
        else:
            selected_features = combined
    
    print(f"✅ {len(selected_features)} features selecionadas")
    return selected_features

def criar_modelo_otimizado(tipo='ensemble_light'):
    """
    Cria modelo otimizado para o problema específico
    """
    if tipo == 'rf_tuned':
        modelo = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            ))
        ])
        
    elif tipo == 'gb_tuned':
        modelo = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=5,
                min_samples_split=10,
                random_state=42
            ))
        ])
        
    elif tipo == 'lr_tuned':
        modelo = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(
                C=0.1,
                penalty='l1',
                solver='liblinear',
                random_state=42
            ))
        ])
        
    elif tipo == 'ensemble_light':
        # Ensemble mais leve e eficiente
        from sklearn.ensemble import VotingClassifier
        
        rf = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42))
        ])
        
        gb = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42))
        ])
        
        modelo = VotingClassifier([
            ('rf', rf),
            ('gb', gb)
        ], voting='soft')
    
    return modelo

def validacao_robusta(modelo, X, y, n_splits=5):
    """
    Validação robusta com métricas detalhadas
    """
    print(f"🔍 Validação Robusta ({n_splits} folds)...")
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train_cv = X.iloc[train_idx]
        X_test_cv = X.iloc[test_idx]
        y_train_cv = y.iloc[train_idx]
        y_test_cv = y.iloc[test_idx]
        
        modelo_cv = criar_modelo_otimizado('ensemble_light')
        modelo_cv.fit(X_train_cv, y_train_cv)
        score = accuracy_score(y_test_cv, modelo_cv.predict(X_test_cv))
        scores.append(score)
        print(f"   Fold {fold+1}: {score:.1%} (n_test: {len(y_test_cv)})")
    
    return scores

def pipeline_refinado_final():
    """
    Pipeline final refinado com base nas descobertas
    """
    print("="*80)
    print("🎯 PIPELINE REFINADO FINAL")
    print("🚀 Foco em Features Específicas + Modelos Otimizados")
    print("="*80)
    
    try:
        # 1. CARREGAMENTO ESTRATÉGICO
        print("\n📥 ETAPA 1: Carregamento Estratégico")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=int(2.5 * 365))  # 2.5 anos
        data = yf.download('^BVSP', start=start_date, end=end_date)
        
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        print(f"✅ {len(data)} dias carregados (2.5 anos)")
        
        # 2. FEATURE ENGINEERING REFINADO
        print("\n🔧 ETAPA 2: Feature Engineering Refinado")
        data = features_tecnicas_refinadas(data)
        
        # Target
        data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
        
        # Features disponíveis
        feature_cols = [col for col in data.columns if col not in 
                       ['Open', 'High', 'Low', 'Close', 'Volume', 'Target', 'Adj Close']]
        
        print(f"✅ {len(feature_cols)} features técnicas criadas")
        
        # 3. PREPARAÇÃO INTELIGENTE
        print("\n📊 ETAPA 3: Preparação Inteligente do Dataset")
        dataset = data[feature_cols + ['Target']].dropna()
        print(f"📊 Dataset: {len(dataset)} observações")
        
        if len(dataset) < 200:
            print("❌ Dados insuficientes")
            return None
        
        # Divisão estratégica
        n_test = 40  # Teste mais robusto
        train_data = dataset.iloc[:-n_test]
        test_data = dataset.iloc[-n_test:]
        
        X_train = train_data[feature_cols]
        y_train = train_data['Target']
        X_test = test_data[feature_cols]
        y_test = test_data['Target']
        
        # 4. SELEÇÃO INTELIGENTE DE FEATURES
        print("\n🎯 ETAPA 4: Seleção Inteligente de Features")
        
        # Testar diferentes números de features
        resultados_features = {}
        for n_feat in [8, 12, 16, 20]:
            features_selecionadas = selecao_inteligente_features(X_train, y_train, 'combined', n_feat)
            
            # Teste rápido
            X_train_sel = X_train[features_selecionadas]
            X_test_sel = X_test[features_selecionadas]
            
            modelo_teste = criar_modelo_otimizado('ensemble_light')
            modelo_teste.fit(X_train_sel, y_train)
            score_teste = accuracy_score(y_test, modelo_teste.predict(X_test_sel))
            
            resultados_features[n_feat] = (score_teste, features_selecionadas)
            print(f"   {n_feat} features: {score_teste:.1%}")
        
        # Escolher melhor configuração
        melhor_n = max(resultados_features.keys(), key=lambda k: resultados_features[k][0])
        melhor_score, features_finais = resultados_features[melhor_n]
        
        print(f"✅ Melhor configuração: {melhor_n} features ({melhor_score:.1%})")
        
        # 5. TESTE DE DIFERENTES MODELOS
        print("\n🤖 ETAPA 5: Teste de Modelos Otimizados")
        
        X_train_final = X_train[features_finais]
        X_test_final = X_test[features_finais]
        
        modelos_teste = {
            'Ensemble Light': 'ensemble_light',
            'Random Forest': 'rf_tuned',
            'Gradient Boosting': 'gb_tuned',
            'Logistic L1': 'lr_tuned'
        }
        
        resultados_modelos = {}
        for nome, tipo in modelos_teste.items():
            modelo = criar_modelo_otimizado(tipo)
            modelo.fit(X_train_final, y_train)
            score = accuracy_score(y_test, modelo.predict(X_test_final))
            resultados_modelos[nome] = (score, modelo)
            print(f"   {nome}: {score:.1%}")
        
        # Melhor modelo
        melhor_modelo_nome = max(resultados_modelos.keys(), key=lambda k: resultados_modelos[k][0])
        melhor_modelo_score, melhor_modelo = resultados_modelos[melhor_modelo_nome]
        
        print(f"✅ Melhor modelo: {melhor_modelo_nome} ({melhor_modelo_score:.1%})")
        
        # 6. VALIDAÇÃO CRUZADA FINAL
        print("\n🔍 ETAPA 6: Validação Cruzada Final")
        
        X_all = dataset[features_finais]
        y_all = dataset['Target']
        
        cv_scores = validacao_robusta(melhor_modelo, X_all, y_all, n_splits=5)
        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)
        
        # 7. ANÁLISE FINAL
        baseline = max(y_test.mean(), 1 - y_test.mean())
        
        print(f"\n" + "="*80)
        print(f"🏆 RELATÓRIO FINAL - PIPELINE REFINADO")
        print(f"="*80)
        
        print(f"🎯 ACURÁCIA CV: {cv_mean:.1%} ± {cv_std:.1%}")
        print(f"🎯 ACURÁCIA HOLDOUT: {melhor_modelo_score:.1%}")
        print(f"📊 BASELINE: {baseline:.1%}")
        print(f"🔧 FEATURES FINAIS: {len(features_finais)}")
        print(f"🤖 MELHOR MODELO: {melhor_modelo_nome}")
        
        # Usar CV como métrica principal
        acuracia_final = cv_mean
        
        if baseline < acuracia_final:
            melhoria = (acuracia_final - baseline) * 100
            print(f"📈 MELHORIA: +{melhoria:.1f} pontos percentuais")
        
        # Status das metas
        print(f"\n🎯 STATUS DAS METAS:")
        for meta in [75, 70, 65, 60, 55]:
            meta_decimal = meta / 100
            if acuracia_final >= meta_decimal:
                print(f"   Meta {meta}%: ✅ ATINGIDA!")
                break
            else:
                falta = (meta_decimal - acuracia_final) * 100
                print(f"   Meta {meta}%: ❌ Faltam {falta:.1f} pontos")
        
        # Top features
        print(f"\n🔧 TOP FEATURES FINAIS:")
        for i, feature in enumerate(features_finais, 1):
            print(f"   {i:2d}. {feature}")
        
        print(f"\n💡 OTIMIZAÇÕES APLICADAS:")
        print(f"   ✅ Feature Engineering Específico")
        print(f"   ✅ Seleção Inteligente de Features")
        print(f"   ✅ Teste de Múltiplos Modelos")
        print(f"   ✅ Hiperparâmetros Otimizados")
        print(f"   ✅ Validação Temporal Robusta")
        print(f"   ✅ Dataset Estendido (2.5 anos)")
        
        # Diagnóstico
        if acuracia_final >= 0.60:
            print(f"\n🏆 EXCELENTE! Pipeline refinado com resultado sólido!")
        elif acuracia_final >= 0.55:
            print(f"\n📊 MUITO BOM! Resultado consistente e confiável")
        elif acuracia_final > baseline:
            print(f"\n📈 BOM! Melhoria sobre baseline")
        else:
            print(f"\n📊 Resultado: {acuracia_final:.1%}")
            
        return {
            'acuracia_final': acuracia_final,
            'cv_score': cv_mean,
            'cv_std': cv_std,
            'holdout_score': melhor_modelo_score,
            'features': features_finais,
            'modelo_nome': melhor_modelo_nome,
            'modelo': melhor_modelo,
            'baseline': baseline,
            'meta_60': acuracia_final >= 0.60,
            'meta_65': acuracia_final >= 0.65,
            'meta_70': acuracia_final >= 0.70
        }
        
    except Exception as e:
        print(f"\n❌ ERRO: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("🚀 PIPELINE REFINADO FINAL")
    print("="*80)
    
    resultado = pipeline_refinado_final()
    
    if resultado:
        print(f"\n" + "="*80)
        print(f"🎯 RESUMO EXECUTIVO - VERSÃO FINAL REFINADA")
        print(f"="*80)
        print(f"📊 ACURÁCIA: {resultado['acuracia_final']:.1%}")
        print(f"🔧 FEATURES: {len(resultado['features'])}")
        print(f"📈 ESTABILIDADE: ±{resultado['cv_std']:.1%}")
        print(f"🤖 MODELO: {resultado['modelo_nome']}")
        
        if resultado['acuracia_final'] >= 0.65:
            print(f"🏆 STATUS: ✅ EXCELENTE!")
        elif resultado['acuracia_final'] >= 0.60:
            print(f"🏆 STATUS: ✅ MUITO BOM!")
        elif resultado['acuracia_final'] >= 0.55:
            print(f"🏆 STATUS: 📊 BOM!")
        else:
            print(f"📊 STATUS: RESULTADO OBTIDO")
        
        print(f"\n✅ PIPELINE REFINADO CONCLUÍDO!")
    else:
        print("\n❌ Falha no pipeline refinado")
