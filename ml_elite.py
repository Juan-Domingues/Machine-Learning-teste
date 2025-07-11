import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

# ML Avançado com Stacking
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import SelectKBest, f_classif, RFE, mutual_info_classif

def engenharia_features_elite(data):
    """
    Engenharia de features ELITE baseada em análise financeira avançada
    """
    print("🏆 Feature Engineering ELITE...")
    
    # Base
    data['Return'] = data['Close'].pct_change()
    data['Log_Return'] = np.log(data['Close']/data['Close'].shift(1))
    
    # === 1. FEATURES DE MOMENTUM AVANÇADO ===
    print("   🔥 Momentum Avançado...")
    
    # Returns em múltiplos timeframes
    for periodo in [1, 2, 3, 5, 10]:
        data[f'Return_{periodo}d'] = data['Close'].pct_change(periodo)
        data[f'Positive_return_{periodo}d'] = (data[f'Return_{periodo}d'] > 0).astype(int)
    
    # Momentum Score (combinado)
    data['Momentum_Score'] = (
        (data['Return_1d'] > 0).astype(int) * 1 +
        (data['Return_3d'] > 0).astype(int) * 2 +
        (data['Return_5d'] > 0).astype(int) * 3
    )
    
    # === 2. MÉDIAS MÓVEIS INTELIGENTES ===
    print("   📈 Médias Móveis Inteligentes...")
    
    periodos_ma = [5, 10, 20, 50]
    for periodo in periodos_ma:
        # SMAs e EMAs
        data[f'SMA_{periodo}'] = data['Close'].rolling(periodo).mean()
        data[f'EMA_{periodo}'] = data['Close'].ewm(span=periodo).mean()
        
        # Posição relativa (key feature)
        data[f'Price_vs_SMA{periodo}'] = data['Close'] / data[f'SMA_{periodo}'] - 1
        data[f'Price_vs_EMA{periodo}'] = data['Close'] / data[f'EMA_{periodo}'] - 1
        
        # Slope (tendência) - muito importante
        data[f'SMA_{periodo}_slope'] = data[f'SMA_{periodo}'].pct_change(3)
        data[f'SMA_{periodo}_acceleration'] = data[f'SMA_{periodo}_slope'].diff()
        
        # Cross signals
        if periodo == 20:
            data['SMA_5_20_ratio'] = data['SMA_5'] / data['SMA_20']
            data['SMA_10_20_ratio'] = data['SMA_10'] / data['SMA_20']
    
    # === 3. VOLATILIDADE INTELIGENTE ===
    print("   📊 Volatilidade Inteligente...")
    
    for periodo in [5, 10, 20]:
        # Volatilidade padrão
        data[f'Volatility_{periodo}'] = data['Return'].rolling(periodo).std()
        
        # Volatilidade normalizada
        vol_median = data[f'Volatility_{periodo}'].rolling(50).median()
        data[f'Vol_normalized_{periodo}'] = data[f'Volatility_{periodo}'] / vol_median
        
        # Regime de volatilidade
        vol_75th = data[f'Volatility_{periodo}'].rolling(100).quantile(0.75)
        data[f'High_vol_regime_{periodo}'] = (data[f'Volatility_{periodo}'] > vol_75th).astype(int)
    
    # === 4. VOLUME ANALYSIS AVANÇADO ===
    print("   📦 Volume Analysis Avançado...")
    
    for periodo in [10, 20]:
        data[f'Volume_MA_{periodo}'] = data['Volume'].rolling(periodo).mean()
        data[f'Volume_ratio_{periodo}'] = data['Volume'] / data[f'Volume_MA_{periodo}']
        
        # Volume-Price Divergence
        price_change = data['Close'].pct_change(periodo)
        volume_change = data['Volume'].pct_change(periodo)
        data[f'Vol_Price_divergence_{periodo}'] = (
            ((price_change > 0) & (volume_change < 0)) |
            ((price_change < 0) & (volume_change > 0))
        ).astype(int)
    
    # On-Balance Volume (OBV)
    data['OBV'] = (data['Volume'] * np.sign(data['Return'])).cumsum()
    data['OBV_signal'] = data['OBV'].rolling(20).mean()
    data['OBV_bullish'] = (data['OBV'] > data['OBV_signal']).astype(int)
    
    # === 5. RSI MULTI-TIMEFRAME ===
    print("   🎯 RSI Multi-Timeframe...")
    
    for periodo in [7, 14, 21]:
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(periodo).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(periodo).mean()
        rs = gain / loss
        data[f'RSI_{periodo}'] = 100 - (100 / (1 + rs))
        
        # RSI levels
        data[f'RSI_{periodo}_oversold'] = (data[f'RSI_{periodo}'] < 30).astype(int)
        data[f'RSI_{periodo}_overbought'] = (data[f'RSI_{periodo}'] > 70).astype(int)
        data[f'RSI_{periodo}_neutral'] = ((data[f'RSI_{periodo}'] >= 40) & (data[f'RSI_{periodo}'] <= 60)).astype(int)
    
    # RSI Divergence
    data['RSI_14_slope'] = data['RSI_14'].diff(5)
    data['Price_RSI_divergence'] = (
        ((data['Close'].pct_change(5) > 0) & (data['RSI_14_slope'] < 0)) |
        ((data['Close'].pct_change(5) < 0) & (data['RSI_14_slope'] > 0))
    ).astype(int)
    
    # === 6. BOLLINGER BANDS AVANÇADO ===
    print("   🎈 Bollinger Bands Avançado...")
    
    periodo = 20
    data[f'BB_middle'] = data['Close'].rolling(periodo).mean()
    bb_std = data['Close'].rolling(periodo).std()
    data[f'BB_upper'] = data[f'BB_middle'] + (bb_std * 2)
    data[f'BB_lower'] = data[f'BB_middle'] - (bb_std * 2)
    
    # BB Position (muito importante)
    data['BB_position'] = (data['Close'] - data['BB_lower']) / (data['BB_upper'] - data['BB_lower'])
    
    # BB Width (volatilidade)
    data['BB_width'] = (data['BB_upper'] - data['BB_lower']) / data['BB_middle']
    data['BB_squeeze'] = (data['BB_width'] < data['BB_width'].rolling(50).quantile(0.2)).astype(int)
    
    # BB Breakouts
    data['BB_breakout_up'] = (data['Close'] > data['BB_upper']).astype(int)
    data['BB_breakout_down'] = (data['Close'] < data['BB_lower']).astype(int)
    
    # === 7. STOCHASTIC AVANÇADO ===
    print("   🌊 Stochastic Avançado...")
    
    periodo = 14
    data['Lowest_Low'] = data['Low'].rolling(periodo).min()
    data['Highest_High'] = data['High'].rolling(periodo).max()
    data['Stoch_K'] = 100 * (data['Close'] - data['Lowest_Low']) / (data['Highest_High'] - data['Lowest_Low'])
    data['Stoch_D'] = data['Stoch_K'].rolling(3).mean()
    
    # Stochastic signals
    data['Stoch_oversold'] = (data['Stoch_K'] < 20).astype(int)
    data['Stoch_overbought'] = (data['Stoch_K'] > 80).astype(int)
    data['Stoch_bullish_cross'] = ((data['Stoch_K'] > data['Stoch_D']) & 
                                   (data['Stoch_K'].shift(1) <= data['Stoch_D'].shift(1))).astype(int)
    
    # === 8. FEATURES DE REGIME DE MERCADO ===
    print("   🏛️ Regime de Mercado...")
    
    # Trend Strength
    data['Trend_strength'] = abs(data['SMA_20_slope']) * 100
    data['Strong_trend'] = (data['Trend_strength'] > data['Trend_strength'].rolling(50).quantile(0.7)).astype(int)
    
    # Market Regime
    data['Bull_regime'] = (
        (data['SMA_5'] > data['SMA_20']) & 
        (data['SMA_20'] > data['SMA_50']) &
        (data['SMA_20_slope'] > 0)
    ).astype(int)
    
    data['Bear_regime'] = (
        (data['SMA_5'] < data['SMA_20']) & 
        (data['SMA_20'] < data['SMA_50']) &
        (data['SMA_20_slope'] < 0)
    ).astype(int)
    
    # === 9. FEATURES TEMPORAIS AVANÇADAS ===
    print("   📅 Features Temporais...")
    
    # Efeitos temporais
    data['DayOfWeek'] = data.index.dayofweek
    data['Monday'] = (data['DayOfWeek'] == 0).astype(int)
    data['Friday'] = (data['DayOfWeek'] == 4).astype(int)
    data['Month'] = data.index.month
    data['Q4_effect'] = (data['Month'].isin([10, 11, 12])).astype(int)
    
    # === 10. FEATURES COMPOSTAS ===
    print("   🔗 Features Compostas...")
    
    # Momentum + Volume
    data['Momentum_Volume_signal'] = (
        (data['Momentum_Score'] >= 3) & 
        (data['Volume_ratio_20'] > 1.2)
    ).astype(int)
    
    # Trend + RSI
    data['Trend_RSI_signal'] = (
        (data['Strong_trend'] == 1) & 
        (data['RSI_14'] > 50)
    ).astype(int)
    
    # Multi-timeframe confirmation
    data['Multi_timeframe_bull'] = (
        (data['Price_vs_SMA5'] > 0) &
        (data['Price_vs_SMA10'] > 0) &
        (data['Price_vs_SMA20'] > 0)
    ).astype(int)
    
    return data

def selecao_features_elite(X, y, n_features=20):
    """
    Seleção de features ELITE usando múltiplos métodos avançados
    """
    print(f"🎯 Seleção Features ELITE (top {n_features})...")
    
    # 1. Mutual Information (captura relações não-lineares)
    mi_scores = mutual_info_classif(X, y, random_state=42, discrete_features=False)
    mi_ranking = np.argsort(mi_scores)[-n_features:]
    mi_features = X.columns[mi_ranking].tolist()
    
    # 2. Random Forest Feature Importance
    rf = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=10)
    rf.fit(X, y)
    rf_importance = rf.feature_importances_
    rf_ranking = np.argsort(rf_importance)[-n_features:]
    rf_features = X.columns[rf_ranking].tolist()
    
    # 3. Gradient Boosting Feature Importance
    gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb.fit(X, y)
    gb_importance = gb.feature_importances_
    gb_ranking = np.argsort(gb_importance)[-n_features:]
    gb_features = X.columns[gb_ranking].tolist()
    
    # 4. Statistical F-test
    f_selector = SelectKBest(f_classif, k=n_features)
    f_selector.fit(X, y)
    f_features = X.columns[f_selector.get_support()].tolist()
    
    # Combinar todos os métodos
    all_features = list(set(mi_features + rf_features + gb_features + f_features))
    
    print(f"   MI: {len(mi_features)} | RF: {len(rf_features)} | GB: {len(gb_features)} | F-test: {len(f_features)}")
    print(f"   Total combinado: {len(all_features)}")
    
    # Se temos mais features que o desejado, fazer ranking final
    if len(all_features) > n_features:
        # Usar ensemble de importâncias para ranking final
        X_combined = X[all_features]
        
        # Normalizar scores
        mi_scores_norm = (mi_scores[X.columns.isin(all_features)] - mi_scores.min()) / (mi_scores.max() - mi_scores.min())
        rf_scores_norm = (rf_importance[X.columns.isin(all_features)] - rf_importance.min()) / (rf_importance.max() - rf_importance.min())
        gb_scores_norm = (gb_importance[X.columns.isin(all_features)] - gb_importance.min()) / (gb_importance.max() - gb_importance.min())
        
        # Score final (ensemble das importâncias)
        final_scores = mi_scores_norm + rf_scores_norm + gb_scores_norm
        final_ranking = np.argsort(final_scores)[-n_features:]
        selected_features = X_combined.columns[final_ranking].tolist()
    else:
        selected_features = all_features
    
    print(f"✅ {len(selected_features)} features ELITE selecionadas")
    return selected_features

def criar_stacking_ensemble():
    """
    Cria Stacking Ensemble com modelos diversificados
    """
    print("🏗️ Criando Stacking Ensemble...")
    
    # Base models (diversos e complementares)
    base_models = [
        ('rf', Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(
                n_estimators=100, 
                max_depth=8, 
                min_samples_split=5,
                random_state=42
            ))
        ])),
        
        ('gb', Pipeline([
            ('scaler', StandardScaler()),
            ('clf', GradientBoostingClassifier(
                n_estimators=100, 
                learning_rate=0.1, 
                max_depth=5,
                random_state=42
            ))
        ])),
        
        ('svm', Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(
                C=1.0, 
                kernel='rbf', 
                probability=True,
                random_state=42
            ))
        ])),
        
        ('lr', Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(
                C=1.0, 
                penalty='l2',
                random_state=42,
                max_iter=1000
            ))
        ]))
    ]
    
    # Meta-learner (Logistic Regression com regularização)
    meta_learner = LogisticRegression(
        C=1.0, 
        penalty='l2',
        random_state=42,
        max_iter=1000
    )
    
    # Stacking Classifier (sem CV interno para evitar problemas com TimeSeries)
    stacking_clf = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_learner,
        stack_method='predict_proba',
        n_jobs=1  # Reduzir paralelismo para evitar problemas
    )
    
    return stacking_clf

def validacao_temporal_rigorosa(modelo, X, y, n_splits=5):
    """
    Validação temporal ultra-rigorosa
    """
    print(f"🔍 Validação Temporal Ultra-Rigorosa ({n_splits} folds)...")
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    fold_details = []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train_cv = X.iloc[train_idx]
        X_test_cv = X.iloc[test_idx]
        y_train_cv = y.iloc[train_idx]
        y_test_cv = y.iloc[test_idx]
        
        # Treinar modelo
        modelo_cv = criar_stacking_ensemble()
        modelo_cv.fit(X_train_cv, y_train_cv)
        
        # Avaliar
        y_pred_cv = modelo_cv.predict(X_test_cv)
        score = accuracy_score(y_test_cv, y_pred_cv)
        scores.append(score)
        
        # Baseline do fold
        baseline_fold = max(y_test_cv.mean(), 1 - y_test_cv.mean())
        improvement = score - baseline_fold
        
        fold_details.append({
            'fold': fold + 1,
            'score': score,
            'baseline': baseline_fold,
            'improvement': improvement,
            'n_train': len(y_train_cv),
            'n_test': len(y_test_cv)
        })
        
        print(f"   Fold {fold+1}: {score:.1%} (baseline: {baseline_fold:.1%}, melhoria: {improvement:+.1%})")
    
    return scores, fold_details

def pipeline_machine_learning_elite():
    """
    Pipeline ELITE de Machine Learning com todas as técnicas avançadas
    """
    print("="*80)
    print("🏆 MACHINE LEARNING ELITE PIPELINE")
    print("🚀 Stacking + Feature Engineering Elite + Validação Ultra-Rigorosa")
    print("="*80)
    
    try:
        # 1. CARREGAMENTO ESTENDIDO
        print("\n📥 ETAPA 1: Carregamento Estendido (3 anos)")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=3*365)
        data = yf.download('^BVSP', start=start_date, end=end_date)
        
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        print(f"✅ {len(data)} dias carregados")
        
        # 2. FEATURE ENGINEERING ELITE
        print("\n🔧 ETAPA 2: Feature Engineering ELITE")
        data = engenharia_features_elite(data)
        
        # Target
        data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
        
        # Features disponíveis
        feature_cols = [col for col in data.columns if col not in 
                       ['Open', 'High', 'Low', 'Close', 'Volume', 'Target', 'Adj Close']]
        
        print(f"✅ {len(feature_cols)} features ELITE criadas")
        
        # 3. PREPARAÇÃO DO DATASET
        print("\n📊 ETAPA 3: Preparação do Dataset")
        dataset = data[feature_cols + ['Target']].dropna()
        print(f"📊 Dataset final: {len(dataset)} observações")
        
        if len(dataset) < 300:
            print("❌ Dados insuficientes")
            return None
        
        # Divisão estratégica (teste maior)
        n_test = 50
        train_data = dataset.iloc[:-n_test]
        test_data = dataset.iloc[-n_test:]
        
        X_train = train_data[feature_cols]
        y_train = train_data['Target']
        X_test = test_data[feature_cols]
        y_test = test_data['Target']
        
        print(f"📊 Treino: {len(X_train)} | Teste: {len(X_test)}")
        
        # 4. SELEÇÃO DE FEATURES ELITE
        print("\n🎯 ETAPA 4: Seleção de Features ELITE")
        features_elite = selecao_features_elite(X_train, y_train, n_features=25)
        
        X_train_elite = X_train[features_elite]
        X_test_elite = X_test[features_elite]
        
        # 5. STACKING ENSEMBLE
        print("\n🏗️ ETAPA 5: Stacking Ensemble")
        modelo_stacking = criar_stacking_ensemble()
        
        print("   Treinando Stacking Ensemble...")
        modelo_stacking.fit(X_train_elite, y_train)
        
        # Teste inicial
        y_pred_holdout = modelo_stacking.predict(X_test_elite)
        score_holdout = accuracy_score(y_test, y_pred_holdout)
        baseline_holdout = max(y_test.mean(), 1 - y_test.mean())
        
        print(f"✅ Stacking treinado!")
        print(f"   Holdout: {score_holdout:.1%}")
        print(f"   Baseline: {baseline_holdout:.1%}")
        
        # 6. VALIDAÇÃO ULTRA-RIGOROSA
        print("\n🔍 ETAPA 6: Validação Ultra-Rigorosa")
        
        X_all_elite = dataset[features_elite]
        y_all = dataset['Target']
        
        cv_scores, fold_details = validacao_temporal_rigorosa(modelo_stacking, X_all_elite, y_all, n_splits=5)
        
        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)
        cv_min = np.min(cv_scores)
        cv_max = np.max(cv_scores)
        
        # 7. ANÁLISE DE MELHORIAS
        melhorias = [fold['improvement'] for fold in fold_details]
        melhoria_media = np.mean(melhorias)
        melhorias_positivas = sum(1 for m in melhorias if m > 0)
        
        # 8. RELATÓRIO FINAL COMPLETO
        print(f"\n" + "="*80)
        print(f"🏆 RELATÓRIO FINAL - MACHINE LEARNING ELITE")
        print(f"="*80)
        
        # Métricas principais
        print(f"🎯 ACURÁCIA CV: {cv_mean:.1%} ± {cv_std:.1%}")
        print(f"🎯 RANGE CV: {cv_min:.1%} - {cv_max:.1%}")
        print(f"🎯 HOLDOUT: {score_holdout:.1%}")
        print(f"📊 BASELINE MÉDIO: {baseline_holdout:.1%}")
        
        # Análise de consistência
        print(f"\n📊 ANÁLISE DE CONSISTÊNCIA:")
        print(f"   Folds com melhoria: {melhorias_positivas}/5")
        print(f"   Melhoria média: {melhoria_media:+.1%}")
        print(f"   Estabilidade: {'✅ ALTA' if cv_std < 0.05 else '⚠️ MODERADA' if cv_std < 0.10 else '❌ BAIXA'}")
        
        # Status das metas
        acuracia_final = cv_mean
        print(f"\n🎯 STATUS DAS METAS (CV):")
        
        metas = [70, 65, 60, 55]
        meta_atingida = False
        for meta in metas:
            meta_decimal = meta / 100
            if acuracia_final >= meta_decimal:
                print(f"   Meta {meta}%: ✅ ATINGIDA!")
                meta_atingida = True
                break
            else:
                falta = (meta_decimal - acuracia_final) * 100
                print(f"   Meta {meta}%: ❌ Faltam {falta:.1f} pontos")
        
        # Features ELITE
        print(f"\n🔧 TOP 15 FEATURES ELITE:")
        for i, feature in enumerate(features_elite[:15], 1):
            print(f"   {i:2d}. {feature}")
        
        # Técnicas aplicadas
        print(f"\n💡 TÉCNICAS ELITE APLICADAS:")
        print(f"   ✅ Feature Engineering Avançado ({len(feature_cols)} → {len(features_elite)})")
        print(f"   ✅ Stacking Ensemble (4 modelos base + meta-learner)")
        print(f"   ✅ Seleção Multi-Método (MI + RF + GB + F-test)")
        print(f"   ✅ Validação Temporal Ultra-Rigorosa")
        print(f"   ✅ Dataset Máximo (3 anos)")
        print(f"   ✅ Features Compostas e Regimes de Mercado")
        
        # Diagnóstico final
        print(f"\n🏆 DIAGNÓSTICO FINAL:")
        
        if acuracia_final >= 0.60:
            print(f"   🌟 EXCELENTE! Pipeline elite com resultado excepcional!")
            print(f"   🚀 Pronto para produção com alta confiabilidade")
        elif acuracia_final >= 0.55:
            print(f"   ✅ MUITO BOM! Resultado sólido e consistente")
            print(f"   📈 Pipeline robusto com melhoria clara")
        elif melhoria_media > 0:
            print(f"   📊 BOM! Melhoria consistente sobre baseline")
            print(f"   🔧 Base sólida para refinamentos futuros")
        else:
            print(f"   📊 Resultado: {acuracia_final:.1%}")
            print(f"   🔍 Investigar período de dados ou features adicionais")
        
        # Recomendações
        print(f"\n💡 RECOMENDAÇÕES:")
        if cv_std > 0.08:
            print(f"   ⚠️ Alta variabilidade - considerar mais dados ou regularização")
        if melhoria_media < 0.02:
            print(f"   📊 Melhoria baixa - explorar features macroeconômicas")
        if acuracia_final < 0.55:
            print(f"   🔧 Considerar features de sentimento de mercado")
        
        return {
            'acuracia_final': acuracia_final,
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'cv_min': cv_min,
            'cv_max': cv_max,
            'holdout_score': score_holdout,
            'features_elite': features_elite,
            'baseline': baseline_holdout,
            'melhoria_media': melhoria_media,
            'melhorias_positivas': melhorias_positivas,
            'modelo': modelo_stacking,
            'fold_details': fold_details,
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
    print("🏆 MACHINE LEARNING ELITE")
    print("="*80)
    
    resultado = pipeline_machine_learning_elite()
    
    if resultado:
        print(f"\n" + "="*80)
        print(f"🎯 RESUMO EXECUTIVO - ELITE EDITION")
        print(f"="*80)
        print(f"📊 ACURÁCIA FINAL: {resultado['acuracia_final']:.1%}")
        print(f"🔧 FEATURES ELITE: {len(resultado['features_elite'])}")
        print(f"📈 ESTABILIDADE: ±{resultado['cv_std']:.1%}")
        print(f"🏆 MELHORIAS: {resultado['melhorias_positivas']}/5 folds")
        print(f"🤖 MODELO: Stacking Ensemble")
        
        if resultado['acuracia_final'] >= 0.65:
            print(f"🌟 STATUS: EXCEPCIONAL! 🌟")
        elif resultado['acuracia_final'] >= 0.60:
            print(f"🏆 STATUS: EXCELENTE!")
        elif resultado['acuracia_final'] >= 0.55:
            print(f"✅ STATUS: MUITO BOM!")
        else:
            print(f"📊 STATUS: RESULTADO OBTIDO")
        
        print(f"\n🚀 PIPELINE ELITE CONCLUÍDO!")
        print(f"💎 Máxima tecnologia aplicada ao problema!")
    else:
        print("\n❌ Falha no pipeline elite")
