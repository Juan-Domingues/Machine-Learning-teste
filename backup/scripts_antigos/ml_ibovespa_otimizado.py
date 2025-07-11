"""
Machine Learning - IBOVESPA OTIMIZADO
Versão aprimorada focada em atingir 75% de acurácia nos últimos 30 dias
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Scikit-Learn
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import SelectKBest, f_classif, RFE

def carregar_dados_completos():
    """Carrega dados com maior histórico e contexto macroeconômico"""
    print("ETAPA 1: Carregando dados completos...")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=15 * 365)  # 15 anos
    
    # IBOVESPA
    ibov = yf.download('^BVSP', start=start_date, end=end_date)
    if isinstance(ibov.columns, pd.MultiIndex):
        ibov.columns = ibov.columns.get_level_values(0)
    
    # Contexto macroeconômico
    try:
        sp500 = yf.download('^GSPC', start=start_date, end=end_date)['Close']
        dollar = yf.download('USDBRL=X', start=start_date, end=end_date)['Close']
        vix = yf.download('^VIX', start=start_date, end=end_date)['Close']
        
        ibov['SP500'] = sp500
        ibov['USD_BRL'] = dollar  
        ibov['VIX'] = vix
        print("✓ Dados macroeconômicos carregados")
    except:
        print("⚠️ Alguns dados macro não disponíveis")
    
    print(f"✓ {len(ibov)} dias carregados (15 anos)")
    return ibov

def criar_features_avancadas(data):
    """Cria features avançadas para maximizar acurácia"""
    print("ETAPA 2: Criando features avançadas...")
    
    # Retornos básicos
    data['Return'] = data['Close'].pct_change()
    data['Return_Abs'] = abs(data['Return'])
    
    # ========== FEATURES TÉCNICAS AVANÇADAS ==========
    
    # Múltiplas médias móveis
    for period in [3, 5, 10, 20, 50]:
        data[f'MA_{period}'] = data['Close'].rolling(period).mean()
        data[f'Price_vs_MA_{period}'] = data['Close'] / data[f'MA_{period}'] - 1
        data[f'MA_Slope_{period}'] = data[f'MA_{period}'].diff(3)
    
    # Cruzamentos de médias
    data['MA_Cross_3_10'] = (data['MA_3'] > data['MA_10']).astype(int)
    data['MA_Cross_5_20'] = (data['MA_5'] > data['MA_20']).astype(int)
    data['MA_Cross_10_50'] = (data['MA_10'] > data['MA_50']).astype(int)
    
    # RSI multi-período
    for period in [7, 14, 21]:
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        data[f'RSI_{period}'] = 100 - (100 / (1 + gain / loss))
        data[f'RSI_{period}_Signal'] = ((data[f'RSI_{period}'] < 30) | (data[f'RSI_{period}'] > 70)).astype(int)
    
    # Bollinger Bands
    for period in [10, 20]:
        ma = data['Close'].rolling(period).mean()
        std = data['Close'].rolling(period).std()
        data[f'BB_Upper_{period}'] = ma + (2 * std)
        data[f'BB_Lower_{period}'] = ma - (2 * std)
        data[f'BB_Position_{period}'] = (data['Close'] - data[f'BB_Lower_{period}']) / (data[f'BB_Upper_{period}'] - data[f'BB_Lower_{period}'])
        data[f'BB_Squeeze_{period}'] = (std / ma < 0.02).astype(int)
    
    # Volatilidade multi-escala
    for period in [5, 10, 20, 60]:
        data[f'Vol_{period}'] = data['Return'].rolling(period).std()
        data[f'Vol_Rank_{period}'] = data[f'Vol_{period}'].rolling(252).rank(pct=True)
        data[f'High_Vol_{period}'] = (data[f'Vol_Rank_{period}'] > 0.8).astype(int)
    
    # Momentum e reversão
    for period in [3, 5, 10, 20]:
        data[f'Momentum_{period}'] = data['Close'].pct_change(period)
        data[f'Return_Sum_{period}'] = data['Return'].rolling(period).sum()
        data[f'Reversal_{period}'] = (data['Return'].shift(1) * data['Return'] < 0).astype(int)
    
    # MACD
    ema_12 = data['Close'].ewm(span=12).mean()
    ema_26 = data['Close'].ewm(span=26).mean()
    data['MACD'] = ema_12 - ema_26
    data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
    data['MACD_Hist'] = data['MACD'] - data['MACD_Signal']
    data['MACD_Cross'] = (data['MACD'] > data['MACD_Signal']).astype(int)
    
    # Stochastic Oscillator
    for period in [14, 21]:
        low_min = data['Low'].rolling(period).min()
        high_max = data['High'].rolling(period).max()
        data[f'Stoch_{period}'] = 100 * (data['Close'] - low_min) / (high_max - low_min)
        data[f'Stoch_{period}_Signal'] = ((data[f'Stoch_{period}'] < 20) | (data[f'Stoch_{period}'] > 80)).astype(int)
    
    # ========== FEATURES DE VOLUME ==========
    if 'Volume' in data.columns:
        # Volume features
        for period in [5, 20]:
            data[f'Vol_MA_{period}'] = data['Volume'].rolling(period).mean()
            data[f'Vol_Ratio_{period}'] = data['Volume'] / data[f'Vol_MA_{period}']
            data[f'High_Volume_{period}'] = (data[f'Vol_Ratio_{period}'] > 1.5).astype(int)
        
        # Volume Price Trend
        data['VPT'] = (data['Volume'] * data['Return']).cumsum()
        data['VPT_MA'] = data['VPT'].rolling(20).mean()
        data['VPT_Signal'] = (data['VPT'] > data['VPT_MA']).astype(int)
    
    # ========== FEATURES MACROECONÔMICAS ==========
    if 'SP500' in data.columns:
        data['SP500_Return'] = data['SP500'].pct_change()
        data['SP500_Vol'] = data['SP500_Return'].rolling(20).std()
        data['Corr_SP500'] = data['Return'].rolling(60).corr(data['SP500_Return'])
        data['SP500_Momentum'] = data['SP500'].pct_change(10)
    
    if 'USD_BRL' in data.columns:
        data['USD_Return'] = data['USD_BRL'].pct_change()
        data['USD_Vol'] = data['USD_Return'].rolling(20).std()
        data['USD_Trend'] = (data['USD_BRL'] > data['USD_BRL'].rolling(20).mean()).astype(int)
        data['USD_Momentum'] = data['USD_BRL'].pct_change(10)
    
    if 'VIX' in data.columns:
        data['VIX_Level'] = data['VIX']
        data['VIX_Change'] = data['VIX'].pct_change()
        data['VIX_High'] = (data['VIX'] > 25).astype(int)
        data['VIX_Spike'] = (data['VIX'].pct_change() > 0.1).astype(int)
    
    # ========== FEATURES DE REGIME ==========
    
    # Regime de trend
    data['Uptrend'] = (data['Close'] > data['MA_50']).astype(int)
    data['Strong_Uptrend'] = ((data['MA_5'] > data['MA_20']) & (data['MA_20'] > data['MA_50'])).astype(int)
    
    # Regime de volatilidade
    data['Vol_Regime'] = pd.cut(data['Vol_20'], bins=3, labels=[0, 1, 2]).astype(float)
    
    # Padrões de preço
    data['Doji'] = (abs(data['Close'] - data['Open']) / (data['High'] - data['Low']) < 0.1).astype(int)
    data['Hammer'] = ((data['Close'] > data['Open']) & 
                     ((data['Close'] - data['Open']) / (data['High'] - data['Low']) > 0.6)).astype(int)
    
    # ========== FEATURES DE LAG (AUTOREFERÊNCIA) ==========
    for lag in [1, 2, 3, 5]:
        data[f'Return_Lag_{lag}'] = data['Return'].shift(lag)
        data[f'Vol_Lag_{lag}'] = data['Vol_20'].shift(lag)
        data[f'RSI_14_Lag_{lag}'] = data['RSI_14'].shift(lag)
    
    # ========== SELEÇÃO FINAL DE FEATURES ==========
    
    # Identificar todas as features criadas (excluir colunas originais)
    original_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'SP500', 'USD_BRL', 'VIX']
    all_features = [col for col in data.columns if col not in original_cols and not col.startswith('MA_') or col.startswith('MA_Cross')]
    
    # Filtrar features válidas (sem muitos NaN)
    valid_features = []
    for feature in all_features:
        if data[feature].notna().sum() > len(data) * 0.7:  # 70% de dados válidos
            valid_features.append(feature)
    
    print(f"✓ {len(valid_features)} features avançadas criadas")
    return data, valid_features

def criar_target_otimizado(data, threshold=0.001):
    """Cria target com threshold para reduzir ruído"""
    print("ETAPA 3: Criando target otimizado...")
    
    # Calcular mudança de preço
    price_change = (data['Close'].shift(-1) - data['Close']) / data['Close']
    
    # Target com threshold para reduzir ruído
    data['Target'] = (price_change > threshold).astype(int)
    
    # Estatísticas
    target_counts = data['Target'].value_counts()
    total = len(data['Target'].dropna())
    
    print(f"✓ Target criado com threshold {threshold*100:.1f}%:")
    print(f"  Altas significativas: {target_counts.get(1, 0)} ({target_counts.get(1, 0)/total:.1%})")
    print(f"  Baixas/laterais: {target_counts.get(0, 0)} ({target_counts.get(0, 0)/total:.1%})")
    
    return data

def selecionar_features_otimas(X_treino, y_treino, max_features=25):
    """Seleciona as melhores features usando múltiplos métodos"""
    print("ETAPA 4: Seleção de features otimizada...")
    
    # Método 1: SelectKBest com f_classif
    selector_f = SelectKBest(f_classif, k=min(max_features, X_treino.shape[1]))
    X_selected_f = selector_f.fit_transform(X_treino, y_treino)
    features_f = X_treino.columns[selector_f.get_support()].tolist()
    
    # Método 2: RFE com Random Forest
    rf_temp = RandomForestClassifier(n_estimators=50, random_state=42)
    selector_rfe = RFE(rf_temp, n_features_to_select=min(max_features, X_treino.shape[1]))
    selector_rfe.fit(X_treino, y_treino)
    features_rfe = X_treino.columns[selector_rfe.support_].tolist()
    
    # Combinar métodos (união das features selecionadas)
    features_combined = list(set(features_f + features_rfe))
    
    print(f"✓ Features selecionadas:")
    print(f"  F-test: {len(features_f)} features")
    print(f"  RFE: {len(features_rfe)} features") 
    print(f"  Combinadas: {len(features_combined)} features")
    
    return features_combined

def testar_modelos_otimizados(X_treino, y_treino):
    """Testa modelos otimizados incluindo ensemble"""
    print("ETAPA 5: Testando modelos otimizados...")
    
    # Modelos individuais otimizados
    modelos = {
        'Logistic Regression': LogisticRegression(
            random_state=42, max_iter=2000, C=0.1, class_weight='balanced'
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=200, max_depth=15, min_samples_split=10, 
            random_state=42, class_weight='balanced'
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=150, max_depth=8, learning_rate=0.05,
            random_state=42
        ),
        'SVM': SVC(
            kernel='rbf', C=1.0, gamma='scale', 
            random_state=42, class_weight='balanced', probability=True
        ),
        'Neural Network': MLPClassifier(
            hidden_layer_sizes=(100, 50), max_iter=1000, 
            random_state=42, alpha=0.01
        )
    }
    
    # Ensemble voting
    voting_clf = VotingClassifier([
        ('lr', modelos['Logistic Regression']),
        ('rf', modelos['Random Forest']),
        ('gb', modelos['Gradient Boosting'])
    ], voting='soft')
    
    modelos['Ensemble Voting'] = voting_clf
    
    resultados = {}
    
    # Testar cada modelo
    for nome, modelo in modelos.items():
        # Pipeline com normalização
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', modelo)
        ])
        
        # Validação cruzada temporal
        cv = KFold(n_splits=5, shuffle=False)
        
        try:
            scores = cross_val_score(pipeline, X_treino, y_treino, cv=cv, scoring='accuracy')
            
            resultados[nome] = {
                'mean': scores.mean(),
                'std': scores.std(),
                'scores': scores
            }
            
            status = "✅" if scores.mean() >= 0.75 else "⚠️" if scores.mean() >= 0.60 else "❌"
            print(f"  {status} {nome}: {scores.mean():.1%} (±{scores.std():.1%})")
            
        except Exception as e:
            print(f"  ❌ {nome}: ERRO - {e}")
            resultados[nome] = {'mean': 0, 'std': 0}
    
    # Encontrar melhor modelo
    melhor_modelo = max(resultados.keys(), key=lambda k: resultados[k]['mean'])
    melhor_score = resultados[melhor_modelo]['mean']
    
    print(f"\n🏆 Melhor modelo: {melhor_modelo} ({melhor_score:.1%})")
    
    return resultados, melhor_modelo

def teste_final_otimizado(X_treino, X_teste, y_treino, y_teste, melhor_modelo_nome):
    """Teste final otimizado nos últimos 30 dias"""
    print("ETAPA 6: Teste final otimizado...")
    
    # Mapeamento de modelos
    modelos_map = {
        'Logistic Regression': LogisticRegression(
            random_state=42, max_iter=2000, C=0.1, class_weight='balanced'
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=200, max_depth=15, min_samples_split=10, 
            random_state=42, class_weight='balanced'
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=150, max_depth=8, learning_rate=0.05,
            random_state=42
        ),
        'SVM': SVC(
            kernel='rbf', C=1.0, gamma='scale', 
            random_state=42, class_weight='balanced', probability=True
        ),
        'Neural Network': MLPClassifier(
            hidden_layer_sizes=(100, 50), max_iter=1000, 
            random_state=42, alpha=0.01
        )
    }
    
    # Ensemble se selecionado
    if melhor_modelo_nome == 'Ensemble Voting':
        modelo = VotingClassifier([
            ('lr', modelos_map['Logistic Regression']),
            ('rf', modelos_map['Random Forest']),
            ('gb', modelos_map['Gradient Boosting'])
        ], voting='soft')
    else:
        modelo = modelos_map[melhor_modelo_nome]
    
    # Pipeline otimizado
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', modelo)
    ])
    
    # Treinar
    pipeline.fit(X_treino, y_treino)
    
    # Predizer
    y_pred = pipeline.predict(X_teste)
    
    # Calcular acurácia
    acuracia = accuracy_score(y_teste, y_pred)
    
    print(f"✓ Modelo: {melhor_modelo_nome}")
    print(f"✓ Acurácia nos últimos 30 dias: {acuracia:.1%}")
    
    # Verificar meta
    if acuracia >= 0.75:
        print("🎉 META ATINGIDA: ≥ 75%!")
        status = "SUCESSO"
    else:
        print(f"❌ META NÃO ATINGIDA: {(0.75 - acuracia)*100:.1f} pontos abaixo")
        status = "FALHA"
    
    # Relatório detalhado
    print(f"\n📊 Relatório:")
    print(classification_report(y_teste, y_pred, target_names=['Baixa', 'Alta']))
    
    return {
        'acuracia': acuracia,
        'status': status,
        'modelo': melhor_modelo_nome,
        'predicoes': y_pred,
        'real': y_teste
    }

def main_otimizado():
    """Pipeline principal otimizado para atingir 75%"""
    print("="*70)
    print("🎯 MACHINE LEARNING IBOVESPA - VERSÃO OTIMIZADA")
    print("META: 75% de acurácia nos últimos 30 dias")
    print("="*70)
    
    try:
        # 1. Dados completos
        data = carregar_dados_completos()
        
        # 2. Features avançadas
        data, features_lista = criar_features_avancadas(data)
        
        # 3. Target otimizado
        data = criar_target_otimizado(data, threshold=0.001)
        
        # 4. Preparar dataset
        dataset = data[features_lista + ['Target']].dropna()
        
        # Divisão: últimos 30 dias para teste
        n_test = 30
        X_treino = dataset[features_lista].iloc[:-n_test]
        X_teste = dataset[features_lista].iloc[-n_test:]
        y_treino = dataset['Target'].iloc[:-n_test]
        y_teste = dataset['Target'].iloc[-n_test:]
        
        print(f"\n✓ Dataset: {len(dataset)} observações")
        print(f"  Treino: {len(X_treino)} | Teste: {len(X_teste)}")
        
        # 5. Seleção de features
        features_otimas = selecionar_features_otimas(X_treino, y_treino)
        X_treino_sel = X_treino[features_otimas]
        X_teste_sel = X_teste[features_otimas]
        
        # 6. Modelos otimizados
        resultados_cv, melhor_modelo = testar_modelos_otimizados(X_treino_sel, y_treino)
        
        # 7. Teste final
        resultado_final = teste_final_otimizado(X_treino_sel, X_teste_sel, y_treino, y_teste, melhor_modelo)
        
        # RESUMO FINAL
        print("\n" + "="*70)
        print("🏆 RESULTADO FINAL")
        print("="*70)
        print(f"🎯 META: 75% de acurácia")
        print(f"📊 RESULTADO: {resultado_final['acuracia']:.1%}")
        print(f"🤖 MODELO: {resultado_final['modelo']}")
        print(f"📈 STATUS: {resultado_final['status']}")
        
        cv_score = resultados_cv[melhor_modelo]['mean']
        print(f"\n📈 CONSISTÊNCIA:")
        print(f"  CV: {cv_score:.1%} | Teste: {resultado_final['acuracia']:.1%}")
        print(f"  Diferença: {(resultado_final['acuracia'] - cv_score)*100:+.1f} pontos")
        
        return resultado_final
        
    except Exception as e:
        print(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    resultado = main_otimizado()
    
    if resultado and resultado['status'] == 'SUCESSO':
        print(f"\n🎉 MISSÃO CUMPRIDA! Acurácia: {resultado['acuracia']:.1%}")
    elif resultado:
        print(f"\n📊 Meta não atingida. Acurácia: {resultado['acuracia']:.1%}")
    else:
        print(f"\n❌ Falha na execução")
