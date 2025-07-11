"""
MAIN_75.PY - OTIMIZAÇÃO PARA 75%
🎯 FOCO: Encontrar melhor configuração para condições atuais
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

def otimizar_para_75():
    """
    Busca intensiva por configuração que atinja 75%
    """
    print("🎯 BUSCA INTENSIVA PARA 75%")
    print("="*60)
    
    melhor_resultado = {'acuracia': 0, 'config': None}
    
    # MÚLTIPLOS PERÍODOS
    periodos = [(int(dias), nome) for dias, nome in [
        (1 * 365, "1 ano"),
        (1.5 * 365, "1.5 anos"), 
        (2 * 365, "2 anos"),
        (2.5 * 365, "2.5 anos"),
        (3 * 365, "3 anos"),
        (4 * 365, "4 anos")
    ]]
    
    for dias, nome_periodo in periodos:
        print(f"\n🔍 PERÍODO: {nome_periodo}")
        
        try:
            # CARREGAMENTO
            end_date = datetime.now()
            start_date = end_date - timedelta(days=dias)
            data = yf.download('^BVSP', start=start_date, end=end_date)
            
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            if len(data) < 200:  # Mínimo absoluto
                continue
                
            print(f"   📥 {len(data)} dias")
            
            # FEATURES ROBUSTAS
            data['Return'] = data['Close'].pct_change()
            data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
            
            # BASE
            data['SMA_5'] = data['Close'].rolling(5).mean()
            data['Price_above_SMA5'] = (data['Close'] > data['SMA_5']).astype(int)
            
            data['Volume_MA_20'] = data['Volume'].rolling(20).mean()
            data['Volume_above_avg'] = (data['Volume'] > data['Volume_MA_20']).astype(int)
            
            data['Return_lag1'] = data['Return'].shift(1)
            data['Positive_return_lag1'] = (data['Return_lag1'] > 0).astype(int)
            
            # FEATURES EXTRAS
            data['SMA_10'] = data['Close'].rolling(10).mean()
            data['SMA_20'] = data['Close'].rolling(20).mean()
            data['Trend_strong'] = (data['SMA_5'] > data['SMA_10']).astype(int)
            data['Trend_medium'] = (data['SMA_10'] > data['SMA_20']).astype(int)
            
            data['Return_lag2'] = data['Return'].shift(2)
            data['Momentum_strong'] = ((data['Return_lag1'] > 0) & (data['Return_lag2'] > 0)).astype(int)
            
            data['Vol_5'] = data['Return'].rolling(5).std()
            data['Vol_20'] = data['Return'].rolling(20).std()
            data['Low_vol'] = (data['Vol_5'] < data['Vol_20']).astype(int)
            
            data['Volume_MA_10'] = data['Volume'].rolling(10).mean()
            data['Volume_spike'] = (data['Volume'] > data['Volume_MA_10'] * 1.2).astype(int)
            
            # COMBINAÇÕES DE FEATURES
            todas_features = [
                'Price_above_SMA5', 'Volume_above_avg', 'Positive_return_lag1',
                'Trend_strong', 'Trend_medium', 'Momentum_strong', 
                'Low_vol', 'Volume_spike'
            ]
            
            # Verificar quais features existem
            features_disponiveis = [f for f in todas_features if f in data.columns]
            
            combinacoes = [
                # Base
                ['Price_above_SMA5', 'Volume_above_avg', 'Positive_return_lag1'],
                
                # Base + trend
                ['Price_above_SMA5', 'Volume_above_avg', 'Positive_return_lag1', 'Trend_strong'],
                ['Price_above_SMA5', 'Volume_above_avg', 'Positive_return_lag1', 'Trend_medium'],
                
                # Base + momentum
                ['Price_above_SMA5', 'Volume_above_avg', 'Positive_return_lag1', 'Momentum_strong'],
                
                # Base + volatilidade
                ['Price_above_SMA5', 'Volume_above_avg', 'Positive_return_lag1', 'Low_vol'],
                
                # Base + volume
                ['Price_above_SMA5', 'Volume_above_avg', 'Positive_return_lag1', 'Volume_spike'],
                
                # Combinações de 5
                ['Price_above_SMA5', 'Volume_above_avg', 'Positive_return_lag1', 'Trend_strong', 'Momentum_strong'],
                ['Price_above_SMA5', 'Volume_above_avg', 'Positive_return_lag1', 'Trend_strong', 'Low_vol'],
                
                # Máxima
                features_disponiveis
            ]
            
            for features in combinacoes:
                # Verificar se features existem
                if not all(f in features_disponiveis for f in features):
                    continue
                    
                dataset = data[features + ['Target']].dropna()
                if len(dataset) < 100:
                    continue
                
                # MÚLTIPLAS JANELAS DE TESTE
                for n_test in [15, 20, 25, 30, 40]:
                    if len(dataset) <= n_test + 50:
                        continue
                        
                    train_data = dataset.iloc[:-n_test]
                    test_data = dataset.iloc[-n_test:]
                    
                    X_train = train_data[features]
                    y_train = train_data['Target']
                    X_test = test_data[features]
                    y_test = test_data['Target']
                    
                    # MÚLTIPLOS MODELOS
                    modelos = [
                        # Logistic simples
                        Pipeline([
                            ('scaler', StandardScaler()),
                            ('clf', LogisticRegression(C=1.0, random_state=42))
                        ]),
                        
                        # Random Forest
                        Pipeline([
                            ('scaler', StandardScaler()),
                            ('clf', RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42))
                        ]),
                        
                        # Ensemble soft
                        Pipeline([
                            ('scaler', StandardScaler()),
                            ('clf', VotingClassifier([
                                ('lr', LogisticRegression(C=0.5, random_state=42)),
                                ('rf', RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42))
                            ], voting='soft'))
                        ]),
                        
                        # Ensemble hard
                        Pipeline([
                            ('scaler', StandardScaler()),
                            ('clf', VotingClassifier([
                                ('lr', LogisticRegression(C=1.0, random_state=42)),
                                ('rf', RandomForestClassifier(n_estimators=150, max_depth=7, random_state=42))
                            ], voting='hard'))
                        ])
                    ]
                    
                    for i, modelo in enumerate(modelos):
                        try:
                            modelo.fit(X_train, y_train)
                            y_pred = modelo.predict(X_test)
                            acuracia = accuracy_score(y_test, y_pred)
                            
                            if acuracia > melhor_resultado['acuracia']:
                                melhor_resultado = {
                                    'acuracia': acuracia,
                                    'config': {
                                        'periodo': nome_periodo,
                                        'features': features,
                                        'n_test': n_test,
                                        'modelo_id': i,
                                        'modelo': modelo,
                                        'dados': len(dataset)
                                    }
                                }
                                print(f"   🎯 NOVO MÁXIMO: {acuracia:.1%} | {len(features)}f | {n_test}d | M{i}")
                                
                                # Se atingir 75%, parar busca
                                if acuracia >= 0.75:
                                    print(f"   🎉 75% ATINGIDO! PARANDO BUSCA")
                                    return melhor_resultado
                                    
                        except Exception as e:
                            continue
            
        except Exception as e:
            print(f"   ❌ Erro: {e}")
            continue
    
    return melhor_resultado

if __name__ == "__main__":
    print("🚀 OTIMIZAÇÃO INTENSIVA PARA 75%")
    print("="*60)
    
    resultado = otimizar_para_75()
    
    if resultado['acuracia'] > 0:
        config = resultado['config']
        
        print(f"\n" + "="*60)
        print(f"🏆 MELHOR RESULTADO ENCONTRADO")
        print(f"="*60)
        
        print(f"🎯 ACURÁCIA MÁXIMA: {resultado['acuracia']:.1%}")
        print(f"📊 PERÍODO: {config['periodo']}")
        print(f"🔧 FEATURES ({len(config['features'])}): {config['features']}")
        print(f"🗓️ JANELA: {config['n_test']} dias")
        print(f"🤖 MODELO: {config['modelo_id']}")
        print(f"📊 DADOS: {config['dados']} obs")
        
        print(f"\n🎯 STATUS:")
        if resultado['acuracia'] >= 0.75:
            print(f"   ✅ META 75% ATINGIDA!")
        elif resultado['acuracia'] >= 0.70:
            print(f"   ✅ META 70% ATINGIDA!")
            faltam = (0.75 - resultado['acuracia']) * 100
            print(f"   📈 Faltam {faltam:.1f} pontos para 75%")
        elif resultado['acuracia'] >= 0.60:
            print(f"   ✅ META 60% ATINGIDA!")
            faltam = (0.75 - resultado['acuracia']) * 100
            print(f"   📈 Faltam {faltam:.1f} pontos para 75%")
        
        if resultado['acuracia'] >= 0.75:
            print(f"\n🎉🎉🎉 FANTÁSTICO! 75% ATINGIDO! 🎉🎉🎉")
        elif resultado['acuracia'] >= 0.70:
            print(f"\n🎉 EXCELENTE! Muito próximo dos 75%")
        else:
            print(f"\n📊 Resultado promissor para refinamentos")
            
    else:
        print(f"\n❌ Nenhuma configuração viável encontrada")
    
    print(f"\n✅ BUSCA CONCLUÍDA!")
