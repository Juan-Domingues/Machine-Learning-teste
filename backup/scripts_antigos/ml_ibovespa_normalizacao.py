"""
Machine Learning Aplicado ao IBOVESPA - Teste de Normalização
Baseado nos conceitos fundamentais do curso:
- Linear Regression
- Diferentes técnicas de normalização/padronização
- Comparação de performance
- Pipeline de Machine Learning
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Imports do Scikit-Learn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Configuração para gráficos mais bonitos
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class MLIbovespaNormalizacao:
    """
    Classe para testar diferentes técnicas de normalização
    no modelo Linear Regression aplicado ao IBOVESPA
    """
    
    def __init__(self):
        self.data = None
        self.X = None
        self.y = None
        self.resultados = {}
        
    def carregar_dados(self, anos=10):
        """
        1. CARREGAMENTO DE DADOS
        Baixa dados reais do IBOVESPA via Yahoo Finance
        Incluindo período do Plano Real para maior robustez
        """
        print("="*50)
        print("1. CARREGANDO DADOS DO IBOVESPA")
        print("="*50)
        
        # Definir período
        end_date = datetime.now()
        start_date = end_date - timedelta(days=anos * 365)
        
        print(f"📊 Baixando dados de {start_date.strftime('%Y-%m-%d')} até {end_date.strftime('%Y-%m-%d')}")
        print(f"🗓️ Período: {anos} anos de dados (incluindo era Plano Real)")
        print("💡 Dados históricos extensos para melhor aprendizado de padrões")
        
        # Baixar dados
        self.data = yf.download('^BVSP', start=start_date, end=end_date)
        
        # Limpar colunas se necessário
        if isinstance(self.data.columns, pd.MultiIndex):
            self.data.columns = self.data.columns.get_level_values(0)
        
        print(f"✅ {len(self.data)} dias de dados carregados")
        print(f"📈 Preço atual: R$ {self.data['Close'].iloc[-1]:,.2f}")
        
        return self.data
    
    def engenharia_features(self):
        """
        2. ENGENHARIA DE FEATURES
        Criação de indicadores técnicos simples
        """
        print("\n" + "="*50)
        print("2. CRIANDO FEATURES (INDICADORES TÉCNICOS)")
        print("="*50)
        
        # Calcular retornos
        self.data['Retorno'] = self.data['Close'].pct_change()
        
        # Médias móveis
        self.data['MM5'] = self.data['Close'].rolling(window=5).mean()
        self.data['MM20'] = self.data['Close'].rolling(window=20).mean()
        self.data['MM50'] = self.data['Close'].rolling(window=50).mean()
        
        # RSI simplificado
        delta = self.data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.data['RSI'] = 100 - (100 / (1 + rs))
        
        # Volatilidade
        self.data['Volatilidade'] = self.data['Retorno'].rolling(window=20).std()
        
        # Volume normalizado
        self.data['Volume_Norm'] = self.data['Volume'] / self.data['Volume'].rolling(window=20).mean()
        
        # Máximos e mínimos
        self.data['Max_20'] = self.data['High'].rolling(window=20).max()
        self.data['Min_20'] = self.data['Low'].rolling(window=20).min()
        self.data['Posicao_Canal'] = (self.data['Close'] - self.data['Min_20']) / (self.data['Max_20'] - self.data['Min_20'])
        
        print("✅ Features criadas:")
        features = ['Retorno', 'MM5', 'MM20', 'MM50', 'RSI', 'Volatilidade', 'Volume_Norm', 'Posicao_Canal']
        for feat in features:
            print(f"   📊 {feat}")
        
        return self.data
    
    def criar_target(self):
        """
        3. CRIAR VARIÁVEL TARGET
        Retorno futuro de 1 dia (REGRESSÃO)
        """
        print("\n" + "="*50)
        print("3. CRIANDO TARGET (RETORNO FUTURO)")
        print("="*50)
        
        # Target: retorno do próximo dia
        self.data['Target'] = self.data['Retorno'].shift(-1)
        
        print("✅ Target criado: Retorno do próximo dia")
        print(f"📊 Distribuição do target:")
        print(self.data['Target'].describe())
        
        return self.data
    
    def preparar_dataset(self, features=None):
        """
        4. PREPARAR DATASET FINAL
        """
        print("\n" + "="*50)
        print("4. PREPARANDO DATASET FINAL")
        print("="*50)
        
        # Usar features fornecidas ou padrão
        if features is None:
            features = ['MM5', 'MM20', 'MM50', 'RSI', 'Volatilidade', 'Volume_Norm', 'Posicao_Canal']
        
        # Criar dataset
        dataset = self.data[features + ['Target']].copy()
        
        # Remover valores nulos
        dataset = dataset.dropna()
        
        # Separar X e y
        self.X = dataset[features]
        self.y = dataset['Target']
        
        print(f"✅ Dataset preparado:")
        print(f"📊 {len(self.X)} observações")
        print(f"📊 {len(features)} features")
        print(f"📊 Target: {self.y.name}")
        
        return self.X, self.y
    
    def comparar_normalizacao(self, features=None):
        """
        Compara diferentes técnicas de normalização/padronização
        Usando divisão 70% treino / 30% teste conforme orientação do professor
        """
        print("\n" + "="*70)
        print("🔬 TESTE DE NORMALIZAÇÃO - COMPARAÇÃO DE SCALERS")
        print("🎯 Divisão: 70% Treino / 30% Teste (orientação professor)")
        print("="*70)
        
        # Preparar dados
        X, y = self.preparar_dataset(features)
        X_treino, X_teste, y_treino, y_teste = train_test_split(
            X, y, test_size=0.3, random_state=42, shuffle=False
        )
        
        print(f"📊 Divisão dos dados:")
        print(f"   🎓 Treino: {len(X_treino)} observações (70%)")
        print(f"   🧪 Teste: {len(X_teste)} observações (30%)")
        
        # Diferentes tipos de scalers
        scalers = {
            'Sem Normalização': None,
            'StandardScaler': StandardScaler(),
            'MinMaxScaler': MinMaxScaler(),
            'RobustScaler': RobustScaler(),
            'Normalizer': Normalizer()
        }
        
        resultados_normalizacao = {}
        
        for nome_scaler, scaler in scalers.items():
            print(f"\n🧪 Testando: {nome_scaler}")
            print("-" * 40)
            
            try:
                # Criar pipeline
                if scaler is None:
                    modelo = LinearRegression()
                    X_treino_scaled = X_treino
                    X_teste_scaled = X_teste
                else:
                    pipeline = Pipeline([
                        ('scaler', scaler),
                        ('regressor', LinearRegression())
                    ])
                    modelo = pipeline
                    X_treino_scaled = X_treino
                    X_teste_scaled = X_teste
                
                # Treinar modelo
                modelo.fit(X_treino_scaled, y_treino)
                
                # Fazer previsões
                y_pred = modelo.predict(X_teste_scaled)
                
                # Calcular métricas de regressão
                r2 = r2_score(y_teste, y_pred)
                rmse = np.sqrt(mean_squared_error(y_teste, y_pred))
                mae = mean_absolute_error(y_teste, y_pred)
                
                # Calcular acurácia de direção
                y_teste_direcao = (y_teste > 0).astype(int)
                y_pred_direcao = (y_pred > 0).astype(int)
                acuracia = np.mean(y_teste_direcao == y_pred_direcao)
                
                # Armazenar resultados
                resultados_normalizacao[nome_scaler] = {
                    'r2': r2,
                    'rmse': rmse,
                    'mae': mae,
                    'acuracia': acuracia
                }
                
                # Exibir resultados
                print(f"📊 R²: {r2:.4f}")
                print(f"📊 RMSE: {rmse:.4f}")
                print(f"📊 MAE: {mae:.4f}")
                print(f"🎯 Acurácia: {acuracia:.1%}")
                
                qualidade = "✅ Excelente" if acuracia >= 0.6 else "✅ Satisfatória" if acuracia >= 0.55 else "⚠️ Pode melhorar"
                print(f"📈 Qualidade: {qualidade}")
                
            except Exception as e:
                print(f"❌ Erro com {nome_scaler}: {e}")
                continue
        
        # Encontrar melhor scaler
        print("\n" + "="*70)
        print("🏆 RANKING DOS SCALERS - ORDENADO POR ACURÁCIA")
        print("="*70)
        
        # Ordenar por acurácia
        ranking = sorted(resultados_normalizacao.items(), 
                        key=lambda x: x[1]['acuracia'], reverse=True)
        
        for i, (nome, metricas) in enumerate(ranking, 1):
            if nome in resultados_normalizacao:
                print(f"{i}º 🥇 {nome}:")
                print(f"   🎯 Acurácia: {metricas['acuracia']:.1%}")
                print(f"   📊 R²: {metricas['r2']:.4f}")
                print(f"   📊 RMSE: {metricas['rmse']:.4f}")
                print(f"   📊 MAE: {metricas['mae']:.4f}")
                print()
        
        # Resumo final
        melhor_scaler = ranking[0][0]
        melhores_metricas = ranking[0][1]
        
        print("🎯 MELHOR CONFIGURAÇÃO:")
        print(f"🏆 Scaler: {melhor_scaler}")
        print(f"🎯 Acurácia: {melhores_metricas['acuracia']:.1%}")
        print(f"📊 R²: {melhores_metricas['r2']:.4f}")
        
        # Comparação com sem normalização
        if 'Sem Normalização' in resultados_normalizacao:
            sem_norm = resultados_normalizacao['Sem Normalização']
            melhor_norm = melhores_metricas
            
            melhora_acuracia = melhor_norm['acuracia'] - sem_norm['acuracia']
            melhora_r2 = melhor_norm['r2'] - sem_norm['r2']
            
            print(f"\n📈 MELHORA COMPARADA À SEM NORMALIZAÇÃO:")
            print(f"🎯 Acurácia: {melhora_acuracia:+.1%}")
            print(f"📊 R²: {melhora_r2:+.4f}")
            
            if melhora_acuracia > 0.01:  # Melhora significativa
                print("✅ Normalização trouxe melhora significativa!")
            elif melhora_acuracia > 0:
                print("✅ Normalização trouxe leve melhora")
            else:
                print("⚠️ Normalização não melhorou o desempenho")
        
        return resultados_normalizacao, melhor_scaler
    
    def executar_teste_completo(self):
        """
        Executa todo o pipeline de teste de normalização e análise de features
        """
        try:
            # 1-3: Preparação dos dados
            self.carregar_dados()
            self.engenharia_features()
            self.criar_target()
            
            # 4: Análise exploratória das features originais
            print("\n" + "="*60)
            print("📊 TESTE 1: FEATURES ORIGINAIS")
            print("="*60)
            correlacoes_orig, corr_matrix_orig, dataset_orig = self.analisar_features()
            resultados_orig, melhor_scaler_orig = self.comparar_normalizacao()
            
            # 5-7: Criação e análise de features avançadas
            print("\n" + "="*60)
            print("🚀 TESTE 2: FEATURES AVANÇADAS")
            print("="*60)
            features_avancadas = self.criar_features_avancadas()
            melhores_features, correlacoes_melhoradas = self.selecionar_melhores_features()
            
            # 8: Teste com melhores features
            print("\n" + "="*60)
            print("🎯 TESTE 3: MELHORES FEATURES SELECIONADAS")
            print("="*60)
            resultados_melhoradas, melhor_scaler_melhorado = self.comparar_normalizacao(melhores_features)
            
            # 9: Validação cruzada com features originais
            print("\n" + "="*60)
            print("🔄 TESTE 4: VALIDAÇÃO CRUZADA - FEATURES ORIGINAIS")
            print("="*60)
            cv_orig, melhor_cv_orig = self.validacao_cruzada_acuracia()
            
            # 10: Validação cruzada com melhores features
            print("\n" + "="*60)
            print("🔄 TESTE 5: VALIDAÇÃO CRUZADA - MELHORES FEATURES")
            print("="*60)
            cv_melhoradas, melhor_cv_melhorado = self.validacao_cruzada_acuracia(melhores_features)
            
            # Comparação final
            print("\n" + "="*60)
            print("� COMPARAÇÃO FINAL DOS RESULTADOS")
            print("="*60)
            
            # Melhor resultado original
            melhor_orig = max(resultados_orig.values(), key=lambda x: x['acuracia'])
            melhor_melhorado = max(resultados_melhoradas.values(), key=lambda x: x['acuracia'])
            
            print("🏆 RESULTADOS COMPARATIVOS:")
            print(f"\n1️⃣ Features Originais (melhor resultado):")
            print(f"   � Acurácia: {melhor_orig['acuracia']:.1%}")
            print(f"   📊 R²: {melhor_orig['r2']:.4f}")
            print(f"   🔧 Scaler: {melhor_scaler_orig}")
            
            print(f"\n2️⃣ Features Melhoradas (melhor resultado):")
            print(f"   🎯 Acurácia: {melhor_melhorado['acuracia']:.1%}")
            print(f"   📊 R²: {melhor_melhorado['r2']:.4f}")
            print(f"   🔧 Scaler: {melhor_scaler_melhorado}")
            
            # Calcular melhoria
            melhoria_acuracia = melhor_melhorado['acuracia'] - melhor_orig['acuracia']
            melhoria_r2 = melhor_melhorado['r2'] - melhor_orig['r2']
            
            print(f"\n📈 MELHORIA OBTIDA:")
            print(f"   🎯 Acurácia: {melhoria_acuracia:+.1%}")
            print(f"   📊 R²: {melhoria_r2:+.4f}")
            
            if melhoria_acuracia > 0.02:
                print("   ✅ MELHORIA SIGNIFICATIVA!")
            elif melhoria_acuracia > 0:
                print("   ✅ Leve melhoria")
            else:
                print("   ⚠️ Sem melhoria significativa")
            
            # Resumo final
            print("\n" + "="*50)
            print("🎯 RESUMO FINAL - COMPLETO")
            print("="*50)
            print(f"📊 Dataset: {len(self.X)} observações do IBOVESPA")
            
            # Resultados do teste holdout
            print(f"\n🧪 TESTE HOLDOUT (70/30):")
            print(f"🏆 Melhor configuração: {melhor_scaler_melhorado} + Features Otimizadas")
            print(f"🎯 Acurácia final: {melhor_melhorado['acuracia']:.1%}")
            print(f"📊 R² final: {melhor_melhorado['r2']:.4f}")
            
            # Resultados da validação cruzada
            melhor_cv_result = cv_melhoradas[melhor_cv_melhorado]
            print(f"\n🔄 VALIDAÇÃO CRUZADA (5-Fold):")
            print(f"🏆 Melhor configuração CV: {melhor_cv_melhorado} + Features Otimizadas")
            print(f"🎯 Acurácia CV: {melhor_cv_result['acuracia_mean']:.1%} (±{melhor_cv_result['acuracia_std']:.1%})")
            print(f"📊 R² CV: {melhor_cv_result['r2_mean']:.4f} (±{melhor_cv_result['r2_std']:.4f})")
            
            # Comparação entre métodos
            diff_acuracia = melhor_melhorado['acuracia'] - melhor_cv_result['acuracia_mean']
            print(f"\n📈 COMPARAÇÃO MÉTODOS:")
            print(f"🔀 Diferença Holdout vs CV: {diff_acuracia:+.1%}")
            
            if abs(diff_acuracia) < 0.02:
                print("✅ Resultados consistentes entre métodos!")
            elif diff_acuracia > 0.02:
                print("⚠️ Possível overfitting (holdout muito superior)")
            else:
                print("⚠️ Holdout pode ter sido pessimista")
                
            print(f"📈 Qualidade: {'✅ Excelente' if melhor_cv_result['acuracia_mean'] >= 0.6 else '✅ Satisfatória' if melhor_cv_result['acuracia_mean'] >= 0.55 else '⚠️ Pode melhorar'}")
            print("")
            print("✅ Análise completa com validação cruzada concluída!")
            
            return {
                'originais': resultados_orig,
                'melhoradas': resultados_melhoradas,
                'melhor_scaler_orig': melhor_scaler_orig,
                'melhor_scaler_melhorado': melhor_scaler_melhorado,
                'melhores_features': melhores_features,
                'melhoria_acuracia': melhoria_acuracia,
                'melhoria_r2': melhoria_r2,
                'cv_originais': cv_orig,
                'cv_melhoradas': cv_melhoradas,
                'melhor_cv_orig': melhor_cv_orig,
                'melhor_cv_melhorado': melhor_cv_melhorado
            }
            
        except Exception as e:
            print(f"❌ Erro durante a execução: {e}")
            return None

    def analisar_features(self):
        """
        5. ANÁLISE EXPLORATÓRIA DAS FEATURES
        Analisa correlações, importância e distribuições
        """
        print("\n" + "="*60)
        print("🔍 ANÁLISE EXPLORATÓRIA DAS FEATURES")
        print("="*60)
        
        # Preparar dados para análise
        features = ['MM5', 'MM20', 'MM50', 'RSI', 'Volatilidade', 'Volume_Norm', 'Posicao_Canal']
        dataset = self.data[features + ['Target']].dropna()
        
        print(f"📊 Estatísticas Descritivas:")
        print(dataset[features].describe())
        
        # Correlação com target
        print(f"\n📈 Correlação com Target (Retorno Futuro):")
        correlacoes = dataset[features].corrwith(dataset['Target']).sort_values(key=abs, ascending=False)
        for feat, corr in correlacoes.items():
            simbolo = "🔴" if abs(corr) > 0.1 else "🟡" if abs(corr) > 0.05 else "🟢"
            print(f"   {simbolo} {feat}: {corr:.4f}")
        
        # Matriz de correlação entre features
        print(f"\n🔄 Correlações entre Features:")
        corr_matrix = dataset[features].corr()
        
        # Identificar correlações altas (multicolinearidade)
        high_corr_pairs = []
        for i in range(len(features)):
            for j in range(i+1, len(features)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.8:
                    high_corr_pairs.append((features[i], features[j], corr_val))
        
        if high_corr_pairs:
            print("   ⚠️ Correlações altas detectadas (multicolinearidade):")
            for feat1, feat2, corr in high_corr_pairs:
                print(f"     {feat1} <-> {feat2}: {corr:.3f}")
        else:
            print("   ✅ Sem multicolinearidade significativa")
        
        # Análise de distribuição do target
        print(f"\n📊 Análise do Target:")
        target_stats = dataset['Target'].describe()
        print(f"   Média: {target_stats['mean']:.6f}")
        print(f"   Desvio: {target_stats['std']:.6f}")
        print(f"   Assimetria: {dataset['Target'].skew():.4f}")
        print(f"   Curtose: {dataset['Target'].kurtosis():.4f}")
        
        # Contar direções
        direcoes = (dataset['Target'] > 0).value_counts()
        print(f"   Altas: {direcoes[True]} ({direcoes[True]/len(dataset):.1%})")
        print(f"   Baixas: {direcoes[False]} ({direcoes[False]/len(dataset):.1%})")
        
        return correlacoes, corr_matrix, dataset
    
    def criar_features_avancadas(self):
        """
        6. ENGENHARIA DE FEATURES AVANÇADA
        Cria features mais sofisticadas baseadas na análise
        """
        print("\n" + "="*60)
        print("🧠 CRIANDO FEATURES AVANÇADAS")
        print("="*60)
        
        # Features de momentum
        self.data['Momentum_3'] = self.data['Close'] / self.data['Close'].shift(3) - 1
        self.data['Momentum_7'] = self.data['Close'] / self.data['Close'].shift(7) - 1
        self.data['Momentum_14'] = self.data['Close'] / self.data['Close'].shift(14) - 1
        
        # Features de volatilidade
        self.data['Volatilidade_5'] = self.data['Retorno'].rolling(5).std()
        self.data['Volatilidade_10'] = self.data['Retorno'].rolling(10).std()
        self.data['Vol_Ratio'] = self.data['Volatilidade_5'] / self.data['Volatilidade_10']
        
        # Features de médias móveis
        self.data['MM_Ratio_5_20'] = self.data['MM5'] / self.data['MM20']
        self.data['MM_Ratio_20_50'] = self.data['MM20'] / self.data['MM50']
        self.data['Distancia_MM20'] = (self.data['Close'] - self.data['MM20']) / self.data['MM20']
        
        # Features de volume
        self.data['Volume_MA5'] = self.data['Volume'].rolling(5).mean()
        self.data['Volume_Ratio_5'] = self.data['Volume'] / self.data['Volume_MA5']
        self.data['Volume_Trend'] = self.data['Volume_MA5'] / self.data['Volume_MA5'].shift(5) - 1
        
        # Features de canal/suporte/resistência
        self.data['Canal_Width'] = (self.data['Max_20'] - self.data['Min_20']) / self.data['Close']
        self.data['Distancia_Max'] = (self.data['Max_20'] - self.data['Close']) / self.data['Close']
        self.data['Distancia_Min'] = (self.data['Close'] - self.data['Min_20']) / self.data['Close']
        
        # Features de retorno acumulado
        self.data['Retorno_Acum_3'] = self.data['Retorno'].rolling(3).sum()
        self.data['Retorno_Acum_7'] = self.data['Retorno'].rolling(7).sum()
        
        # Features de RSI
        self.data['RSI_MA'] = self.data['RSI'].rolling(5).mean()
        self.data['RSI_Divergencia'] = self.data['RSI'] - self.data['RSI_MA']
        
        # Features de diferenciação temporal
        self.data['RSI_Diff'] = self.data['RSI'].diff()
        self.data['Vol_Diff'] = self.data['Volatilidade'].diff()
        
        print("✅ Features avançadas criadas:")
        features_avancadas = [
            'Momentum_3', 'Momentum_7', 'Momentum_14',
            'Volatilidade_5', 'Vol_Ratio', 
            'MM_Ratio_5_20', 'MM_Ratio_20_50', 'Distancia_MM20',
            'Volume_Ratio_5', 'Volume_Trend',
            'Canal_Width', 'Distancia_Max', 'Distancia_Min',
            'Retorno_Acum_3', 'Retorno_Acum_7',
            'RSI_MA', 'RSI_Divergencia', 'RSI_Diff', 'Vol_Diff'
        ]
        
        for i, feat in enumerate(features_avancadas, 1):
            print(f"   {i:2d}. {feat}")
        
        return features_avancadas
    
    def selecionar_melhores_features(self, max_features=10):
        """
        7. SELEÇÃO DE FEATURES BASEADA EM CORRELAÇÃO
        Seleciona as features com maior correlação absoluta com o target
        """
        print("\n" + "="*60)
        print("🎯 SELEÇÃO DAS MELHORES FEATURES")
        print("="*60)
        
        # Todas as features disponíveis
        todas_features = [
            'MM5', 'MM20', 'MM50', 'RSI', 'Volatilidade', 'Volume_Norm', 'Posicao_Canal',
            'Momentum_3', 'Momentum_7', 'Momentum_14',
            'Volatilidade_5', 'Vol_Ratio', 
            'MM_Ratio_5_20', 'MM_Ratio_20_50', 'Distancia_MM20',
            'Volume_Ratio_5', 'Volume_Trend',
            'Canal_Width', 'Distancia_Max', 'Distancia_Min',
            'Retorno_Acum_3', 'Retorno_Acum_7',
            'RSI_MA', 'RSI_Divergencia', 'RSI_Diff', 'Vol_Diff'
        ]
        
        # Preparar dataset
        dataset = self.data[todas_features + ['Target']].dropna()
        
        # Calcular correlações absolutas
        correlacoes = dataset[todas_features].corrwith(dataset['Target']).abs().sort_values(ascending=False)
        
        # Selecionar top features
        melhores_features = correlacoes.head(max_features).index.tolist()
        
        print(f"🏆 Top {max_features} Features (por correlação absoluta):")
        for i, (feat, corr) in enumerate(correlacoes.head(max_features).items(), 1):
            simbolo = "🔴" if corr > 0.1 else "🟡" if corr > 0.05 else "🟢"
            print(f"   {i:2d}. {simbolo} {feat}: {corr:.4f}")
        
        # Verificar melhoria
        print(f"\n📈 Comparação com features originais:")
        features_originais = ['MM5', 'MM20', 'MM50', 'RSI', 'Volatilidade', 'Volume_Norm', 'Posicao_Canal']
        corr_originais = dataset[features_originais].corrwith(dataset['Target']).abs().mean()
        corr_melhores = correlacoes.head(max_features).mean()
        
        print(f"   Features originais: {corr_originais:.4f} (correlação média)")
        print(f"   Melhores features: {corr_melhores:.4f} (correlação média)")
        print(f"   Melhoria: {((corr_melhores/corr_originais - 1) * 100):+.1f}%")
        
        return melhores_features, correlacoes
    
    def validacao_cruzada_acuracia(self, features=None, cv_folds=5):
        """
        Realiza validação cruzada para medir acurácia de direção
        """
        print("\n" + "="*70)
        print("🔄 VALIDAÇÃO CRUZADA - ACURÁCIA DE DIREÇÃO")
        print(f"🎯 K-Fold: {cv_folds} folds")
        print("="*70)
        
        # Preparar dados
        X, y = self.preparar_dataset(features)
        
        # Diferentes tipos de scalers
        scalers = {
            'Sem Normalização': None,
            'StandardScaler': StandardScaler(),
            'MinMaxScaler': MinMaxScaler(),
            'RobustScaler': RobustScaler(),
            'Normalizer': Normalizer()
        }
        
        resultados_cv = {}
        
        # Função personalizada para calcular acurácia de direção
        def acuracia_direcao(y_true, y_pred):
            y_true_dir = (y_true > 0).astype(int)
            y_pred_dir = (y_pred > 0).astype(int)
            return np.mean(y_true_dir == y_pred_dir)
        
        from sklearn.model_selection import cross_val_score
        from sklearn.metrics import make_scorer
        
        # Criar scorer personalizado
        scorer_acuracia = make_scorer(acuracia_direcao)
        
        for nome_scaler, scaler in scalers.items():
            print(f"\n🧪 Validação cruzada: {nome_scaler}")
            print("-" * 50)
            
            try:
                # Criar pipeline
                if scaler is None:
                    modelo = LinearRegression()
                else:
                    modelo = Pipeline([
                        ('scaler', scaler),
                        ('regressor', LinearRegression())
                    ])
                
                # Validação cruzada para acurácia de direção
                scores_acuracia = cross_val_score(
                    modelo, X, y, 
                    cv=cv_folds, 
                    scoring=scorer_acuracia,
                    n_jobs=-1
                )
                
                # Validação cruzada para R²
                scores_r2 = cross_val_score(
                    modelo, X, y, 
                    cv=cv_folds, 
                    scoring='r2',
                    n_jobs=-1
                )
                
                # Armazenar resultados
                resultados_cv[nome_scaler] = {
                    'acuracia_scores': scores_acuracia,
                    'acuracia_mean': scores_acuracia.mean(),
                    'acuracia_std': scores_acuracia.std(),
                    'r2_scores': scores_r2,
                    'r2_mean': scores_r2.mean(),
                    'r2_std': scores_r2.std()
                }
                
                # Exibir resultados
                print(f"📊 Acurácia CV: {scores_acuracia.mean():.1%} (±{scores_acuracia.std():.1%})")
                print(f"📊 R² CV: {scores_r2.mean():.4f} (±{scores_r2.std():.4f})")
                print(f"📊 Folds individuais acurácia: {[f'{s:.1%}' for s in scores_acuracia]}")
                
                qualidade = "✅ Excelente" if scores_acuracia.mean() >= 0.6 else "✅ Satisfatória" if scores_acuracia.mean() >= 0.55 else "⚠️ Pode melhorar"
                print(f"📈 Qualidade: {qualidade}")
                
            except Exception as e:
                print(f"❌ Erro com {nome_scaler}: {e}")
                continue
        
        # Ranking por acurácia CV
        print("\n" + "="*70)
        print("🏆 RANKING VALIDAÇÃO CRUZADA - ACURÁCIA")
        print("="*70)
        
        ranking_cv = sorted(resultados_cv.items(), 
                           key=lambda x: x[1]['acuracia_mean'], reverse=True)
        
        for i, (nome, metricas) in enumerate(ranking_cv, 1):
            print(f"{i}º 🥇 {nome}:")
            print(f"   🎯 Acurácia CV: {metricas['acuracia_mean']:.1%} (±{metricas['acuracia_std']:.1%})")
            print(f"   📊 R² CV: {metricas['r2_mean']:.4f} (±{metricas['r2_std']:.4f})")
            print()
        
        # Resumo da melhor configuração
        melhor_nome_cv = ranking_cv[0][0]
        melhor_cv = ranking_cv[0][1]
        
        print("🎯 MELHOR CONFIGURAÇÃO (VALIDAÇÃO CRUZADA):")
        print(f"🏆 Scaler: {melhor_nome_cv}")
        print(f"🎯 Acurácia CV: {melhor_cv['acuracia_mean']:.1%} (±{melhor_cv['acuracia_std']:.1%})")
        print(f"📊 R² CV: {melhor_cv['r2_mean']:.4f} (±{melhor_cv['r2_std']:.4f})")
        
        # Intervalo de confiança aproximado (95%)
        ic_inf = melhor_cv['acuracia_mean'] - 1.96 * melhor_cv['acuracia_std']
        ic_sup = melhor_cv['acuracia_mean'] + 1.96 * melhor_cv['acuracia_std']
        print(f"📊 Intervalo 95%: [{ic_inf:.1%}, {ic_sup:.1%}]")
        
        return resultados_cv, melhor_nome_cv


def main():
    """
    Função principal para executar o projeto
    Seguindo orientações do professor: 10 anos de dados + divisão 70/30
    """
    print("📚 MACHINE LEARNING - LINEAR REGRESSION NO IBOVESPA")
    print("🔬 ANÁLISE DE FEATURES + TESTE DE NORMALIZAÇÃO")
    print("🎯 Configuração Professor: 10 anos + Divisão 70/30")
    print("💡 Dados do Plano Real para maior robustez histórica")
    print("=" * 70)
    
    # Criar modelo
    ml = MLIbovespaNormalizacao()
    
    # Executar análise completa
    resultados = ml.executar_teste_completo()
    
    if resultados:
        print(f"\n🎉 Análise completa concluída!")
        
        # Extrair melhores resultados
        melhor_orig = max(resultados['originais'].values(), key=lambda x: x['acuracia'])
        melhor_melhorado = max(resultados['melhoradas'].values(), key=lambda x: x['acuracia'])
        
        print(f"\n📊 RESUMO EXECUTIVO:")
        print(f"🔧 Melhor scaler original: {resultados['melhor_scaler_orig']}")
        print(f"🔧 Melhor scaler otimizado: {resultados['melhor_scaler_melhorado']}")
        print(f"📈 Melhoria na acurácia: {resultados['melhoria_acuracia']:+.1%}")
        print(f"� Melhoria no R²: {resultados['melhoria_r2']:+.4f}")
        print(f"🎯 Acurácia final: {melhor_melhorado['acuracia']:.1%}")
    else:
        print("\n❌ Análise não foi concluída devido a erros.")
    
    return ml, resultados


if __name__ == "__main__":
    modelo, resultados = main()
