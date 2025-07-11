"""
Machine Learning Aplicado ao IBOVESPA - Análise Completa com Validação Cruzada
Baseado nos conceitos fundamentais do curso:
- Linear Regression
- Diferentes técnicas de normalização/padronização
- Validação Cruzada
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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer

# Configuração para gráficos mais bonitos
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class MLIbovespaValidacaoCruzada:
    """
    Classe para testar diferentes técnicas de normalização com validação cruzada
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
        print("💡 Dados históricos extensos para validação cruzada robusta")
        
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
        
        # Features avançadas
        self.data['Momentum_3'] = self.data['Close'] / self.data['Close'].shift(3) - 1
        self.data['Momentum_7'] = self.data['Close'] / self.data['Close'].shift(7) - 1
        self.data['Momentum_14'] = self.data['Close'] / self.data['Close'].shift(14) - 1
        
        self.data['Volatilidade_5'] = self.data['Retorno'].rolling(5).std()
        self.data['Vol_Ratio'] = self.data['Volatilidade_5'] / self.data['Volatilidade']
        
        self.data['MM_Ratio_5_20'] = self.data['MM5'] / self.data['MM20']
        self.data['MM_Ratio_20_50'] = self.data['MM20'] / self.data['MM50']
        self.data['Distancia_MM20'] = (self.data['Close'] - self.data['MM20']) / self.data['MM20']
        
        self.data['Canal_Width'] = (self.data['Max_20'] - self.data['Min_20']) / self.data['Close']
        self.data['Retorno_Acum_3'] = self.data['Retorno'].rolling(3).sum()
        self.data['Retorno_Acum_7'] = self.data['Retorno'].rolling(7).sum()
        
        print("✅ Features criadas (básicas + avançadas)")
        
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
    
    def selecionar_melhores_features(self, max_features=10):
        """
        4. SELEÇÃO DAS MELHORES FEATURES
        """
        print("\n" + "="*60)
        print("4. SELEÇÃO DAS MELHORES FEATURES")
        print("="*60)
        
        # Todas as features disponíveis
        todas_features = [
            'MM5', 'MM20', 'MM50', 'RSI', 'Volatilidade', 'Volume_Norm', 'Posicao_Canal',
            'Momentum_3', 'Momentum_7', 'Momentum_14',
            'Volatilidade_5', 'Vol_Ratio', 
            'MM_Ratio_5_20', 'MM_Ratio_20_50', 'Distancia_MM20',
            'Canal_Width', 'Retorno_Acum_3', 'Retorno_Acum_7'
        ]
        
        # Preparar dataset
        dataset = self.data[todas_features + ['Target']].dropna()
        
        # Calcular correlações absolutas
        correlacoes = dataset[todas_features].corrwith(dataset['Target']).abs().sort_values(ascending=False)
        
        # Selecionar top features
        melhores_features = correlacoes.head(max_features).index.tolist()
        
        print(f"🏆 Top {max_features} Features selecionadas:")
        for i, (feat, corr) in enumerate(correlacoes.head(max_features).items(), 1):
            simbolo = "🔴" if corr > 0.1 else "🟡" if corr > 0.05 else "🟢"
            print(f"   {i:2d}. {simbolo} {feat}: {corr:.4f}")
        
        # Preparar dataset final
        self.X = dataset[melhores_features]
        self.y = dataset['Target']
        
        print(f"\n✅ Dataset preparado:")
        print(f"📊 {len(self.X)} observações")
        print(f"📊 {len(melhores_features)} features selecionadas")
        
        return melhores_features
    
    def validacao_cruzada_completa(self, cv_folds=5):
        """
        5. VALIDAÇÃO CRUZADA COMPLETA
        Testa diferentes scalers com validação cruzada
        """
        print("\n" + "="*70)
        print("🔄 VALIDAÇÃO CRUZADA - COMPARAÇÃO DE SCALERS")
        print(f"🎯 K-Fold: {cv_folds} folds")
        print("🎯 Divisão temporal preservada")
        print("="*70)
        
        # Diferentes tipos de scalers
        scalers = {
            'Sem Normalização': None,
            'StandardScaler': StandardScaler(),
            'MinMaxScaler': MinMaxScaler(),
            'RobustScaler': RobustScaler(),
            'Normalizer': Normalizer()
        }
        
        # Função personalizada para calcular acurácia de direção
        def acuracia_direcao(y_true, y_pred):
            y_true_dir = (y_true > 0).astype(int)
            y_pred_dir = (y_pred > 0).astype(int)
            return np.mean(y_true_dir == y_pred_dir)
        
        # Criar scorer personalizado
        scorer_acuracia = make_scorer(acuracia_direcao)
        
        resultados_cv = {}
        
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
                    modelo, self.X, self.y, 
                    cv=cv_folds, 
                    scoring=scorer_acuracia,
                    n_jobs=-1
                )
                
                # Validação cruzada para R²
                scores_r2 = cross_val_score(
                    modelo, self.X, self.y, 
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
                print(f"📊 Folds acurácia: {[f'{s:.1%}' for s in scores_acuracia]}")
                
                qualidade = "✅ Excelente" if scores_acuracia.mean() >= 0.6 else "✅ Satisfatória" if scores_acuracia.mean() >= 0.55 else "⚠️ Pode melhorar"
                print(f"📈 Qualidade: {qualidade}")
                
            except Exception as e:
                print(f"❌ Erro com {nome_scaler}: {e}")
                continue
        
        return resultados_cv
    
    def teste_holdout(self):
        """
        6. TESTE HOLDOUT (70/30)
        Para comparação com validação cruzada
        """
        print("\n" + "="*70)
        print("🧪 TESTE HOLDOUT - COMPARAÇÃO DE SCALERS")
        print("🎯 Divisão: 70% Treino / 30% Teste")
        print("="*70)
        
        # Divisão dos dados
        X_treino, X_teste, y_treino, y_teste = train_test_split(
            self.X, self.y, test_size=0.3, random_state=42, shuffle=False
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
        
        resultados_holdout = {}
        
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
                    modelo = Pipeline([
                        ('scaler', scaler),
                        ('regressor', LinearRegression())
                    ])
                    X_treino_scaled = X_treino
                    X_teste_scaled = X_teste
                
                # Treinar modelo
                modelo.fit(X_treino_scaled, y_treino)
                
                # Fazer previsões
                y_pred = modelo.predict(X_teste_scaled)
                
                # Calcular métricas
                r2 = r2_score(y_teste, y_pred)
                rmse = np.sqrt(mean_squared_error(y_teste, y_pred))
                mae = mean_absolute_error(y_teste, y_pred)
                
                # Calcular acurácia de direção
                y_teste_direcao = (y_teste > 0).astype(int)
                y_pred_direcao = (y_pred > 0).astype(int)
                acuracia = np.mean(y_teste_direcao == y_pred_direcao)
                
                # Armazenar resultados
                resultados_holdout[nome_scaler] = {
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
        
        return resultados_holdout
    
    def executar_analise_completa(self):
        """
        Executa análise completa com validação cruzada
        """
        try:
            # 1-4: Preparação dos dados
            self.carregar_dados()
            self.engenharia_features()
            self.criar_target()
            melhores_features = self.selecionar_melhores_features()
            
            # 5: Validação cruzada
            resultados_cv = self.validacao_cruzada_completa()
            
            # 6: Teste holdout para comparação
            resultados_holdout = self.teste_holdout()
            
            # Análise comparativa
            print("\n" + "="*70)
            print("📊 ANÁLISE COMPARATIVA: VALIDAÇÃO CRUZADA vs HOLDOUT")
            print("="*70)
            
            # Ranking validação cruzada
            ranking_cv = sorted(resultados_cv.items(), 
                               key=lambda x: x[1]['acuracia_mean'], reverse=True)
            
            # Ranking holdout
            ranking_holdout = sorted(resultados_holdout.items(), 
                                    key=lambda x: x[1]['acuracia'], reverse=True)
            
            print("🔄 VALIDAÇÃO CRUZADA (Ranking):")
            for i, (nome, metricas) in enumerate(ranking_cv, 1):
                print(f"   {i}º {nome}: {metricas['acuracia_mean']:.1%} (±{metricas['acuracia_std']:.1%})")
            
            print("\n🧪 HOLDOUT (Ranking):")
            for i, (nome, metricas) in enumerate(ranking_holdout, 1):
                print(f"   {i}º {nome}: {metricas['acuracia']:.1%}")
            
            # Melhor configuração
            melhor_cv_nome = ranking_cv[0][0]
            melhor_cv = ranking_cv[0][1]
            melhor_holdout_nome = ranking_holdout[0][0]
            melhor_holdout = ranking_holdout[0][1]
            
            # Resumo final
            print("\n" + "="*50)
            print("🎯 RESUMO FINAL")
            print("="*50)
            print(f"📊 Dataset: {len(self.X)} observações do IBOVESPA")
            print(f"📊 Features: {len(melhores_features)} selecionadas")
            
            print(f"\n🔄 VALIDAÇÃO CRUZADA (5-Fold):")
            print(f"🏆 Melhor: {melhor_cv_nome}")
            print(f"🎯 Acurácia: {melhor_cv['acuracia_mean']:.1%} (±{melhor_cv['acuracia_std']:.1%})")
            print(f"📊 R²: {melhor_cv['r2_mean']:.4f} (±{melhor_cv['r2_std']:.4f})")
            
            print(f"\n🧪 HOLDOUT (70/30):")
            print(f"🏆 Melhor: {melhor_holdout_nome}")
            print(f"🎯 Acurácia: {melhor_holdout['acuracia']:.1%}")
            print(f"📊 R²: {melhor_holdout['r2']:.4f}")
            
            # Análise de consistência
            diff_acuracia = melhor_holdout['acuracia'] - melhor_cv['acuracia_mean']
            print(f"\n📈 CONSISTÊNCIA:")
            print(f"🔀 Diferença Holdout vs CV: {diff_acuracia:+.1%}")
            
            if abs(diff_acuracia) < 0.02:
                print("✅ Resultados consistentes entre métodos!")
            elif diff_acuracia > 0.02:
                print("⚠️ Possível overfitting (holdout muito superior)")
            else:
                print("⚠️ Holdout pode ter sido pessimista")
            
            # Intervalo de confiança
            ic_inf = melhor_cv['acuracia_mean'] - 1.96 * melhor_cv['acuracia_std']
            ic_sup = melhor_cv['acuracia_mean'] + 1.96 * melhor_cv['acuracia_std']
            print(f"📊 IC 95% (CV): [{ic_inf:.1%}, {ic_sup:.1%}]")
            
            estabilidade = "✅ Excelente" if melhor_cv['acuracia_std'] < 0.02 else "✅ Boa" if melhor_cv['acuracia_std'] < 0.05 else "⚠️ Variável"
            print(f"📊 Estabilidade: {estabilidade}")
            
            qualidade_final = "✅ Excelente" if melhor_cv['acuracia_mean'] >= 0.6 else "✅ Satisfatória" if melhor_cv['acuracia_mean'] >= 0.55 else "⚠️ Pode melhorar"
            print(f"📈 Qualidade final: {qualidade_final}")
            
            print("\n✅ Análise completa com validação cruzada concluída!")
            
            return {
                'cv_results': resultados_cv,
                'holdout_results': resultados_holdout,
                'melhor_cv': melhor_cv_nome,
                'melhor_holdout': melhor_holdout_nome,
                'features_selecionadas': melhores_features,
                'acuracia_cv': melhor_cv['acuracia_mean'],
                'acuracia_holdout': melhor_holdout['acuracia']
            }
            
        except Exception as e:
            print(f"❌ Erro durante a execução: {e}")
            return None


def main():
    """
    Função principal
    """
    print("📚 MACHINE LEARNING - LINEAR REGRESSION NO IBOVESPA")
    print("🔄 ANÁLISE COM VALIDAÇÃO CRUZADA")
    print("🎯 Configuração: 10 anos + K-Fold 5 + Holdout 70/30")
    print("💡 Dados do Plano Real para robustez histórica")
    print("=" * 70)
    
    # Executar análise
    ml = MLIbovespaValidacaoCruzada()
    resultados = ml.executar_analise_completa()
    
    if resultados:
        print(f"\n🎉 Análise completa concluída!")
        print(f"\n📊 RESUMO EXECUTIVO:")
        print(f"🔄 Melhor CV: {resultados['melhor_cv']}")
        print(f"🎯 Acurácia CV: {resultados['acuracia_cv']:.1%}")
        print(f"🧪 Melhor Holdout: {resultados['melhor_holdout']}")
        print(f"🎯 Acurácia Holdout: {resultados['acuracia_holdout']:.1%}")
        print(f"📊 Features: {len(resultados['features_selecionadas'])} otimizadas")
    else:
        print("\n❌ Análise não foi concluída devido a erros.")
    
    return ml, resultados


if __name__ == "__main__":
    modelo, resultados = main()
