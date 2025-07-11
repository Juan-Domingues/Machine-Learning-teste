"""
Machine Learning Aplicado ao IBOVESPA - Versão Simplificada
Baseado nos conceitos fundamentais do curso:
- Modelos de Classificação
- Validação Cruzada
- Métricas de Avaliação
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

# Imports do Scikit-Learn (conceitos do curso - REGRESSÃO)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression

def main():
    print("📚 MACHINE LEARNING - LINEAR REGRESSION NO IBOVESPA")
    print("🔬 TESTE DE NORMALIZAÇÃO E PADRONIZAÇÃO DOS DADOS")
    print("🎯 Métricas: R², RMSE, MAE + ACURÁCIA")
    print("=" * 60)
    
    # Criar modelo
    ml = MLIbovespaRegressao()
    
    # Carregar dados básicos
    ml.carregar_dados()
    ml.engenharia_features()
    ml.criar_target()
    
    # Executar teste de normalização
    resultados_norm, melhor_scaler = ml.comparar_normalizacao()
    
    print(f"\n🎉 Análise de normalização concluída!")
    print(f"🏆 Melhor scaler: {melhor_scaler}")
    
    return ml, resultados_normsemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Configuração para gráficos mais bonitos
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class MLIbovespaRegressao:
    """
    Classe para Machine Learning de REGRESSÃO no IBOVESPA
    Aplicando os modelos de regressão do curso de ML
    """
    
    def __init__(self):
        self.data = None
        self.X = None
        self.y = None
        self.modelos = {}
        self.resultados = {}
        
    def carregar_dados(self, anos=3):
        """
        1. CARREGAMENTO DE DADOS
        Baixa dados reais do IBOVESPA via Yahoo Finance
        """
        print("="*50)
        print("1. CARREGANDO DADOS DO IBOVESPA")
        print("="*50)
        
        # Definir período
        end_date = datetime.now()
        start_date = end_date - timedelta(days=anos * 365)
        
        print(f"📊 Baixando dados de {start_date.strftime('%Y-%m-%d')} até {end_date.strftime('%Y-%m-%d')}")
        
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
        Criação de indicadores técnicos simples (conceitos básicos)
        """
        print("\n" + "="*50)
        print("2. CRIANDO FEATURES (INDICADORES TÉCNICOS)")
        print("="*50)
        
        # Features básicas de preço
        self.data['Retorno_1d'] = self.data['Close'].pct_change()
        self.data['Retorno_5d'] = self.data['Close'].pct_change(5)
        
        # Médias móveis (conceito fundamental)
        self.data['MA_5'] = self.data['Close'].rolling(5).mean()
        self.data['MA_20'] = self.data['Close'].rolling(20).mean()
        self.data['MA_50'] = self.data['Close'].rolling(50).mean()
        
        # RSI simplificado
        delta = self.data['Close'].diff()
        ganhos = delta.where(delta > 0, 0).rolling(14).mean()
        perdas = (-delta.where(delta < 0, 0)).rolling(14).mean()
        self.data['RSI'] = 100 - (100 / (1 + ganhos / perdas))
        
        # Volatilidade
        self.data['Volatilidade'] = self.data['Retorno_1d'].rolling(20).std()
        
        # Volume relativo
        self.data['Volume_MA'] = self.data['Volume'].rolling(20).mean()
        self.data['Volume_Ratio'] = self.data['Volume'] / self.data['Volume_MA']
        
        # Bandas de Bollinger simples
        self.data['BB_Superior'] = self.data['MA_20'] + (2 * self.data['Close'].rolling(20).std())
        self.data['BB_Inferior'] = self.data['MA_20'] - (2 * self.data['Close'].rolling(20).std())
        
        # Posição relativa nas Bandas
        self.data['Posicao_BB'] = (self.data['Close'] - self.data['BB_Inferior']) / (self.data['BB_Superior'] - self.data['BB_Inferior'])
        
        print("Features criadas:")
        features = ['Retorno_1d', 'Retorno_5d', 'MA_5', 'MA_20', 'MA_50', 'RSI', 
                   'Volatilidade', 'Volume_Ratio', 'Posicao_BB']
        for i, feat in enumerate(features, 1):
            print(f"  {i}. {feat}")
        
        return features
    
    def criar_target(self):
        """
        3. CRIAÇÃO DA VARIÁVEL TARGET
        REGRESSÃO: Prever o retorno futuro (valor contínuo)
        """
        print("\n" + "="*50)
        print("3. CRIANDO VARIÁVEL TARGET (REGRESSÃO)")
        print("="*50)
        
        # Retorno futuro (próximo dia) - VALOR CONTÍNUO para regressão
        self.data['Retorno_Futuro'] = self.data['Close'].shift(-1) / self.data['Close'] - 1
        
        # Multiplicar por 100 para trabalhar com percentuais
        self.data['Target'] = self.data['Retorno_Futuro'] * 100
        
        print(f"📊 Estatísticas do Target (Retorno % futuro):")
        print(f"   Média: {self.data['Target'].mean():.4f}%")
        print(f"   Desvio: {self.data['Target'].std():.4f}%")
        print(f"   Min: {self.data['Target'].min():.4f}%")
        print(f"   Max: {self.data['Target'].max():.4f}%")
        
        return self.data['Target']
    
    def preparar_dataset(self):
        """
        4. PREPARAÇÃO DO DATASET
        Selecionar features e limpar dados
        """
        print("\n" + "="*50)
        print("4. PREPARANDO DATASET PARA REGRESSÃO")
        print("="*50)
        
        # Features para o modelo
        feature_names = ['Retorno_1d', 'Retorno_5d', 'RSI', 'Volatilidade', 
                        'Volume_Ratio', 'Posicao_BB']
        
        # Remover dados faltantes
        self.data = self.data.dropna()
        
        # Separar X e y
        self.X = self.data[feature_names]
        self.y = self.data['Target']
        
        print(f"📊 Dataset final:")
        print(f"   - Observações: {len(self.X)}")
        print(f"   - Features: {len(feature_names)}")
        print(f"   - Target (retorno %): {self.y.mean():.4f} ± {self.y.std():.4f}")
        
        print(f"\n📋 Features selecionadas:")
        for i, feat in enumerate(feature_names, 1):
            print(f"   {i}. {feat}")
        
        return self.X, self.y
    
    def dividir_dados(self):
        """
        5. DIVISÃO TREINO/TESTE
        Conceito fundamental: separar dados para validação
        """
        print("\n" + "="*50)
        print("5. DIVISÃO TREINO/TESTE")
        print("="*50)
        
        # Divisão temporal (mais realista para séries temporais)
        split_idx = int(len(self.X) * 0.8)
        
        X_treino = self.X.iloc[:split_idx]
        X_teste = self.X.iloc[split_idx:]
        y_treino = self.y.iloc[:split_idx]
        y_teste = self.y.iloc[split_idx:]
        
        print(f"📊 Divisão dos dados:")
        print(f"   - Treino: {len(X_treino)} observações ({len(X_treino)/len(self.X):.1%})")
        print(f"   - Teste:  {len(X_teste)} observações ({len(X_teste)/len(self.X):.1%})")
        
        return X_treino, X_teste, y_treino, y_teste
    
    def treinar_modelos(self, X_treino, y_treino):
        """
        6. TREINAMENTO DO MELHOR MODELO
        Focando apenas na Linear Regression (melhor resultado)
        """
        print("\n" + "="*50)
        print("6. TREINANDO LINEAR REGRESSION (MELHOR MODELO)")
        print("="*50)
        
        # Apenas o melhor modelo: Linear Regression
        self.modelos = {
            'Linear Regression': Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', LinearRegression())
            ])
        }
        
        print("🤖 Modelo selecionado:")
        print("   ✅ Linear Regression (melhor performance anterior)")
        
        # Treinar o modelo
        print("\n🔄 Treinando Linear Regression...")
        self.modelos['Linear Regression'].fit(X_treino, y_treino)
        
        print("✅ Linear Regression treinado com sucesso!")
        
        return self.modelos
    
    def validacao_cruzada(self, X_treino, y_treino):
        """
        7. VALIDAÇÃO CRUZADA
        Validando apenas a Linear Regression
        """
        print("\n" + "="*50)
        print("7. VALIDAÇÃO CRUZADA - LINEAR REGRESSION")
        print("="*50)
        
        cv_results = {}
        
        print("🔄 Executando validação cruzada (5-fold)...")
        modelo = self.modelos['Linear Regression']
        
        # Usar R² como métrica para regressão
        scores = cross_val_score(modelo, X_treino, y_treino, cv=5, scoring='r2')
        cv_results['Linear Regression'] = {
            'mean': scores.mean(),
            'std': scores.std(),
            'scores': scores
        }
        print(f"   Linear Regression: R² = {scores.mean():.3f} (±{scores.std():.3f})")
        
        print(f"\n✅ Validação cruzada concluída!")
        
        return cv_results
    
    def avaliar_modelos(self, X_teste, y_teste):
        """
        8. AVALIAÇÃO NO CONJUNTO DE TESTE
        Métricas de REGRESSÃO + ACURÁCIA do curso
        """
        print("\n" + "="*50)
        print("8. AVALIAÇÃO - LINEAR REGRESSION (MELHOR MODELO)")
        print("="*50)
        
        # Focar apenas no melhor modelo: Linear Regression
        modelo = self.modelos['Linear Regression']
        
        # Previsões
        y_pred = modelo.predict(X_teste)
        
        # Métricas de regressão
        mse = mean_squared_error(y_teste, y_pred)
        mae = mean_absolute_error(y_teste, y_pred)
        r2 = r2_score(y_teste, y_pred)
        rmse = np.sqrt(mse)
        
        # ADICIONAR ACURÁCIA: Converter para classificação binária
        # Se o retorno real e predito têm o mesmo sinal = acerto
        acertos_direcao = ((y_teste > 0) == (y_pred > 0)).sum()
        total_predicoes = len(y_teste)
        acuracia = acertos_direcao / total_predicoes
        
        # Classificação binária mais detalhada
        y_real_binario = (y_teste > 0).astype(int)  # 1 = alta, 0 = baixa
        y_pred_binario = (y_pred > 0).astype(int)   # 1 = alta, 0 = baixa
        
        # Contar acertos por categoria
        altas_corretas = ((y_real_binario == 1) & (y_pred_binario == 1)).sum()
        baixas_corretas = ((y_real_binario == 0) & (y_pred_binario == 0)).sum()
        total_altas = (y_real_binario == 1).sum()
        total_baixas = (y_real_binario == 0).sum()
        
        self.resultados['Linear Regression'] = {
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'mse': mse,
            'acuracia': acuracia,
            'y_pred': y_pred,
            'acertos_direcao': acertos_direcao,
            'total_predicoes': total_predicoes,
            'altas_corretas': altas_corretas,
            'baixas_corretas': baixas_corretas,
            'total_altas': total_altas,
            'total_baixas': total_baixas
        }
        
        print(f"📊 LINEAR REGRESSION - Resultados:")
        print(f"   R²: {r2:.3f}")
        print(f"   RMSE: {rmse:.4f}%")
        print(f"   MAE: {mae:.4f}%")
        print(f"   🎯 ACURÁCIA: {acuracia:.3f} ({acuracia:.1%})")
        
        print(f"\n📈 Análise de Direção:")
        print(f"   Acertos totais: {acertos_direcao}/{total_predicoes}")
        print(f"   Altas corretas: {altas_corretas}/{total_altas} ({altas_corretas/total_altas:.1%} se total_altas > 0)")
        print(f"   Baixas corretas: {baixas_corretas}/{total_baixas} ({baixas_corretas/total_baixas:.1%} se total_baixas > 0)")
        
        # Verificar meta de acurácia
        if acuracia >= 0.60:
            print("✅ EXCELENTE: Acurácia >= 60% para mercado financeiro!")
        elif acuracia >= 0.55:
            print("✅ BOM: Acurácia >= 55% é satisfatória")
        else:
            print("⚠️ Pode melhorar: Acurácia abaixo de 55%")
        
        return self.resultados
    
    def relatorio_detalhado(self, X_teste, y_teste):
        """
        9. RELATÓRIO DETALHADO
        Métricas completas da Linear Regression
        """
        print("\n" + "="*50)
        print("9. RELATÓRIO DETALHADO - LINEAR REGRESSION")
        print("="*50)
        
        # Dados do modelo
        resultado = self.resultados['Linear Regression']
        y_pred = resultado['y_pred']
        
        print(f"📋 Relatório Completo - Linear Regression:")
        print(f"\n📊 Métricas de Regressão:")
        print(f"   R² (Coef. Determinação): {resultado['r2']:.4f}")
        print(f"   RMSE (Erro Quadrático): {resultado['rmse']:.4f}%")
        print(f"   MAE (Erro Absoluto): {resultado['mae']:.4f}%")
        
        print(f"\n🎯 Métricas de Acurácia:")
        print(f"   Acurácia de Direção: {resultado['acuracia']:.4f} ({resultado['acuracia']:.1%})")
        print(f"   Acertos: {resultado['acertos_direcao']}/{resultado['total_predicoes']}")
        
        print(f"\n📈 Análise Detalhada:")
        print(f"   Previsões de Alta: {resultado['altas_corretas']}/{resultado['total_altas']}")
        print(f"   Previsões de Baixa: {resultado['baixas_corretas']}/{resultado['total_baixas']}")
        
        # Análise dos resíduos
        residuos = y_teste - y_pred
        print(f"\n� Análise dos Resíduos:")
        print(f"   Média dos resíduos: {residuos.mean():.4f}%")
        print(f"   Desvio dos resíduos: {residuos.std():.4f}%")
        print(f"   Resíduo máximo: {residuos.abs().max():.4f}%")
        
        return 'Linear Regression'
    
    def visualizar_resultados(self, X_teste, y_teste):
        """
        10. VISUALIZAÇÕES
        Gráficos para análise dos resultados de REGRESSÃO
        """
        print("\n" + "="*50)
        print("10. CRIANDO VISUALIZAÇÕES - REGRESSÃO")
        print("="*50)
        
        # Melhor modelo
        melhor_nome = max(self.resultados.keys(), key=lambda x: self.resultados[x]['r2'])
        y_pred = self.resultados[melhor_nome]['y_pred']
        
        # Criar figura com subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Análise dos Resultados - Regressão IBOVESPA', fontsize=16, fontweight='bold')
        
        # 1. R² dos modelos
        nomes = list(self.resultados.keys())
        r2_scores = [self.resultados[nome]['r2'] for nome in nomes]
        
        bars = ax1.bar(range(len(nomes)), r2_scores, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
        ax1.set_title('R² dos Modelos de Regressão')
        ax1.set_ylabel('R² Score')
        ax1.set_xticks(range(len(nomes)))
        ax1.set_xticklabels(nomes, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # Adicionar valores nas barras
        for bar, r2 in zip(bars, r2_scores):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{r2:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Scatter Plot: Real vs Predito
        ax2.scatter(y_teste, y_pred, alpha=0.6, color='blue')
        ax2.plot([y_teste.min(), y_teste.max()], [y_teste.min(), y_teste.max()], 'r--', lw=2)
        ax2.set_xlabel('Retorno Real (%)')
        ax2.set_ylabel('Retorno Predito (%)')
        ax2.set_title(f'Real vs Predito - {melhor_nome}')
        ax2.grid(True, alpha=0.3)
        
        # Adicionar R² no gráfico
        r2 = self.resultados[melhor_nome]['r2']
        ax2.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax2.transAxes, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=12, fontweight='bold')
        
        # 3. Resíduos
        residuos = y_teste - y_pred
        ax3.scatter(y_pred, residuos, alpha=0.6, color='green')
        ax3.axhline(y=0, color='red', linestyle='--')
        ax3.set_xlabel('Valores Preditos (%)')
        ax3.set_ylabel('Resíduos (%)')
        ax3.set_title('Análise dos Resíduos')
        ax3.grid(True, alpha=0.3)
        
        # 4. Série temporal das previsões
        ultimos_30_dias = -30
        datas = self.data.index[ultimos_30_dias:]
        y_teste_30 = y_teste.iloc[ultimos_30_dias:]
        y_pred_30 = y_pred[ultimos_30_dias:]
        
        ax4.plot(datas, y_teste_30, 'b-', linewidth=2, label='Real', alpha=0.7)
        ax4.plot(datas, y_pred_30, 'r--', linewidth=2, label='Predito', alpha=0.7)
        ax4.set_title('Retornos: Real vs Predito (Últimos 30 dias)')
        ax4.set_ylabel('Retorno (%)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('ml_ibovespa_regressao_resultados.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📊 Gráficos de regressão criados e salvos em 'ml_ibovespa_regressao_resultados.png'")
        
        return None
    
    def executar_pipeline_completo(self):
        """
        PIPELINE COMPLETO DE MACHINE LEARNING
        Executa todas as etapas do processo
        """
        print("🚀 INICIANDO PIPELINE DE MACHINE LEARNING - IBOVESPA")
        print("📚 Aplicando REGRESSÃO com os modelos do curso")
        print("")
        
        try:
            # 1-4: Preparação dos dados
            self.carregar_dados()
            self.engenharia_features()
            self.criar_target()
            X, y = self.preparar_dataset()
            
            # 5: Divisão dos dados
            X_treino, X_teste, y_treino, y_teste = self.dividir_dados()
            
            # 6-7: Treinamento e validação
            self.treinar_modelos(X_treino, y_treino)
            self.validacao_cruzada(X_treino, y_treino)
            
            # 8-10: Avaliação e resultados
            self.avaliar_modelos(X_teste, y_teste)
            melhor_modelo = self.relatorio_detalhado(X_teste, y_teste)
            self.visualizar_resultados(X_teste, y_teste)
            
            # Resumo final
            print("\n" + "="*50)
            print("🎯 RESUMO FINAL")
            print("="*50)
            print(f"📊 Dataset: {len(X)} observações do IBOVESPA")
            print(f"🏆 Melhor modelo: {melhor_modelo}")
            print(f"🎯 R²: {self.resultados[melhor_modelo]['r2']:.3f}")
            print(f"🎯 Acurácia: {self.resultados[melhor_modelo]['acuracia']:.1%}")
            print(f"📈 Qualidade: {'✅ Excelente' if self.resultados[melhor_modelo]['acuracia'] >= 0.6 else '✅ Satisfatória' if self.resultados[melhor_modelo]['acuracia'] >= 0.55 else '⚠️ Pode melhorar'}")
            print("")
            print("✅ Pipeline de Machine Learning (LINEAR REGRESSION) concluído com sucesso!")
            
            return melhor_modelo
            
        except Exception as e:
            print(f"❌ Erro durante a execução: {e}")
            return None

    def comparar_normalizacao(self):
        """
        Compara diferentes técnicas de normalização/padronização
        """
        print("\n" + "="*70)
        print("🔬 TESTE DE NORMALIZAÇÃO - COMPARAÇÃO DE SCALERS")
        print("="*70)
        
        # Preparar dados
        X, y = self.preparar_dataset()
        X_treino, X_teste, y_treino, y_teste = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        
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


def main():
    """
    Função principal para executar o projeto
    """
    print("📚 MACHINE LEARNING - LINEAR REGRESSION NO IBOVESPA")
    print("� Modelo otimizado com melhor performance")
    print("🎯 Métricas: R², RMSE, MAE + ACURÁCIA")
    print("=" * 60)
    
    # Criar e executar o modelo
    ml = MLIbovespaRegressao()
    melhor_modelo = ml.executar_pipeline_completo()
    
    if melhor_modelo:
        acuracia = ml.resultados['Linear Regression']['acuracia']
        r2 = ml.resultados['Linear Regression']['r2']
        print(f"\n🎉 Projeto concluído!")
        print(f"📊 Linear Regression: R² = {r2:.3f}, Acurácia = {acuracia:.1%}")
    else:
        print("\n❌ Projeto não foi concluído devido a erros.")
    
    return ml


if __name__ == "__main__":
    modelo = main()
