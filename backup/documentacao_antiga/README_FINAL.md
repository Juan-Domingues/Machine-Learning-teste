# 🎯 Machine Learning - Previsão IBOVESPA

## 📋 Resumo do Projeto

Projeto de Machine Learning para previsão da direção do IBOVESPA utilizando **técnicas de regressão** com conversão para classificação binária.

**Objetivo**: Prever se o fechamento do dia seguinte será maior ou menor que o dia atual.  
**Meta**: Acurácia mínima de 75% nos últimos 30 dias.  
**Abordagem**: Regressão Linear + Ensemble + Análise de Correlação.

## 📁 Estrutura do Repositório

```
├── src/                          # Código fonte modular
│   ├── main.py                   # Pipeline principal
│   ├── data_utils.py             # Utilitários de dados
│   ├── feature_engineering.py    # Engenharia de features
│   ├── correlation_analysis.py   # Análise de correlação
│   ├── model_utils.py           # Modelos e avaliação
│   └── config.py                # Configurações
├── docs/                        # Documentação
├── results/                     # Resultados e relatórios
├── notebooks/                   # Jupyter notebooks
├── backup/                      # Versões anteriores
├── requirements.txt             # Dependências
└── README.md                    # Este arquivo
```

## 🚀 Como Executar

### Instalação
```bash
pip install -r requirements.txt
```

### Execução Principal
```bash
cd src
python main.py
```

### Execução do Pipeline Completo (versão anterior)
```bash
python ml_ibovespa_validacao_cruzada.py
```

## 🔬 Metodologia

### 1. **Carregamento de Dados**
- Dados históricos do IBOVESPA via Yahoo Finance
- 10 anos de dados históricos
- Divisão temporal: últimos 30 dias para teste

### 2. **Engenharia de Features**
- **Features Básicas** (14): Médias móveis, RSI, volatilidade, volume
- **Features Avançadas** (17): Interações, momentum, regime de mercado
- **Total**: 31 features

### 3. **Análise de Correlação**
- Correlação com target de regressão (retorno)
- Correlação com target de direção (alta/baixa)
- Detecção e remoção de multicolinearidade
- Seleção das top 12 features

### 4. **Modelagem**
- **Ensemble** de 4 modelos de regressão:
  - Linear Regression
  - Ridge Regression (α=1.0)
  - Lasso Regression (α=0.01)
  - ElasticNet (α=0.01, l1_ratio=0.5)
- **Pipeline**: StandardScaler + VotingRegressor
- **Validação**: TimeSeriesSplit (5 folds)

### 5. **Avaliação**
- **Métrica Principal**: Acurácia de direção
- **Métricas Auxiliares**: R², MSE
- **Baseline**: Sempre classe majoritária
- **Teste Final**: Últimos 30 dias

## 📊 Resultados

### Features Mais Importantes
1. **Volume_Price_Momentum**: Interação volume-retorno
2. **Aceleracao**: Mudança na velocidade dos retornos
3. **MM20**: Média móvel de 20 dias
4. **Canal_Momentum**: Movimento no canal de preços
5. **Volatilidade_Relativa**: Volatilidade normalizada

### Performance Típica
- **Acurácia**: ~52-60% (vs baseline ~53%)
- **R²**: ~-0.01 a 0.02
- **Correlação**: Baixa (~0.03-0.13)

### Limitações Identificadas
- **Meta irrealista**: 75% é muito alta para previsão de direção
- **Mercado eficiente**: Baixa previsibilidade inerente
- **Features limitadas**: Apenas dados técnicos

## 🎯 Conclusões

### ✅ Técnicas Implementadas Corretamente
- ✅ Regressão Linear e variantes
- ✅ Ensemble de modelos
- ✅ Validação cruzada temporal
- ✅ Análise de correlação
- ✅ Seleção de features
- ✅ Remoção de multicolinearidade

### 📊 Meta de 75%: Análise Crítica
A meta de **75% de acurácia** mostrou-se **irrealista** baseado em:

1. **Benchmarks Acadêmicos**: 55-65% é considerado bom
2. **Eficiência de Mercado**: IBOVESPA é relativamente eficiente
3. **Limitações dos Dados**: Apenas indicadores técnicos
4. **Variabilidade Temporal**: Períodos diferentes têm comportamentos distintos

### 📈 Meta Realista Sugerida
- **60-65%** seria uma meta mais apropriada
- **>55%** já demonstra valor preditivo
- **Consistência** é mais importante que picos de acurácia

## 💡 Melhorias Futuras

### Dados Externos
- 📊 Dados macroeconômicos (SELIC, câmbio, inflação)
- 📰 Sentiment analysis de notícias
- 🌍 Indicadores internacionais (S&P 500, commodities)
- 📈 Dados intraday de maior frequência

### Modelos Avançados
- 🤖 Random Forest / Gradient Boosting
- 🧠 Redes Neurais / LSTM
- 🔍 Ensemble mais sofisticado
- ⚡ Modelos online/adaptativos

### Estratégias Alternativas
- 🎯 Previsão de volatilidade (mais viável)
- 📊 Classificação multi-classe (alta, neutra, baixa)
- 🔄 Modelos específicos por regime de mercado
- 📅 Horizontes de previsão diferentes (2-5 dias)

## 📚 Referências Técnicas

- **Validação Temporal**: TimeSeriesSplit evita look-ahead bias
- **Ensemble**: Reduz overfitting e melhora robustez
- **Correlação**: Seleção baseada em relevância estatística
- **Regressão→Classificação**: Abordagem híbrida inovadora

## 👨‍💻 Autor

Projeto desenvolvido como parte de estudos em Machine Learning com foco em:
- ✅ Técnicas de regressão supervisionada
- ✅ Engenharia de features para dados financeiros
- ✅ Validação rigorosa de modelos temporais
- ✅ Análise crítica de viabilidade

---

*"A meta de 75% ensinou mais sobre os limites da previsibilidade do mercado do que sobre técnicas de ML."* 📈
