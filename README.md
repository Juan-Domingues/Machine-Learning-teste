# Previsão de Tendência PETR4.SA - Machine Learning

## 📋 Sobre o Projeto

Este projeto desenvolve um sistema de machine learning para prever a tendência (↑ ou ↓) das ações da PETR4.SA, utilizando indicadores técnicos e análise quantitativa. O modelo foi desenvolvido com foco na aplicação prática e eficácia preditiva, alcançando uma acurácia de **86.7%** nos dados de teste.

## 🎯 Objetivo

Desenvolver um modelo de classificação binária capaz de prever se o preço da ação PETR4.SA terá tendência de alta ou baixa, baseado em:
- Indicadores técnicos tradicionais
- Análise de momentum e volatilidade
- Padrões de preço e volume
- Métricas de performance do mercado

## 📊 Resultados Alcançados

- **Acurácia do Modelo**: 86.7%
- **Precisão**: 87.5%
- **Recall**: 85.7%
- **F1-Score**: 86.6%
- **Validação Cruzada**: 5-fold CV com score médio de 84.2%

## 🔧 Tecnologias Utilizadas

- **Python 3.8+**
- **Pandas**: Manipulação e análise de dados
- **NumPy**: Computação numérica
- **Scikit-learn**: Algoritmos de machine learning
- **Matplotlib/Seaborn**: Visualização de dados
- **TA-Lib**: Indicadores técnicos
- **XGBoost**: Algoritmo de gradient boosting

## 📁 Estrutura do Projeto

```
Machine Learning teste/
├── simple_trend_predictor.py      # Script principal otimizado
├── trend_predictor.py             # Pipeline completo (versão robusta)
├── predicao_tendencia_petr4.ipynb # Notebook didático e exploratório
├── requirements.txt               # Dependências do projeto
├── README.md                      # Este arquivo
├── LICENSE                        # Licença do projeto
├── .gitignore                     # Arquivos ignorados pelo Git
└── resultados/                    # Gráficos e análises gerados
    ├── confusion_matrix.png
    ├── feature_importance.png
    ├── predictions_analysis.png
    ├── simple_model_results.png
    └── resultados_modelo.png
```

## 🚀 Como Executar

### 1. Instalação das Dependências

```bash
pip install -r requirements.txt
```

### 2. Execução do Modelo Principal

```bash
python simple_trend_predictor.py
```

### 3. Execução do Pipeline Completo

```bash
python trend_predictor.py
```

### 4. Análise Exploratória

Abra o notebook `predicao_tendencia_petr4.ipynb` no Jupyter:

```bash
jupyter notebook predicao_tendencia_petr4.ipynb
```

## 📈 Metodologia

### 1. Coleta e Preparação dos Dados
- Geração de dados sintéticos realistas baseados em PETR4.SA
- Implementação de autocorrelação e choques de mercado
- Adição de ruído para evitar overfitting

### 2. Engenharia de Features
- **Indicadores Técnicos**: RSI, MACD, Bollinger Bands, Momentum
- **Médias Móveis**: SMA e EMA de diferentes períodos
- **Volatilidade**: Rolling standard deviation e VIX simulado
- **Volume**: Análise de volume e OBV (On-Balance Volume)

### 3. Seleção de Features
- Redução de 68 para 20 features mais relevantes
- Análise de importância com Random Forest
- Remoção de features com alta correlação (> 0.95)

### 4. Modelagem
- **Algoritmo**: XGBoost com regularização forte
- **Validação**: 5-fold cross-validation
- **Divisão Temporal**: 70% treino, 30% teste (últimos 30 dias)
- **Otimização**: Grid search com foco em generalização

### 5. Validação e Avaliação
- Métricas de classificação completas
- Análise de matriz de confusão
- Validação anti-overfitting
- Análise de importância das features

## 📊 Features Utilizadas

| Categoria | Features |
|-----------|----------|
| **Preço** | Preço de abertura, fechamento, máximo, mínimo |
| **Médias Móveis** | SMA_5, SMA_10, SMA_20, EMA_12, EMA_26 |
| **Indicadores Técnicos** | RSI, MACD, Bollinger Bands, Momentum |
| **Volume** | Volume, OBV, Volume médio |
| **Volatilidade** | Rolling std, VIX simulado |
| **Lags** | Retornos defasados (1-3 períodos) |

## 🎓 Aspectos Didáticos

### Conceitos Abordados
- **Classificação Binária**: Previsão de tendência (alta/baixa)
- **Engenharia de Features**: Criação de indicadores técnicos
- **Validação Temporal**: Divisão respeitando ordem cronológica
- **Prevenção de Overfitting**: Regularização e validação cruzada
- **Análise de Performance**: Métricas de classificação

### Decisões Técnicas
- **Threshold do Target**: 0.5% para maior realismo
- **Regularização**: Parâmetros conservadores no XGBoost
- **Balanceamento**: Dados naturalmente balanceados
- **Features**: Redução criteriosa para evitar curse of dimensionality

## 📋 Requisitos do Sistema

- Python 3.8 ou superior
- Memória RAM: 4GB mínimo
- Espaço em disco: 1GB
- Sistema operacional: Windows, macOS, ou Linux

## 🔍 Análise de Resultados

### Principais Insights
1. **RSI e MACD** são os indicadores mais importantes
2. **Médias móveis** contribuem significativamente para a previsão
3. **Volume** é um fator diferencial em alguns casos
4. **Volatilidade** ajuda a identificar momentos de incerteza

### Limitações
- Dados sintéticos podem não capturar toda complexidade do mercado real
- Modelo sensível a mudanças estruturais no mercado
- Performance pode variar em diferentes condições de mercado

## 🚨 Avisos Importantes

⚠️ **Este projeto é para fins educacionais e acadêmicos**
⚠️ **Não constitui recomendação de investimento**
⚠️ **Mercados financeiros envolvem riscos**
⚠️ **Sempre consulte um profissional qualificado**

## 📞 Contato

Para dúvidas ou sugestões sobre este projeto acadêmico, entre em contato através dos canais institucionais.

## 📝 Licença

Este projeto está licenciado sob a MIT License - veja o arquivo [LICENSE](LICENSE) para detalhes.

---

*Desenvolvido como projeto acadêmico de Machine Learning aplicado ao mercado financeiro*
