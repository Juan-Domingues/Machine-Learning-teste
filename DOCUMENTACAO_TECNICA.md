# Documentação Técnica - Previsão PETR4.SA

## Arquitetura do Sistema

### Visão Geral
O sistema é composto por três componentes principais:
1. **Gerador de Dados Sintéticos**: Simula dados realistas de PETR4.SA
2. **Pipeline de Features**: Extrai e seleciona indicadores técnicos
3. **Modelo de Classificação**: XGBoost otimizado para previsão de tendência

### Fluxo de Dados

```
Dados Sintéticos → Engenharia de Features → Seleção → Modelo → Previsão
```

## Detalhes Técnicos

### 1. Geração de Dados Sintéticos

**Características implementadas:**
- Movimento browniano geométrico como base
- Autocorrelação temporal (AR(1))
- Choques de mercado aleatórios
- Ruído gaussiano calibrado
- Volume correlacionado com volatilidade

**Parâmetros:**
- Drift: 0.0005 (0.05% ao dia)
- Volatilidade: 0.02 (2% ao dia)
- Autocorrelação: 0.1
- Frequência de choques: 2%

### 2. Indicadores Técnicos

#### Médias Móveis
- **SMA (Simple Moving Average)**: 5, 10, 20 períodos
- **EMA (Exponential Moving Average)**: 12, 26 períodos

#### Osciladores
- **RSI (Relative Strength Index)**: Período 14
- **MACD**: (12, 26, 9)
- **Momentum**: Período 10

#### Bandas e Volatilidade
- **Bollinger Bands**: Período 20, desvios 2
- **Rolling Std**: Período 20
- **VIX Simulado**: Baseado em volatilidade rolling

#### Volume
- **OBV (On-Balance Volume)**: Acumulativo
- **Volume Ratio**: Volume atual / média de 20 períodos

### 3. Seleção de Features

#### Critérios de Seleção
1. **Importância**: Score > 0.01 no Random Forest
2. **Correlação**: Remoção de features com correlação > 0.95
3. **Relevância Financeira**: Manutenção de indicadores fundamentais

#### Features Finais (20)
```python
final_features = [
    'open', 'high', 'low', 'close', 'volume',
    'sma_5', 'sma_10', 'sma_20', 'ema_12', 'ema_26',
    'rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower',
    'momentum', 'rolling_std', 'vix_sim', 'obv', 'volume_ratio'
]
```

### 4. Modelo XGBoost

#### Hiperparâmetros Otimizados
```python
best_params = {
    'n_estimators': 100,        # Número de árvores
    'max_depth': 3,             # Profundidade máxima
    'learning_rate': 0.1,       # Taxa de aprendizado
    'subsample': 0.8,           # Amostragem de dados
    'colsample_bytree': 0.8,    # Amostragem de features
    'reg_alpha': 0.1,           # Regularização L1
    'reg_lambda': 1.0,          # Regularização L2
    'random_state': 42
}
```

#### Justificativas
- **max_depth=3**: Evita overfitting em dados temporais
- **reg_lambda=1.0**: Regularização forte para generalização
- **subsample=0.8**: Reduz variância do modelo
- **learning_rate=0.1**: Convergência estável

### 5. Validação e Métricas

#### Divisão Temporal
- **Treino**: 70% dos dados (cronologicamente)
- **Teste**: 30% dos dados (últimos períodos)
- **Validação**: 5-fold cross-validation no conjunto de treino

#### Métricas Principais
- **Acurácia**: Proporção de predições corretas
- **Precisão**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1-Score**: Média harmônica entre precisão e recall

#### Anti-Overfitting
- Alerta para acurácia > 90% (suspeita de overfitting)
- Comparação treino vs. validação cruzada
- Análise de distribuição de features

## Decisões de Design

### 1. Dados Sintéticos vs. Dados Reais
**Decisão**: Usar dados sintéticos
**Justificativa**: 
- Controle total sobre características estatísticas
- Evita problemas de direitos autorais
- Permite calibração específica para fins didáticos

### 2. Threshold do Target
**Decisão**: 0.5% para definir alta/baixa
**Justificativa**:
- Filtra ruído de movimentos pequenos
- Cria classes mais balanceadas
- Realismo para trading strategies

### 3. Redução de Features
**Decisão**: De 68 para 20 features
**Justificativa**:
- Curse of dimensionality
- Redução de multicolinearidade
- Foco em indicadores mais relevantes

### 4. Regularização Forte
**Decisão**: reg_lambda=1.0, max_depth=3
**Justificativa**:
- Dados financeiros são ruidosos
- Prevenção de overfitting
- Melhor generalização

## Limitações Técnicas

### 1. Regime Shifts
O modelo assume estacionariedade estatística, mas mercados podem mudar de regime (bull/bear markets).

### 2. Features Limitadas
Não inclui:
- Dados fundamentais da empresa
- Sentiment analysis
- Indicadores macroeconômicos
- Dados de order book

### 3. Lookback Bias
Mesmo com divisão temporal, existe risco de lookahead bias na engenharia de features.

### 4. Dados Sintéticos
Podem não capturar complexidades específicas do mercado brasileiro ou da PETR4.SA.

## Possíveis Melhorias

### 1. Features Adicionais
- Indicadores macroeconômicos
- Preço do petróleo (específico para PETR4)
- Câmbio USD/BRL
- Índices setoriais

### 2. Modelos Ensemble
- Combinação XGBoost + Random Forest + SVM
- Stacking com meta-learner
- Voting classifier

### 3. Feature Engineering Avançada
- Features baseadas em tempo (dia da semana, mês)
- Interações entre indicadores
- Features de lag variável

### 4. Validação Robusta
- Walk-forward analysis
- Purged cross-validation
- Monte Carlo validation

## Considerações de Deployment

### 1. Dados em Tempo Real
- API para coleta de dados
- Pipeline de atualização
- Monitoramento de data drift

### 2. Retreinamento
- Frequência de retreinamento
- Detecção de degradação
- A/B testing de modelos

### 3. Risk Management
- Position sizing
- Stop-loss automático
- Monitoramento de drawdown

## Conclusão

O sistema apresenta uma abordagem metodologicamente sólida para previsão de tendências, com foco na prevenção de overfitting e generalização. A arquitetura modular permite futuras extensões e melhorias mantendo a robustez do core do sistema.
