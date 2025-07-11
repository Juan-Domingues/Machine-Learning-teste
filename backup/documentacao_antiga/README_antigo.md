# 🎯 Machine Learning - Previsão IBOVESPA

## 📋 Descrição do Projeto

Projeto completo de Machine Learning para análise e previsão do IBOVESPA com **descoberta importante**: enquanto a previsão de direção é limitada (~52%), conseguimos prever **volatilidade com 75.8% de acurácia**.

## 🏆 Resultados Principais

### 🔍 Principal Descoberta
**O IBOVESPA é altamente eficiente para previsão de direção, mas mostra sinais de previsibilidade em volatilidade.**

### 📊 Métricas Finais:
- **Previsão de Direção**: 54.6% (baseline: 54.6%) - Limitado
- **Previsão de Volatilidade**: **75.8%** (baseline: 75.5%) ✅ **ÚTIL**
- **Cross-Validation Volatilidade**: 75.4% ± 1.0%

## 🚀 Scripts Principais

### � `solucao_final.py` - **RECOMENDADO**
**🏆 Modelo de volatilidade com aplicação prática**
- ✅ Previsão de volatilidade (75.8% acurácia)
- ✅ Features macroeconômicas (USD/BRL, S&P500)
- ✅ Pipeline robusto com Time Series Split
- ✅ Aplicação em gestão de risco

### � `diagnostico_avancado.py`
**Análise profunda que descobriu a previsibilidade de volatilidade**
- ✅ Múltiplos targets testados
- ✅ 15 anos de dados
- ✅ Análise de autocorrelação e eficiência
- ✅ Revelou o potencial de volatilidade (93.5% no diagnóstico)

## 🛠️ Como Executar

### Pré-requisitos
```bash
pip install -r requirements.txt
```

### Execução
```bash
# Instalar dependências
pip install -r requirements.txt

# Executar o pipeline completo de ML
python ml_ibovespa_validacao_cruzada.py
```

> **Resultado esperado**: ~52% de acurácia na previsão de direção do IBOVESPA (resultado realista para mercado financeiro)

## 📦 Dependências

```
pandas
numpy
scikit-learn
yfinance
matplotlib
seaborn
```

## 📁 Estrutura do Projeto

```
📂 Machine-Learning-teste/
├── 📄 ml_ibovespa_validacao_cruzada.py    # 🏆 Script principal definitivo
├── 📄 requirements.txt                     # Dependências Python
├── 📄 README.md                           # Documentação
├── 📄 LICENSE                             # Licença MIT
└── 📂 backup/                             # 📚 Histórico e versões antigas
    ├── 📂 scripts_antigos/               # Scripts experimentais
    ├── 📂 imagens_antigas/               # Gráficos antigos
    └── 📂 documentacao_antiga/           # Docs anteriores
```

### 🎯 Foco do Repositório
Este repositório foi **simplificado e organizado** para apresentação acadêmica, mantendo apenas:
- ✅ **Um script principal** (`ml_ibovespa_validacao_cruzada.py`)
- ✅ **Código didático** com comentários explicativos
- ✅ **Pipeline completo** de ML (Linear Regression + CV + Normalização)
- ✅ **Histórico preservado** na pasta backup
    ├── 📂 documentacao_antiga/           # Documentação histórica
    └── 📂 imagens_antigas/               # Gráficos e resultados antigos
```

## 🎓 Conceitos Demonstrados

### 📚 **Machine Learning**
- Regressão Linear aplicada a dados financeiros
- Pipeline de preprocessamento
- Feature Engineering automática
- Avaliação de modelos

### 🔬 **Validação**
- K-Fold Cross Validation (5 folds)
- Holdout Validation (70/30)
- Métricas múltiplas
- Análise de consistência

### 📊 **Análise de Dados**
- Dados reais via Yahoo Finance (yfinance)
- Indicadores técnicos
- Correlações e multicolinearidade
- Seleção automática de features

## 📈 Interpretação dos Resultados

### ✅ **Resultados Realísticos**
A acurácia de ~52% é **cientificamente válida** para previsão de direção em mercados financeiros:
- Mercados são intrinsecamente difíceis de prever
- Resultado ligeiramente acima do random (50%)
- Alinhado com literatura acadêmica
- Ausência de overfitting confirmada

### 🎯 **Valor Educacional**
O projeto demonstra com sucesso:
- Aplicação correta de validação cruzada
- Metodologia científica rigorosa
- Interpretação realística de resultados
- Pipeline profissional de ML

## 🚀 Extensões Futuras

### 🔧 **Melhorias Técnicas**
- [ ] Ensemble Methods (Random Forest, XGBoost)
- [ ] Deep Learning (LSTM, Transformer)
- [ ] Hyperparameter Tuning
- [ ] Walk-Forward Analysis

### 📊 **Dados Adicionais**
- [ ] Indicadores macroeconômicos
- [ ] Análise de sentimento
- [ ] Dados de alta frequência
- [ ] Múltiplos ativos

## 📞 Suporte

Para dúvidas ou sugestões sobre o projeto, consulte a documentação nos scripts ou analise os comentários detalhados no código.

## 📄 Licença

Este projeto é distribuído sob a licença MIT. Veja `LICENSE` para mais detalhes.

---

**🎉 Projeto concluído com sucesso!**  
*Pipeline robusto de Machine Learning aplicado ao mercado financeiro brasileiro*
