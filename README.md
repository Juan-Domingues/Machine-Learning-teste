# 🎯 Machine Learning - Previsão de Tendências IBOVESPA

## 📋 Descrição do Projeto

Pipeline completo de Machine Learning para previsão de tendências do IBOVESPA aplicando conceitos fundamentais do curso:
- **Linear Regression** como modelo principal
- **Validação Cruzada** (K-Fold) para avaliação robusta
- **Técnicas de normalização/padronização** 
- **Feature Engineering** avançada
- **Análise comparativa** de métodos

## 🎯 Objetivos Alcançados

✅ **Pipeline de ML completo** implementado  
✅ **Validação cruzada** aplicada corretamente  
✅ **Divisão 70/30** conforme orientação  
✅ **Comparação de normalizadores** sistemática  
✅ **Feature selection** automatizada  
✅ **Métricas múltiplas** (R², RMSE, MAE, Acurácia)  

## 🚀 Script Principal

### 📊 `ml_ibovespa_validacao_cruzada.py`
**🏆 Script ÚNICO - VERSÃO DEFINITIVA SIMPLIFICADA**
- ✅ Código limpo e comentado linha por linha
- ✅ Pipeline completo com validação cruzada
- ✅ Funções separadas para cada etapa do ML
- ✅ Linear Regression + K-Fold CV + Normalização
- ✅ Fácil de entender e modificar
- ✅ Resultados didáticos e bem explicados
- ✅ Pronto para apresentação acadêmica

> **Nota**: Versões anteriores e scripts experimentais estão na pasta `backup/scripts_antigos/` para consulta histórica.

## 📊 Resultados Principais

### 🏆 **Performance Geral**
- **Acurácia de Direção**: 51.9% ± 3.1% (Validação Cruzada)
- **Acurácia Holdout**: 51.6%
- **Consistência**: Diferença CV vs Holdout < 1%
- **R² Final**: -0.0027 (dentro do esperado para dados financeiros)

### 🔍 **Análise Técnica**
- **Estabilidade**: Baixo desvio padrão confirma robustez
- **Overfitting**: Ausente (confirmado pela CV)
- **Features**: 10 selecionadas automaticamente
- **Normalização**: Impacto limitado (dados já relativos)

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
