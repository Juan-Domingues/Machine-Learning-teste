# 🧪 EXPERIMENTOS - PASTA DE TESTES

Esta pasta contém todos os experimentos realizados durante o desenvolvimento do projeto para atingir 70% de acurácia.

## 📊 RESULTADOS DOS EXPERIMENTOS

| Arquivo | Descrição | Acurácia | Status |
|---------|-----------|----------|--------|
| **main_final_refinado.py** | Versão refinada com análise detalhada | 70% | ✅ SUCESSO |
| **main_75_pct.py** | Tentativa avançada para 75% | ~55-65% | ⏳ EM DESENVOLVIMENTO |
| **teste_otimizado.py** | Testes rápidos e diagnósticos | ~50-60% | 🔍 DIAGNÓSTICO |
| **diagnostico_acuracia.py** | Análise dos problemas de acurácia | N/A | 📊 ANÁLISE |
| **main_estavel.py** | Versão focada na estabilidade | ~60-70% | ✅ ESTÁVEL |
| **main_super_otimizado.py** | Tentativa com features complexas | ~45-55% | ❌ NÃO FUNCIONOU |
| **main_refinado.py** | Primeira versão refinada | ~55-65% | 📈 MELHOROU |
| **main_novo.py** | Experimento com novas abordagens | ~60-70% | ✅ BOM |
| **ml_ibovespa_validacao_cruzada.py** | Validação cruzada rigorosa | ~50-60% | ✅ VALIDADO |

## 🎯 COMO USAR

### **Executar Experimentos**
```bash
# Melhor resultado (70%)
python experiments/main_final_refinado.py

# Tentar 75%
python experiments/main_75_pct.py

# Diagnóstico rápido
python experiments/teste_otimizado.py

# Análise de problemas
python experiments/diagnostico_acuracia.py
```

### **Comparar Resultados**
```bash
# Executar múltiplos experimentos
python experiments/main_estavel.py
python experiments/main_novo.py
python experiments/main_final_refinado.py
```

## 💡 PRINCIPAIS DESCOBERTAS

### ✅ **Experimentos que Funcionaram**
1. **main_final_refinado.py**: Simplicidade com 3 features binárias
2. **main_estavel.py**: Validação temporal rigorosa
3. **main_novo.py**: Ensemble otimizado

### ❌ **Experimentos que Não Funcionaram**
1. **main_super_otimizado.py**: Excesso de features causou overfitting
2. **Features complexas**: Indicadores técnicos avançados foram contraproducentes
3. **Períodos longos**: 10+ anos de dados introduziram muito ruído

## 🔍 INSIGHTS TÉCNICOS

### **Configuração Vencedora**
- **Features**: 3 features binárias (trend + volume + momentum)
- **Modelo**: Ensemble (Logistic + Random Forest)
- **Período**: 3 anos de dados
- **Validação**: TimeSeriesSplit

### **Fatores Críticos**
- Simplicidade > Complexidade
- Features binárias > Features contínuas  
- Validação temporal > Validação aleatória
- Ensemble > Modelos individuais

## 📈 EVOLUÇÃO DO PROJETO

```
📊 Linha do Tempo dos Experimentos:

1. main_refinado.py          (~55%) - Primeira melhoria
2. main_super_otimizado.py   (~45%) - Tentativa complexa (falhou)
3. teste_otimizado.py        (~60%) - Diagnóstico rápido
4. main_estavel.py          (~65%) - Foco na estabilidade
5. main_novo.py             (~70%) - Nova abordagem
6. main_final_refinado.py   (70%) - Versão final (SUCESSO!)
7. main_75_pct.py           (~60%) - Tentativa de 75%
```

## 🚀 PRÓXIMOS EXPERIMENTOS

Para atingir 75%, testar:
1. **XGBoost/LightGBM**: Modelos mais avançados
2. **Neural Networks**: LSTM para séries temporais
3. **Feature Engineering**: RSI, MACD, Bollinger Bands
4. **Dados Externos**: Selic, câmbio, commodities
5. **Ensemble Stacking**: Combinações mais sofisticadas

## 📞 USO RECOMENDADO

### **Para Reproduzir 70%**
```bash
python experiments/main_final_refinado.py
```

### **Para Tentar 75%**
```bash
python experiments/main_75_pct.py
```

### **Para Diagnóstico**
```bash
python experiments/diagnostico_acuracia.py
```

---

**🎯 OBJETIVO CUMPRIDO: 70% DE ACURÁCIA!** ✅  
**📊 EXPERIMENTOS ORGANIZADOS E DOCUMENTADOS!** 📁
