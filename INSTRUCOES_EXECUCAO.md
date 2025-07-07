# Guia de Execução - PETR4 ML Predictor

## 🚀 Início Rápido

### Pré-requisitos
- Python 3.8 ou superior
- pip (gerenciador de pacotes Python)
- 4GB RAM disponível
- Conexão com internet (para instalação de dependências)

### Instalação em 3 Passos

#### 1. Clone/Download do Projeto
```bash
# Se usando Git
git clone <url-do-repositorio>
cd "Machine Learning teste"

# Ou extraia o ZIP e navegue até a pasta
```

#### 2. Instale as Dependências
```bash
pip install -r requirements.txt
```

#### 3. Execute o Modelo
```bash
python simple_trend_predictor.py
```

## 📋 Execução Detalhada

### Opção 1: Script Principal (Recomendado)
```bash
python simple_trend_predictor.py
```

**O que acontece:**
- Gera dados sintéticos de PETR4.SA
- Calcula indicadores técnicos
- Treina modelo XGBoost
- Avalia performance
- Salva gráficos na pasta `resultados/`

**Tempo de execução:** ~30-60 segundos

### Opção 2: Pipeline Completo
```bash
python trend_predictor.py
```

**O que acontece:**
- Versão mais robusta com validações extras
- Grid search de hiperparâmetros
- Análises adicionais
- Mais gráficos e métricas

**Tempo de execução:** ~2-5 minutos

### Opção 3: Notebook Interativo
```bash
# Instale Jupyter se não tiver
pip install jupyter

# Execute o notebook
jupyter notebook predicao_tendencia_petr4.ipynb
```

**O que acontece:**
- Análise exploratória passo a passo
- Explicações detalhadas
- Visualizações interativas
- Ideal para estudo e compreensão

## 📊 Interpretando os Resultados

### Console Output
```
=== RESULTADOS DO MODELO ===
Acurácia: 86.70%
Precisão: 87.50%
Recall: 85.71%
F1-Score: 86.60%
Cross-Validation Score: 84.20% (+/- 3.15%)
```

### Arquivos Gerados
- `resultados/simple_model_results.png` - Métricas principais
- `resultados/confusion_matrix.png` - Matriz de confusão
- `resultados/feature_importance.png` - Importância das features
- `resultados/predictions_analysis.png` - Análise das predições

### Como Ler os Gráficos

#### 1. Matriz de Confusão
```
        Predito
        Alta  Baixa
Real Alta  85    14    ← 85 acertos de alta
    Baixa  13    87    ← 87 acertos de baixa
```

#### 2. Feature Importance
- Barras mostram quais indicadores são mais importantes
- RSI e MACD geralmente no topo
- Features com score < 0.01 são menos relevantes

#### 3. Análise de Predições
- Distribuição de probabilidades
- Confiança do modelo nas predições
- Histogramas de alta vs. baixa

## 🔧 Personalização

### Modificar Parâmetros do Modelo
Edite `simple_trend_predictor.py`:

```python
# Linha ~200
best_params = {
    'n_estimators': 200,     # Era 100, aumentar para mais árvores
    'max_depth': 4,          # Era 3, aumentar para mais complexidade
    'learning_rate': 0.05,   # Era 0.1, diminuir para mais precisão
}
```

### Alterar Período de Dados
```python
# Linha ~50
num_days = 365  # Era 300, aumentar para mais dados
```

### Modificar Target Threshold
```python
# Linha ~180
threshold = 0.01  # Era 0.005, aumentar para movimentos maiores
```

### Adicionar Novas Features
```python
# Linha ~120, adicione após as existentes
data['nova_feature'] = sua_funcao(data)
```

## 🐛 Solução de Problemas

### Erro: "ModuleNotFoundError"
```bash
# Instale a dependência específica
pip install <nome-do-modulo>

# Ou reinstale tudo
pip install -r requirements.txt --force-reinstall
```

### Erro: "Memory Error"
```python
# Reduza o tamanho dos dados
num_days = 200  # Em vez de 300+
```

### Erro: "Convergence Warning"
```python
# Aumente o número de iterações
'n_estimators': 200,  # Em vez de 100
```

### Gráficos não aparecem
```bash
# Instale backend de gráficos
pip install matplotlib --upgrade
```

### Performance baixa
```python
# Verifique se os dados estão balanceados
print(data['target'].value_counts())

# Ajuste o threshold se necessário
threshold = 0.005  # Movimentos menores = mais dados
```

## 📈 Interpretação dos Resultados

### Acurácia Boa (80-90%)
✅ Modelo está funcionando bem
✅ Features são relevantes
✅ Parâmetros adequados

### Acurácia Muito Alta (>95%)
⚠️ Possível overfitting
⚠️ Verifique data leakage
⚠️ Considere mais regularização

### Acurácia Baixa (<70%)
❌ Modelo pode estar underfitting
❌ Features insuficientes
❌ Threshold muito restritivo

### Desbalanceamento
```python
# Se classes muito desbalanceadas
target_counts = data['target'].value_counts()
print(f"Alta: {target_counts[1]}, Baixa: {target_counts[0]}")

# Ajuste threshold para balancear
```

## 🔍 Validação dos Resultados

### 1. Cross-Validation
- CV Score deve estar próximo da acurácia
- Diferença > 10% indica overfitting
- Desvio padrão baixo indica estabilidade

### 2. Matriz de Confusão
- Diagonal principal deve ter valores altos
- Off-diagonal (erros) devem ser baixos
- Classes balanceadas têm valores similares

### 3. Feature Importance
- Top 5 features devem fazer sentido financeiro
- RSI, MACD, SMAs geralmente importantes
- Features com score 0 podem ser removidas

## 📚 Próximos Passos

### Para Estudo
1. Execute o notebook cell por cell
2. Modifique parâmetros e observe mudanças
3. Adicione novas features
4. Teste outros algoritmos

### Para Desenvolvimento
1. Implemente dados reais via API
2. Adicione mais indicadores técnicos
3. Experimente ensemble methods
4. Implemente backtesting

### Para Produção
1. Configure pipeline de dados em tempo real
2. Implemente monitoramento de drift
3. Configure retreinamento automático
4. Adicione logging e alertas

## 📞 Suporte

Em caso de problemas:
1. Verifique se todos os arquivos estão presentes
2. Confirme versão do Python (3.8+)
3. Reinstale dependências
4. Consulte a documentação técnica

---

*Última atualização: 2024*
*Versão: 1.0*
