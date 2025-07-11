# Exemplos de Uso - PETR4 ML Predictor

## 🎯 Cenários de Uso

### 1. Análise Rápida (Execução Básica)
```python
from simple_trend_predictor import FinancialTrendAnalyzer

# Criar analisador
analisador = FinancialTrendAnalyzer('PETR4.SA')

# Executar análise completa
acuracia, modelo = analisador.executar_analise()

print(f"Acurácia obtida: {acuracia:.2%}")
```

### 2. Análise Customizada
```python
# Criar dados com período específico
analisador = FinancialTrendAnalyzer('PETR4.SA')
analisador.criar_dados_simulados(num_dias=1000)  # 1000 dias

# Calcular indicadores
analisador.calcular_indicadores_tecnicos()

# Criar target personalizado
analisador.criar_variavel_target()

# Ver distribuição das classes
print(analisador.data['Target'].value_counts())
```

### 3. Análise de Features
```python
# Executar até seleção de features
analisador = FinancialTrendAnalyzer('PETR4.SA')
analisador.criar_dados_simulados()
analisador.calcular_indicadores_tecnicos()
analisador.criar_variavel_target()

# Selecionar features
features = analisador.preparar_features()

# Ver features mais importantes
print("Top 10 features selecionadas:")
for i, feat in enumerate(features[:10]):
    print(f"{i+1:2d}. {feat}")
```

### 4. Treinamento Manual
```python
# Preparar dados
analisador = FinancialTrendAnalyzer('PETR4.SA')
analisador.criar_dados_simulados()
analisador.calcular_indicadores_tecnicos()
analisador.criar_variavel_target()
analisador.preparar_features()

# Dividir dados
X_treino, X_teste, y_treino, y_teste = analisador.dividir_dados()

# Treinar modelo
modelo = analisador.treinar_modelo(X_treino, y_treino)

# Avaliar
acuracia, y_pred, y_prob = analisador.avaliar_modelo(X_teste, y_teste)
```

### 5. Análise de Predições
```python
# Após treinar o modelo
probabilidades = modelo.predict_proba(X_teste)

# Análise de confiança
for i, (real, pred, prob) in enumerate(zip(y_teste, y_pred, probabilidades)):
    confianca = max(prob) * 100
    print(f"Dia {i+1}: Real={real}, Pred={pred}, Confiança={confianca:.1f}%")
```

## 📊 Interpretação dos Resultados

### Acurácia por Faixa
- **90-100%**: Possível overfitting, verificar regularização
- **80-90%**: Excelente resultado, modelo bem calibrado
- **70-80%**: Bom resultado, adequado para uso prático
- **60-70%**: Resultado moderado, considerar melhorias
- **<60%**: Resultado insatisfatório, revisar metodologia

### Análise da Matriz de Confusão
```python
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Exemplo de interpretação
cm = confusion_matrix(y_teste, y_pred)
print("Matriz de Confusão:")
print(f"Verdadeiros Negativos: {cm[0,0]}")
print(f"Falsos Positivos: {cm[0,1]}")
print(f"Falsos Negativos: {cm[1,0]}")
print(f"Verdadeiros Positivos: {cm[1,1]}")
```

### Feature Importance
```python
# Analisar importância das features
coefs = abs(modelo.named_steps['classificador'].coef_[0])
importancia = pd.DataFrame({
    'feature': features_selecionadas,
    'importancia': coefs
}).sort_values('importancia', ascending=False)

print("Top 5 features mais importantes:")
for i, row in importancia.head(5).iterrows():
    print(f"{row['feature']}: {row['importancia']:.3f}")
```

## 🔧 Modificações Avançadas

### 1. Alterar Algoritmo
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Substituir regressão logística por Random Forest
# Na função treinar_modelo, substitua:
# LogisticRegression(...) por:
RandomForestClassifier(n_estimators=100, random_state=42)

# Ou SVM:
SVC(kernel='rbf', probability=True, random_state=42)
```

### 2. Adicionar Novos Indicadores
```python
# Na função calcular_indicadores_tecnicos, adicione:

# Stochastic Oscillator
def stochastic_oscillator(data, periodo=14):
    low_min = data['Low'].rolling(periodo).min()
    high_max = data['High'].rolling(periodo).max()
    k_percent = 100 * (data['Close'] - low_min) / (high_max - low_min)
    return k_percent

data['Stochastic'] = stochastic_oscillator(data)

# Williams %R
def williams_r(data, periodo=14):
    high_max = data['High'].rolling(periodo).max()
    low_min = data['Low'].rolling(periodo).min()
    wr = -100 * (high_max - data['Close']) / (high_max - low_min)
    return wr

data['Williams_R'] = williams_r(data)
```

### 3. Ensemble de Modelos
```python
from sklearn.ensemble import VotingClassifier

# Criar ensemble
rf = RandomForestClassifier(n_estimators=100, random_state=42)
svm = SVC(kernel='rbf', probability=True, random_state=42)
lr = LogisticRegression(random_state=42)

ensemble = VotingClassifier(
    estimators=[('rf', rf), ('svm', svm), ('lr', lr)],
    voting='soft'
)

# Treinar ensemble
ensemble.fit(X_treino, y_treino)
```

### 4. Validação Temporal Avançada
```python
# Walk-forward validation
def walk_forward_validation(data, features, initial_size=500, step=30):
    results = []
    
    for i in range(initial_size, len(data) - step, step):
        # Treino
        X_train = data[features].iloc[:i]
        y_train = data['Target'].iloc[:i]
        
        # Teste
        X_test = data[features].iloc[i:i+step]
        y_test = data['Target'].iloc[i:i+step]
        
        # Modelo
        model = LogisticRegression(random_state=42)
        model.fit(X_train, y_train)
        
        # Avaliação
        accuracy = model.score(X_test, y_test)
        results.append(accuracy)
    
    return results
```

## 🎓 Casos de Estudo

### Caso 1: Mercado em Alta
```python
# Simular mercado em alta (trend positivo)
# Modificar na função criar_dados_simulados:
drift = 0.001  # Aumentar drift para simular bull market
```

### Caso 2: Mercado Volátil
```python
# Simular alta volatilidade
# Modificar na função criar_dados_simulados:
volatility = 0.03  # Aumentar volatilidade
```

### Caso 3: Dados Desbalanceados
```python
# Verificar balanceamento
print(data['Target'].value_counts(normalize=True))

# Se muito desbalanceado, ajustar threshold:
threshold = 0.001  # Threshold menor = mais eventos de alta
```

## 📈 Métricas de Negócio

### Simulação de Trading
```python
def simular_trading(data, modelo, capital_inicial=10000):
    capital = capital_inicial
    posicao = 0
    historico = []
    
    for i in range(len(data)):
        # Predição
        pred = modelo.predict(data.iloc[i:i+1])[0]
        preco = data['Close'].iloc[i]
        
        # Estratégia simples
        if pred == 1 and posicao == 0:  # Comprar
            posicao = capital / preco
            capital = 0
        elif pred == 0 and posicao > 0:  # Vender
            capital = posicao * preco
            posicao = 0
        
        valor_total = capital + (posicao * preco)
        historico.append(valor_total)
    
    return historico

# Uso
historico = simular_trading(dados_teste, modelo)
retorno_total = (historico[-1] / historico[0] - 1) * 100
print(f"Retorno total: {retorno_total:.2f}%")
```

### Análise de Risco
```python
def calcular_metricas_risco(retornos):
    # Sharpe Ratio
    sharpe = np.mean(retornos) / np.std(retornos) * np.sqrt(252)
    
    # Maximum Drawdown
    cumulative = (1 + retornos).cumprod()
    max_dd = (cumulative / cumulative.cummax() - 1).min()
    
    # Volatilidade anualizada
    vol_anual = np.std(retornos) * np.sqrt(252)
    
    return {
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'volatilidade_anual': vol_anual
    }
```

## 🚀 Próximos Passos

1. **Dados Reais**: Integrar com APIs financeiras
2. **Mais Features**: Sentiment analysis, dados macroeconômicos
3. **Deep Learning**: Redes neurais LSTM/GRU
4. **Backtesting**: Simulação histórica completa
5. **Deploy**: API REST para predições em tempo real

---

*Estes exemplos demonstram a flexibilidade e extensibilidade do sistema desenvolvido*
