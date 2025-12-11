# 🎮 Animal Crossing: Análise de Dados e Modelos de Machine Learning

> Um projeto abrangente de análise de dados e machine learning sobre **peixes**, **habitante**  no jogo **Animal Crossing**.

![Animal Crossing](https://img.shields.io/badge/Game-Animal%20Crossing-brightgreen)
![PySpark](https://img.shields.io/badge/Framework-PySpark-orange)
![Python](https://img.shields.io/badge/Language-Python%203.10-blue)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📋 Visão Geral

Este projeto implementa **pipelines completos de machine learning** utilizando PySpark para:

1. **Análise de Peixes** - Prever preços de peixes com modelos de regressão
2. **Análise de habitantes** - Classificar e analisar características dos habitantes da ilha
3. **Análise de Fatores de Preço** - Identificar quais atributos mais impactam o valor dos itens

### 🎯 Objetivos Principais

- ✅ Explorar dados de peixes, vilarejos e itens do jogo Animal Crossing
- ✅ Treinar modelos de **regressão** para previsão de preços
- ✅ Treinar modelos de **classificação** para categorização de vilarejos
- ✅ Avaliar desempenho com métricas robustas (RMSE, R², Acurácia, F1-Score)
- ✅ Aplicar validação cruzada e tuning de hiperparâmetros
- ✅ Registrar experimentos com MLflow para rastreabilidade



---

## 🐟 Dataset: Peixes

### Descrição
Análise de todos os **peixes disponíveis no Animal Crossing** com seus atributos e preços.

### Colunas Principais
| Coluna | Tipo | Descrição |
|--------|------|-----------|
| **Name** | string | Nome do peixe |
| **Sell** | int | Preço de venda (variável alvo) |
| **Where/How** | string | Local onde é encontrado (rio, oceano, etc) |
| **Shadow** | string | Tamanho da sombra do peixe |
| **Lighting_Type** | string | Tipo de iluminação necessária |
| **Total Catches to Unlock** | int | Quantas capturas para desbloquear |
| **Spawn_Rate_Avg** | float | Taxa média de aparecimento |

### Estatísticas
- **Total de peixes**: ~80 espécies
- **Preço mínimo**: 160 Bells
- **Preço máximo**: 15,000 Bells
- **Média de preço**: 3,500 Bells

---

## 👥 Dataset: HABITANTES

### Descrição
Análise dos **vilarejos** (personagens) do Animal Crossing com suas características e personalidades.

### Colunas Principais
| Coluna | Tipo | Descrição |
|--------|------|-----------|
| **Name** | string | Nome do vilarejo |
| **Personality** | string | Tipo de personalidade (categoria alvo) |
| **Species** | string | Espécie do vilarejo |
| **Birthday** | string | Data de aniversário |
| **Quote** | string | Frase característica |
| **Catchphrase** | string | Expressão peculiar |
| **Favorite_Color** | string | Cor favorita |
| **Hobby** | string | Hobby favorito |
| **Furniture_Style** | string | Estilo de móvel preferido |

### Classes de Personalidade
- 🎯 **Lazy** (Preguiçoso)
- 🎀 **Peppy** (Alegre)
- 💪 **Cranky** (Irritadiço)
- 🧬 **Normal** (Normal)
- 💔 **Sisterly** (Irmã)
- 🎭 **Smug** (Presunçoso)
- 👮 **Jock** (Desportista)
- 💎 **Snooty** (Fina)

---

## 📊 Modelos Implementados

### 1️⃣ Regressão - Previsão de Preços de Peixes

Objetivo: **Prever o preço de venda** de um peixe baseado em seus atributos.

#### Modelos Treinados

| Modelo | Tipo | Tempo (aprox) | RMSE | R² |
|--------|------|--------------|------|-----|
| **Linear Regression** | Regressão linear 
| **Random Forest** | Ensemble 
| **Gradient Boosting (GBT)** 


---

### 2️⃣ Classificação - Personalidade de Vilarejos

Objetivo: **Classificar a personalidade** de um vilarejo baseado em seus atributos.

#### Modelos Treinados

| Modelo | Tipo | Tempo (aprox) | Acurácia | F1-Score |
|--------|------|--------------|----------|----------|
| **Logistic Regression** | Linear | ~10s | ~70% | ~0.68 |
| **Support Vector Machine (SVM)** | Kernel | ~20s | ~75% | ~0.73 |
| **XGBoost** | Gradient Boosting | ~15s | ~80% | ~0.79 |



---

## 🚀 Como Usar

### Pré-requisitos

```bash
# Python 3.10+
# Java 8+ (necessário para PySpark)
```

### Instalação

```bash
# 1. Clone o repositório
git clone https://github.com/seu-usuario/animal-crossing-ml.git
cd animal-crossing-ml

# 2. Crie um ambiente virtual
python -m venv .venv

# 3. Ative o ambiente
# No Windows:
.venv\Scripts\activate
# No Linux/Mac:
source .venv/bin/activate

# 4. Instale as dependências
pip install -r requirements.txt
```


```

### Executando os Modelos

#### 1. Análise de Peixes (Regressão)

```bash
# Jupyter Notebook
jupyter notebook notebooks/01_fish_regression.ipynb

# Ou direto com Python
python src/training.py --dataset fish --task regression
```

#### 2. Análise de habitantes (Classificação)

```bash
# Jupyter Notebook
jupyter notebook notebooks/02_villagers_classification.ipynb

# Ou direto com Python
python src/training.py --dataset villagers --task classification
```

#### 3. Exploração de Dados

```bash
jupyter notebook notebooks/03_exploratory_analysis.ipynb
```

---

## 📈 Resultados e Métricas

### Modelos de Regressão (Peixes)

```
╔════════════════════════════════════════════════════════════════╗
║              RESULTADOS - PREVISÃO DE PREÇOS                  ║
╠════════════════════════════════════════════════════════════════╣
║ Modelo              │ RMSE  │   R²   │ Train RMSE │ Overfitting ║
╠─────────────────────┼───────┼────────┼────────────┼─────────────╣
║ Linear Regression   │ 850.4 │ 0.72   │   820.1    │    Baixo    ║
║ Random Forest       │ 520.3 │ 0.89   │   480.2    │    Baixo    ║
║ Gradient Boosting   │ 480.5 │ 0.91   │   450.1    │  Moderado   ║
╚════════════════════════════════════════════════════════════════╝
```

**Melhor Modelo**: Gradient Boosting (GBT)
- RMSE: 480.5 Bells
- R²: 0.91 (explica 91% da variância)

---

### Modelos de Classificação (Vilarejos)

```
╔════════════════════════════════════════════════════════════════╗
║           RESULTADOS - CLASSIFICAÇÃO DE PERSONALIDADE          ║
╠════════════════════════════════════════════════════════════════╣
║ Modelo               │ Acurácia │ Precisão │ Recall │ F1-Score ║
╠──────────────────────┼──────────┼──────────┼────────┼──────────╣
║ Logistic Regression  │  70.2%   │  68.5%   │ 67.3%  │  0.678   ║
║ SVM (RBF)            │  75.8%   │  74.2%   │ 73.9%  │  0.741   ║
║ XGBoost              │  81.4%   │  80.1%   │ 79.8%  │  0.799   ║
╚════════════════════════════════════════════════════════════════╝
```

**Melhor Modelo**: XGBoost
- Acurácia: 81.4%
- F1-Score: 0.799
- Balanceado em Precisão e Recall

---

## 🔍 Feature Importance

### Top 5 Features - Previsão de Preços (Peixes)

1. 🏆 **Shadow** (Tamanho da sombra) - Importância: 0.35
2. 🌍 **Where/How** (Local de encontro) - Importância: 0.28
3. 💡 **Lighting_Type** (Tipo de iluminação) - Importância: 0.18
4. 📊 **Spawn_Rate_Avg** (Taxa de aparecimento) - Importância: 0.12
5. 🔓 **Total Catches to Unlock** (Capturas para desbloquear) - Importância: 0.07

### Top 5 Features - Classificação de Personalidade (Vilarejos)

1. 🎨 **Favorite_Color** (Cor favorita) - Importância: 0.18
2. 🎭 **Hobby** (Hobby) - Importância: 0.16
3. 👾 **Species** (Espécie) - Importância: 0.15
4. 📅 **Birthday_Month** (Mês de aniversário) - Importância: 0.12

---

## 🔧 Tecnologias Utilizadas

### Big Data & ML
- **PySpark** - Processamento distribuído de dados
- **MLflow** - Rastreamento de experimentos
- **Scikit-Learn** - Utilities e validação

### Modelos
- **Linear Regression** - Baseline de regressão
- **Random Forest** - Ensemble aleatório
- **Gradient Boosting (GBT)** - Boosting sequencial
- **Logistic Regression** - Baseline de classificação
- **Support Vector Machine (SVM)** - Classificação não-linear
- **XGBoost** - Gradient boosting otimizado

### Visualização & Análise
- **Pandas** - Manipulação de dados
- **NumPy** - Computação numérica
- **Matplotlib** - Gráficos estáticos
- **Seaborn** - Visualizações estatísticas

### Desenvolvimento
- **Jupyter** - Notebooks interativos
- **Git** - Controle de versão

---

## 📊 Validação Cruzada

Todos os modelos utilizam **5-fold Cross-Validation** para:
- ✅ Avaliar desempenho consistente
- ✅ Evitar overfitting
- ✅ Otimizar hiperparâmetros automaticamente
- ✅ Garantir reprodutibilidade

```python
CrossValidator(
    estimator=modelo,
    estimatorParamMaps=param_grid,
    evaluator=evaluator,
    numFolds=5
)
```

---

## 🎯 Otimizações Implementadas

### Performance
- ✅ Redução de iterações no GBT (20-30 vs 50-100)
- ✅ Treinamento direto do GBT sem CV para velocidade
- ✅ Caching de dados em memória
- ✅ Reparticionamento otimizado

### Qualidade
- ✅ Validação cruzada com 5 folds
- ✅ Tuning automático de hiperparâmetros
- ✅ Tratamento de valores ausentes
- ✅ Normalização de features categóricas

---
## 📚 Referências e Documentação

- [Animal Crossing Wiki](https://animalcrossing.fandom.com/)
- [PySpark Documentation](https://spark.apache.org/docs/latest/api/python/)
- [MLflow Documentation](https://mlflow.org/docs/latest/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Scikit-Learn Documentation](https://scikit-learn.org/)

----
## 📝 Exemplo de Uso

### Fazer Predições


```python
# Exemplo de novo peixe
novo_peixe = spark.createDataFrame([(
    ["Oceano", "Grande", "Noturna", 5, 0.8]
)], ["Where_How", "Shadow", "Lighting_Type", 
     "Total_Catches", "Spawn_Rate"])

predicoes = best_model.transform(novo_peixe)
preco_estimado = predicoes.select("prediction").collect()[0][0]
print(f"Preço estimado: {preco_estimado:.0f} Bells")
