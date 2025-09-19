# Inteligência Artificial - Trabalho de Regressão e Classificação

Este repositório contém a implementação completa de um trabalho de Inteligência Artificial focado em **Modelos Preditivos** para **Regressão** e **Classificação**.

## 📋 Sobre o Trabalho

O trabalho é composto por duas etapas principais que utilizam conceitos de IA baseados em modelos preditivos que realizam seu processo de aprendizagem através da minimização de uma função custo (loss function). Ambas as etapas utilizam o paradigma supervisionado para aprender a partir de pares (vetor de características e variável dependente).

## 🎯 Etapas do Trabalho

### 1. **Tarefa de Regressão** (3,0 pts)
- **Dados**: `aerogerador.dat` - Velocidade do vento vs. Potência gerada
- **Modelos Implementados**:
  - MQO Tradicional
  - MQO Regularizado (Tikhonov) com λ = {0, 0.25, 0.5, 0.75, 1}
  - Média de Valores Observáveis
- **Validação**: Monte Carlo (R=500 rodadas, 80/20 split)
- **Métrica**: Residual Sum of Squares (RSS)

### 2. **Tarefa de Classificação** (7,0 pts)
- **Dados**: `EMGsDataset.csv` - Sinais de eletromiografia facial (5 classes)
- **Modelos Implementados**:
  - MQO Tradicional
  - Classificador Gaussiano Tradicional
  - Classificador Gaussiano (Cov. de todo cj. treino)
  - Classificador Gaussiano (Cov. Agregada)
  - Classificador de Bayes Ingênuo
  - Classificador Gaussiano Regularizado (Friedman) com λ = {0.25, 0.5, 0.75, 1.0}
- **Validação**: Monte Carlo (R=500 rodadas, 80/20 split)
- **Métrica**: Acurácia (Taxa de Acerto)

## 📁 Estrutura do Projeto

```
├── data/
│   ├── aerogerador.dat          # Dados de regressão
│   └── EMGsDataset.csv          # Dados de classificação
├── regrecao.py                  # Implementação completa da regressão
├── regrecao_simples.py          # Versão simplificada da regressão
├── classificacao.py             # Implementação completa da classificação
├── resultados_regressao_aerogerador.txt    # Resultados da regressão
├── resultados_classificacao_emg.txt        # Resultados da classificação
├── requirements.txt             # Dependências do projeto
└── README.md                    # Este arquivo
```

## 🚀 Como Executar

### Pré-requisitos
```bash
pip install -r requirements.txt
```

### Executar Regressão
```bash
python regrecao_simples.py
```

### Executar Classificação
```bash
python classificacao.py
```

## 📊 Principais Resultados

### Regressão (Aerogerador)
- **Melhor Modelo**: MQO Regularizado λ=0.25
- **RSS Médio**: ~0.45
- **Regularização**: Efetiva para λ ≤ 0.5

### Classificação (EMG)
- **Melhor Modelo**: Classificador Gaussiano Regularizado (Friedman λ=0.25)
- **Acurácia**: 98.98% ± 0.09%
- **Observação**: Modelos gaussianos superaram significativamente o MQO tradicional

## 🛠️ Tecnologias Utilizadas

- **Python 3.x**
- **NumPy**: Operações matemáticas e álgebra linear
- **Matplotlib**: Visualizações e gráficos
- **Algoritmos**: MQO, Regularização Tikhonov, Classificadores Gaussianos, Bayes Ingênuo

## 📈 Características Técnicas

- **Validação Robusta**: 500 simulações Monte Carlo
- **Proteções**: Contra divisão por zero e matrizes singulares
- **Visualizações**: Gráficos de dispersão, boxplots, histogramas
- **Análise Estatística**: Média, desvio-padrão, valores min/max
- **Documentação**: Código bem comentado e estruturado

## 👨‍💻 Autor

**Davi Alencar** - Trabalho de Inteligência Artificial

## 📝 Licença

Este projeto é parte de um trabalho acadêmico de Inteligência Artificial.