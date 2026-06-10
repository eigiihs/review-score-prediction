# O Peso das Palavras — Análise de Sentimento em Avaliações de Marketplace

> Projeto desenvolvido durante a graduação em Inteligência e Análise de Dados

## !! Observações sobre o Repositório

Este repositório está organizado em duas branches:

- **`main`**: contém a versão completa do projeto, com todas as bibliotecas utilizadas — incluindo o `pysentimiento`. O arquivo `sentiment_analysis.ipynb` presente nessa branch **não pode ser visualizado diretamente pelo GitHub** (conflito na exportação do notebook). Para acessá-lo, faça o **download do arquivo** e abra localmente.

- **`giovanna`**: branch criada para contornar o problema de visualização no GitHub. O notebook foi reenviado e pode ser visualizado diretamente na plataforma. No entanto, esta versão está **incompleta** — não contém a etapa com a biblioteca `pysentimiento`.

---

## → Problema de Negócio

Em marketplaces digitais, nem sempre os clientes atribuem notas coerentes com seus comentários — ou até mesmo deixam de avaliar numericamente um produto. Surge então a necessidade de desenvolver modelos capazes de interpretar automaticamente o sentimento de um texto e convertê-lo em uma nota de satisfação estimada.

## → Objetivo

Construir um modelo preditivo capaz de estimar a nota de um produto (de 1 a 5 estrelas) com base no conteúdo textual dos comentários deixados pelos consumidores.

O desafio técnico proposto foi utilizar bibliotecas de **análise de sentimento** (NLP) para encontrar a relação entre o "tom" do texto e a "nota" atribuída, avaliando qual abordagem melhor representa a percepção do consumidor.

---

## Estrutura do Repositório

```
├── bronze/                  # Bases originais do dataset
├── silver/
│   ├── df_avaliacao.parquet         # Base tratada e limpa
│   ├── df_vader.parquet             # Resultados do VADER
│   ├── df_leia.parquet              # Resultados do LeIA
│   ├── df_roberta.parquet           # Resultado do XLM-RoBERTa
|   ├── df_bert.parquet              # Resultados do BERT      
│   └── df_pysentimiento.parquet     # Resultados do PySentimiento
└── sentiment_analysis.ipynb         # Notebook principal (branch main — download necessário)
```

---

## Bibliotecas Utilizadas

```bash
pip install pandas tqdm matplotlib seaborn nltk leia-br transformers torch pysentimiento
```

| Biblioteca | Finalidade |
|---|---|
| `pandas`, `numpy` | Manipulação e análise de dados |
| `matplotlib`, `seaborn` | Visualizações e gráficos |
| `nltk` | Stopwords e VADER |
| `leia-br` | Análise de sentimento em português (LeIA) |
| `transformers` + `torch` | Modelo BERT multilingual |
| `pysentimiento` | Modelo treinado em português (Twitter/PT-BR) |
| `scipy`, `sklearn` | Correlações e matriz de confusão |
| `tqdm` | Barra de progresso nas inferências |

---

## Etapas do Projeto

### 1. Leitura das Tabelas

Carregamento das bases da pasta `bronze/`. O foco principal foi a tabela de **avaliações**, que contém os comentários textuais dos consumidores e as notas atribuídas. Nesta etapa foi realizada uma análise exploratória inicial verificando shape, valores nulos e duplicatas de cada tabela.

---

### 2. Transformação e Limpeza

**2.1 Tipificação dos dados**
Conversão das colunas para os tipos corretos: `string`, `int8` (economia de memória) e `datetime`.

**2.2 Unificação dos campos textuais**
Como as avaliações possuem campo de título (`review_comment_title`) e corpo (`review_comment_message`) separados, foi criada a coluna unificada `review_text`, concatenando ambos. _Registros sem nenhum conteúdo textual foram identificados e removidos._

**2.3 Padronização do texto**
Foi criada a coluna `review_text_modificado` com as seguintes transformações:
- Conversão para minúsculas
- Remoção de acentos
- Remoção de pontuações
- Remoção de espaços extras
- Remoção de **stopwords** em português (palavras como "o", "a", "de", "para", que não carregam sentimento relevante)

A base tratada foi salva em formato `.parquet` na pasta `silver/`.

---

### 3. Análise de Sentimento

Foram aplicados **5 modelos** de análise de sentimento, cada um com características distintas. Para modelos que retornam scores contínuos (entre -1 e 1) como o VADER e LeIA, foi utilizada uma função de mapeamento para estrelas (1 a 5):

| Score | Estrelas |
|---|---|
| ≤ -0.6 | ⭐ 1 |
| -0.6 a -0.2 | ⭐⭐ 2 |
| -0.2 a 0.2 | ⭐⭐⭐ 3 |
| 0.2 a 0.6 | ⭐⭐⭐⭐ 4 |
| > 0.6 | ⭐⭐⭐⭐⭐ 5 |

Cada modelo foi avaliado tanto no **texto original** quanto no **texto modificado** (sem stopwords), gerando comparações de desempenho.

---

#### 3.1 VADER

O **VADER (Valence Aware Dictionary and sEntiment Reasoner)** é um modelo baseado em léxico e regras linguísticas, com suporte a intensificadores e negações. Originalmente desenvolvido para inglês, foi utilizado aqui como baseline. Retorna um score contínuo (`compound`) entre -1 e 1.

---

#### 3.2 LeIA

O **LeIA** é uma adaptação do VADER para a língua portuguesa. Possui funcionamento idêntico ao VADER, porém com léxico ajustado para o português brasileiro — tornando-o mais adequado para comentários de consumidores brasileiros.

---

#### 3.3 Transformers (XLM-RoBERTa, BERT e PySentimiento)
 
Os modelos Transformer representam o estado de NLP (Processamento de Linguagem Natural). Diferentemente dos modelos baseados em léxico, conseguem compreender contexto, relações semânticas, negações, intensidade emocional e dependências entre palavras devido sua maior complexidade.

 
**BERT Multilingual Sentiment (`nlptown/bert-base-multilingual-uncased-sentiment`)**
Modelo BERT treinado especificamente para prever avaliações em formato de estrelas (1 a 5). Transforma o texto em representações vetoriais contextualizadas e retorna diretamente a classe de estrelas estimada. Por seu treinamento ser voltado a avaliações de produtos, espera-se alta correlação com as notas reais.

**XLM-RoBERTa (`cardiffnlp/twitter-xlm-roberta-base-sentiment`)**
Modelo Transformer multilíngue treinado em diversos idiomas, incluindo português. Capaz de interpretar contexto textual de forma profunda, com melhor tratamento de frases negativas, sarcasmo leve, negações e semântica complexa. Retorna labels de polaridade (`positive`, `negative`, `neutral`) com score de confiança, convertidos para estrelas com a mesma lógica do PySentimiento.

**PySentimiento (pt)**
Modelo baseado em BERT, treinado com dados em português (especialmente tweets). Retorna labels de polaridade (`POS`, `NEG`, `NEU`) com probabilidade associada. Para este projeto, foi aplicada a seguinte conversão:
 
| Label | Score | Estrelas |
|---|---|---|
| negative/NEG | ≥ 0.75 | ⭐ 1 |
| negative/NEG | < 0.75 | ⭐⭐ 2 |
| neutral/NEU | — | ⭐⭐⭐ 3 |
| positive/POS | < 0.75 | ⭐⭐⭐⭐ 4 |
| positive/POS | ≥ 0.75 | ⭐⭐⭐⭐⭐ 5 |

---

### 4. Avaliação dos Modelos

Para cada modelo e cada versão do texto (original vs. modificado), foram gerados:

- **Gráficos de distribuição** das notas reais vs. preditas
- **Correlação de Pearson** (relação linear entre nota real e nota predita)
- **Correlação de Spearman** (relação monotônica, menos sensível a outliers)
- **Matriz de Confusão** (visualização dos acertos e erros por classe de estrela)

---

## 🏆 Conclusões
 
Após a aplicação dos 5 modelos e avaliação por correlação de Pearson e Spearman, os destaques foram:
 
| Métrica | 🥇 Melhor Modelo | Correlação | 🥈 2º Melhor | Correlação |
|---|---|---|---|---|
| Pearson | **BERT** | 0.7284 | PySentimiento | 0.7182 |
| Spearman | **PySentimiento** | 0.7055 | BERT | 0.6963 |

O **PySentimiento** se destacou na correlação de Spearman (= **0.7055**), sendo o modelo que melhor preservou a **ordem relativa** das avaliações — ou seja, conseguiu distinguir de forma mais consistente comentários mais positivos dos mais negativos, mesmo sem acertar o valor exato da nota.
