# Explorando o Viés em Deep Learning: Classificação de Gênero e Raça com Datasets Balanceados e Enviesados

### 🎯 Objetivo do Experimento
Comparar o impacto do viés nos datasets FairFace (balanceado) e CelebA (enviesado), treinando um modelo de IA (ConvNeXt-Tiny) para prever raça e/ou gênero, e avaliando justiça e desempenho por grupo.

### 🔎 Preparação dos Dados
- Padronizou ambos os datasets (FairFace e CelebA).
- Criou atributo raça no CelebA (que não tinha originalmente).
- Dividiu os dois datasets igualmente:
- 85 mil imagens para treinamento
- 10 mil imagens para validação
- Salvou os rótulos em arquivos CSV, sem usar subpastas por classe.

Foi realizada a padronização dos datasets FairFace e CelebA por meio de técnicas de dataset downsampling, schema alignment e data preprocessing, incluindo limpeza de atributos, seleção de instâncias e reformatação de caminhos de arquivos. Para lidar com a ausência do atributo racial no CelebA, foi utilizado pseudo-labeling via modelo pré-treinado (FairFace), com posterior unificação de estrutura de metadados em arquivos CSV padronizados.

### 🧠 Modelo Escolhido

Vai usar o ConvNeXt-Tiny, que é moderno, poderoso e sensível a padrões sutis — ideal para detectar viés.
Adaptará a última camada para prever raça ou gênero (multi tarefa).

### ✅ Plano de Treinamento e Avaliação
A) Treinar um modelos com dois datasets:
- CelebA (Gênero e Raça desequilibrados)
- FairFace (Gênero e Raça equilibrados)

B) Avaliar cada modelo:
- No seu próprio conjunto de validação (padrão)
- No conjunto de validação do outro dataset (teste cruzado)
- Comparar desempenho por grupo, como: Mulheres negras, homens brancos, etc.

### 📊 Métricas e Análises
- Acurácia, Precision, Recall, F1 por grupo (raça × gênero)

### 🛠️ Ferramentas úteis
- PyTorch para treino
- Pandas

