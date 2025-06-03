import pandas as pd

# Carregar o CSV
df = pd.read_csv('celeba_train.csv')
total_registros = len(df)

# Contagens
gender_counts = df['gender'].value_counts().to_dict()
race_counts = df['race'].value_counts().to_dict()
combo_counts = df.groupby(['gender', 'race']).size().reset_index(name='count')

# Função para criar uma tabela de estatísticas
def gerar_linhas(tipo, dados_dict):
    linhas = []
    for categoria, count in dados_dict.items():
        fora = total_registros - count
        perc = (count / total_registros) * 100
        linhas.append([tipo, categoria, count, fora, round(perc, 2)])
    # Linha total
    linhas.append([tipo, "Total", total_registros, 0, 100.0])
    return linhas

# Gênero
linhas_genero = gerar_linhas("gênero", gender_counts)

# Raça
linhas_raca = gerar_linhas("raça", race_counts)

# Combinação
linhas_combo = []
for _, row in combo_counts.iterrows():
    categoria = f'{row["gender"]},{row["race"]}'
    count = row["count"]
    fora = total_registros - count
    perc = (count / total_registros) * 100
    linhas_combo.append(["combinação", categoria, count, fora, round(perc, 2)])
linhas_combo.append(["combinação", "Total", 0, total_registros, 0, 100.0])

# Juntar tudo com linhas em branco entre blocos
linhas_final = (
    [["Tipo", "Categoria", "Quantidade", "Fora do Grupo", "Porcentagem (%)"]] +
    linhas_genero +
    [["", "", "", "", ""]] +  # linha em branco
    [["Tipo", "Categoria", "Quantidade", "Fora do Grupo", "Porcentagem (%)"]] +
    linhas_raca +
    [["", "", "", "", ""]] +  # linha em branco
    [["Tipo", "Gender", "Race", "Quantidade", "Fora do Grupo", "Porcentagem (%)"]] +
    linhas_combo
)

# Escrever manualmente para manter as quebras
with open("train_dados.csv", "w", encoding="utf-8") as f:
    for linha in linhas_final:
        f.write(",".join(str(c) for c in linha) + "\n")

print("Arquivo gerado: train_dados.csv")
