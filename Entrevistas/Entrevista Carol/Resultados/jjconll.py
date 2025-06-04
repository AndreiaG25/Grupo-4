import spacy
import sys
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# Carrega o modelo spaCy em português
nlp = spacy.load("pt_core_news_lg")

# Verifica se o nome do arquivo foi passado
if len(sys.argv) < 2:
    print("Uso: python3 jjconll.py nome_arquivo.txt")
    sys.exit(1)

nome_arquivo = sys.argv[1]

# Lê o texto
with open(nome_arquivo, "r", encoding="utf-8") as f:
    texto = f.read()

# Processa o texto
doc = nlp(texto)

dados = []

for token in doc:
    if not token.is_space:
        dados.append({
            "palavra": token.text,
            "lema": token.lemma_,
            "pos": token.pos_,
            "entidade": token.ent_type_ if token.ent_type_ else "-",
            "dependencia": token.dep_,
            "ranking": token.rank if token.has_vector else "-"
        })

# Cria DataFrame e exporta análise completa
df = pd.DataFrame(dados)
df.to_csv("analise_morfossintatica.csv", index=False, encoding="utf-8")
print("Análise morfossintática concluída. Resultado salvo em 'analise_morfossintatica.csv'.")

# Contar as classes gramaticais (POS)
contagem_pos = Counter([token.pos_ for token in doc])

# Criar DataFrame com as contagens
df_pos = pd.DataFrame(contagem_pos.items(), columns=["POS", "Contagem de POS"])
df_pos = df_pos.sort_values(by="Contagem de POS", ascending=False)

# Calcular total geral
total_geral = df_pos["Contagem de POS"].sum()

# Adicionar linha de total geral no final
linha_total = pd.DataFrame([["Total Geral", total_geral]], columns=["POS", "Contagem de POS"])
df_pos = pd.concat([df_pos, linha_total], ignore_index=True)

# Exportar tabela para CSV
df_pos.to_csv("tabela_pos.csv", index=False, encoding="utf-8")

# Criar DataFrame para imagem sem a linha "Total Geral"
df_pos_sem_total = df_pos[df_pos["POS"] != "Total Geral"]

# Gerar tabela em imagem PNG com cor laranja
fig, ax = plt.subplots(figsize=(6, len(df_pos_sem_total)*0.5 + 1))
ax.axis('off')
tabela = ax.table(cellText=df_pos_sem_total.values, colLabels=df_pos_sem_total.columns, cellLoc='center', loc='center')

# Estilo da tabela
tabela.auto_set_font_size(False)
tabela.set_fontsize(10)
tabela.scale(1.2, 1.2)
colors = ['#87CEFA'] * len(df_pos_sem_total)  # azul claro (LightSkyBlue)

# Aplica cor na coluna POS
for i in range(1, len(df_pos_sem_total)+1):
    tabela[i, 0].set_facecolor(colors[i-1])

plt.savefig("tabela_pos.png", bbox_inches='tight', dpi=300)

# Gerar gráfico com cor laranja
plt.figure(figsize=(10, 6))
plt.bar(df_pos_sem_total["POS"], df_pos_sem_total["Contagem de POS"], color="#87CEFA", edgecolor="black")
plt.title("Total de Ocorrências por Classe Gramatical (POS)", fontsize=14)
plt.xlabel("Classe Gramatical (POS)")
plt.ylabel("Frequência")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("grafico_pos.png", dpi=300)
plt.show()
