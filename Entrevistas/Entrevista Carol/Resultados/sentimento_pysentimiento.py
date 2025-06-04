import sys
import pandas as pd
from pysentimiento import create_analyzer
import matplotlib.pyplot as plt

# Verifica se o nome do arquivo foi passado
if len(sys.argv) < 2:
    print("Uso: python3 sentimento_pysentimiento.py nome_arquivo.txt")
    sys.exit(1)

nome_arquivo = sys.argv[1]

# Lê o conteúdo do arquivo
with open(nome_arquivo, "r", encoding="utf-8") as f:
    texto = f.read()

# Divide o texto por parágrafos
paragrafos = [p.strip() for p in texto.split("\n") if p.strip()]

# Cria o analisador de sentimento em português
analisador = create_analyzer(task="sentiment", lang="pt")

# Aplica a análise a cada parágrafo
resultados = []

for i, paragrafo in enumerate(paragrafos, 1):
    resultado = analisador.predict(paragrafo)
    resultados.append({
        "Parágrafo": i,
        "Texto": paragrafo,
        "Sentimento": resultado.output,
        "Positivo": resultado.probas.get("POS", 0),
        "Neutro": resultado.probas.get("NEU", 0),
        "Negativo": resultado.probas.get("NEG", 0),
    })

# Salva os resultados em um CSV
df = pd.DataFrame(resultados)
df.to_csv("analise_sentimentos.csv", index=False, encoding="utf-8")
print("CSV de análise salvo como 'analise_sentimentos.csv'.")

# Agrupa por sentimento para total geral
totais = df["Sentimento"].value_counts().reindex(["POS", "NEU", "NEG"], fill_value=0)

# Gráfico 1: Total de cada sentimento
plt.figure(figsize=(6, 4))
plt.bar(totais.index, totals := totais.values, color="#FFA500", edgecolor="black")
plt.title("Total de Sentimentos na Entrevista")
plt.xlabel("Sentimento")
plt.ylabel("Número de Parágrafos")
plt.tight_layout()
plt.savefig("sentimentos_totais.png", dpi=300)
print("Gráfico de totais salvo como 'sentimentos_totais.png'.")

# Gráfico 2: Sentimento por parágrafo
plt.figure(figsize=(12, 6))
cores = {"POS": "#32CD32", "NEU": "#FFD700", "NEG": "#FF4500"}
df["Cor"] = df["Sentimento"].map(cores)

plt.bar(df["Parágrafo"], df["Positivo"], color="#32CD32", label="Positivo", bottom=df["Neutro"] + df["Negativo"])
plt.bar(df["Parágrafo"], df["Neutro"], color="#FFD700", label="Neutro", bottom=df["Negativo"])
plt.bar(df["Parágrafo"], df["Negativo"], color="#FF4500", label="Negativo")
plt.xlabel("Parágrafo")
plt.ylabel("Probabilidade")
plt.title("Distribuição de Sentimentos por Parágrafo")
plt.legend()
plt.tight_layout()
plt.savefig("sentimentos_por_paragrafo.png", dpi=300)
print("Gráfico por parágrafo salvo como 'sentimentos_por_paragrafo.png'.")

print("\n✅ Análise de sentimentos concluída com sucesso!")
print("→ Arquivo CSV: analise_sentimentos.csv")
print("→ Gráficos: sentimentos_totais.png e sentimentos_por_paragrafo.png")

