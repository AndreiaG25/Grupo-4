import sys
import pandas as pd
import matplotlib.pyplot as plt
from pysentimiento import create_analyzer
import seaborn as sns

def salvar_tabela_como_imagem(df, imagem_saida_base, titulo_base, max_linhas=20):
    total_linhas = len(df)
    num_partes = (total_linhas // max_linhas) + int(total_linhas % max_linhas > 0)

    for parte in range(num_partes):
        inicio = parte * max_linhas
        fim = inicio + max_linhas
        df_parte = df.iloc[inicio:fim]

        fig, ax = plt.subplots(figsize=(12, len(df_parte) * 0.5 + 1))
        ax.axis('off')
        ax.axis('tight')

        tabela = ax.table(cellText=df_parte.values,
                          colLabels=df_parte.columns,
                          cellLoc='center',
                          loc='center')

        tabela.auto_set_font_size(False)
        tabela.set_fontsize(10)
        tabela.auto_set_column_width(col=list(range(len(df.columns))))

        titulo = f"{titulo_base} (Parte {parte + 1})"
        plt.title(titulo, fontsize=14, weight='bold')
        plt.tight_layout()

        nome_arquivo = f"{imagem_saida_base.replace('.jpg', '')}_parte{parte + 1}.jpg"
        plt.savefig(nome_arquivo, dpi=300)
        plt.close()

# Verifica argumento
if len(sys.argv) < 2:
    print("Uso: python3 sentimento_detalhado.py nome_arquivo.txt")
    sys.exit(1)

nome_arquivo = sys.argv[1]

# Lê texto e divide por parágrafos
with open(nome_arquivo, "r", encoding="utf-8") as f:
    texto = f.read()

paragrafos = [p.strip() for p in texto.split("\n") if p.strip()]

analisador = create_analyzer(task="sentiment", lang="pt")

# Análise por parágrafo
resultados_paragrafos = []
resultados_palavras = []

for i, p in enumerate(paragrafos, 1):
    res = analisador.predict(p)
    pos = res.probas["POS"]
    neu = res.probas["NEU"]
    neg = res.probas["NEG"]

    polaridade = round(pos - neg, 3)

    if polaridade > 0.05:
        classificacao = "POSITIVE"
    elif polaridade < -0.05:
        classificacao = "NEGATIVE"
    else:
        classificacao = "NEUTRAL"

    resultados_paragrafos.append({
        "Parágrafo": i,
        "Polaridade": polaridade,
        "Classificação": classificacao
    })

    palavras = p.split()
    total_palavras = len(palavras)
    n_positivas = round(total_palavras * pos)
    n_negativas = round(total_palavras * neg)
    ratio_pos = round(n_positivas / total_palavras, 3) if total_palavras > 0 else 0
    ratio_neg = round(n_negativas / total_palavras, 3) if total_palavras > 0 else 0

    resultados_palavras.append({
        "Parágrafo": i,
        "Polaridade": polaridade,
        "Classificação": classificacao,
        "Palavras_totais": total_palavras,
        "Nº_positivas": n_positivas,
        "Nº_negativas": n_negativas,
        "Ratio_posi": ratio_pos,
        "Ratio_neg": ratio_neg
    })

# Salva CSVs
df_paragrafos = pd.DataFrame(resultados_paragrafos)
df_paragrafos.to_csv("analise_sentimento_paragrafos.csv", index=False, encoding="utf-8")

df_palavras = pd.DataFrame(resultados_palavras)
df_palavras.to_csv("analise_sentimento_palavras.csv", index=False, encoding="utf-8")

# Salva imagens divididas
salvar_tabela_como_imagem(df_paragrafos, "tabela_sentimento_paragrafos.jpg", "Análise de Sentimento por Parágrafos")
salvar_tabela_como_imagem(df_palavras, "tabela_sentimento_palavras.jpg", "Análise Estimada de Sentimentos por Palavras")

# Gráficos de barras
sns.set(style="whitegrid")

# Gráfico 1: Distribuição de sentimentos por parágrafos
contagem_paragrafos = df_paragrafos["Classificação"].value_counts().reindex(["POSITIVE", "NEUTRAL", "NEGATIVE"], fill_value=0)

plt.figure(figsize=(7, 5))
sns.barplot(x=contagem_paragrafos.index, y=contagem_paragrafos.values, palette=["#66B2FF", "#A0A0A0", "#FF6B6B"])
plt.title("Distribuição de Sentimentos por Parágrafos", fontsize=14)
plt.xlabel("Sentimento")
plt.ylabel("Quantidade de Parágrafos")
plt.tight_layout()
plt.savefig("grafico_sentimentos_paragrafos.png", dpi=300)
plt.close()

# Gráfico 2: Distribuição de sentimentos por palavras
contagem_palavras = df_palavras["Classificação"].value_counts().reindex(["POSITIVE", "NEUTRAL", "NEGATIVE"], fill_value=0)

plt.figure(figsize=(7, 5))
sns.barplot(x=contagem_palavras.index, y=contagem_palavras.values, palette=["#66B2FF", "#A0A0A0", "#FF6B6B"])
plt.title("Distribuição Estimada de Sentimentos por Palavras", fontsize=14)
plt.xlabel("Sentimento Estimado")
plt.ylabel("Quantidade de Parágrafos")
plt.tight_layout()
plt.savefig("grafico_sentimentos_palavras.png", dpi=300)
plt.close()

print("✅ Análise concluída. CSVs, tabelas divididas e gráficos salvos com sucesso.")

