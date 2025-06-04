import spacy
import pandas as pd
import sys
from collections import Counter

# Carrega o modelo spaCy
nlp = spacy.load("pt_core_news_lg")

# Verifica se o nome do arquivo foi passado
if len(sys.argv) < 2:
    print("Uso: python freq.py nome_do_arquivo.txt")
    sys.exit(1)

nome_arquivo = sys.argv[1]

# Lê o texto
with open(nome_arquivo, "r", encoding="utf-8") as f:
    texto = f.read()

# Processa o texto
doc = nlp(texto)

# Extrai palavras válidas
palavras = [
    token.text.lower()
    for token in doc
    if token.is_alpha and not token.is_stop
]

# Lista de palavras a excluir da análise
palavras_excluir = {"entrevistada", "entrevistadora"}

# Extrair palavras relevantes, removendo as de 'palavras_excluir'
palavras = [
    token.text.lower()
    for token in doc
    if token.is_alpha and not token.is_stop and token.text.lower() not in palavras_excluir
]

# Conta palavras
contagem = Counter(palavras)
total_tokens = sum(contagem.values())

# 📥 Corpus de referência
corpus_ref = pd.read_csv("freq_portugues.csv")  # <-- substitua por seu CSV de referência
corpus_dict = dict(zip(corpus_ref["palavra"], corpus_ref["freq_por_milhao"]))

# Preparar os dados
dados = []
lexico = nlp.vocab

for palavra, freq in contagem.items():
    freq_por_milhao = (freq / total_tokens) * 1_000_000

    # Obtemos do corpus de referência
    usual = corpus_dict.get(palavra, 0.01)  # frequência esperada
    ratio = freq_por_milhao / usual

    # Tentamos pegar o ranking spaCy
    lexema = lexico[palavra]
    ranking = lexema.rank if lexema.has_vector else None

    dados.append({
        "palavra": palavra,
        "ocorrencias": freq,
        "freq_per_mill": round(freq_por_milhao, 2),
        "usual": round(usual, 2),
        "ranking": ranking,
        "ratio": round(ratio, 2)
    })

# Cria o DataFrame
df = pd.DataFrame(dados)
df = df.sort_values(by="ocorrencias", ascending=False)

# Exporta os resultados
df.to_csv("freq_resultado.csv", index=False, encoding="utf-8")
df.head(20).to_csv("freq_top20.csv", index=False, encoding="utf-8")
print("Análise concluída.")

import matplotlib.pyplot as plt

# Tabela 2 - ordenar por rácio decrescente
df_ratio = df.sort_values(by="ratio", ascending=False)
df_ratio.to_csv("freq_ratio_decrescente.csv", index=False, encoding="utf-8")

# Mostrar as top 20 palavras por rácio
print("\nTop 20 palavras por rácio:")
print(df_ratio.head(20))

# Gráfico das 20 palavras com maior rácio
top20_ratio = df_ratio.head(20)
plt.figure(figsize=(12,6))
plt.barh(top20_ratio["palavra"], top20_ratio["ratio"], color='skyblue')
plt.xlabel("Rácio (frequência entrevista / frequência usual)")
plt.title("Top 20 palavras mais incomuns na entrevista")
plt.gca().invert_yaxis()  # inverte para maior em cima
plt.tight_layout()
plt.savefig("grafico_ratio_top20.png")
plt.show()


