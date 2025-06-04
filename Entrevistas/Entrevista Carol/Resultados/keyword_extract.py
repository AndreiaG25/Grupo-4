import spacy
import sys
from collections import Counter
import matplotlib.pyplot as plt

# Carrega o modelo spaCy em português
nlp = spacy.load("pt_core_news_lg")

# Verifica se o nome do arquivo foi passado
if len(sys.argv) < 2:
    print("Uso: python3 keyword_extract.py nome_arquivo.txt")
    sys.exit(1)

nome_arquivo = sys.argv[1]

# Lê o texto
with open(nome_arquivo, "r", encoding="utf-8") as f:
    texto = f.read()

# Processa o texto
doc = nlp(texto)

# Lista de palavras a excluir (ajustada)
palavras_excluir = {
    "entrevistador", "entrevistadora", "entrevistado", "entrevistada",
    "pergunta", "resposta", "diz", "disse", "falar", "falou", "né", "aham",
    "tá", "dia", "gente", "coisa", "pra", "pessoa", "certo", "ficar", "esperar", "mensagem"
}

# Contador de palavras-chave (substantivos, pronomes, verbos, advérbios, adjetivos)
keyword_counts = Counter()

for token in doc:
    if (
        not token.is_stop
        and not token.is_punct
        and token.pos_ in {"NOUN", "VERB", "PRON", "ADV", "ADJ"}
        and token.lemma_.lower() not in palavras_excluir
        and len(token.text) > 2
    ):
        keyword_counts[token.lemma_.lower()] += 1

# Obter top 5
top_keywords = keyword_counts.most_common(5)

# Exibir no terminal
print("Top 5 palavras-chave (mais refinadas):")
for palavra, freq in top_keywords:
    print(f"{palavra}: {freq}")

# -------------------------
# Gráfico em azul
# -------------------------
labels, valores = zip(*top_keywords)

plt.figure(figsize=(8, 5))
plt.bar(labels, valores, color="#87CEFA", edgecolor="black")
plt.title("Top 5 Palavras-chave", fontsize=14, weight='bold')
plt.xlabel("Palavras-chave")
plt.ylabel("Frequência")
plt.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.7)
plt.tight_layout()
plt.savefig("grafico_keywords.png", dpi=300)
plt.show()



