from collections import Counter
import re

# Passo 1: carregar o texto da entrevista
with open('entrevista_vitoria.txt', 'r', encoding='utf-8') as f:
    texto = f.read().lower()

# Passo 2: tokenizar as palavras usando regex (só letras, para evitar pontuação)
palavras = re.findall(r'\b[a-zá-úãõç]+\b', texto)

# Passo 3: contar frequência das palavras
freq_palavras = Counter(palavras)

# Passo 4: ordenar por frequência decrescente
palavras_ordenadas = freq_palavras.most_common()

# Passo 5: mostrar as 20 palavras mais comuns
print("Palavra | Frequência")
print("--------------------")
for palavra, freq in palavras_ordenadas[:20]:
    print(f"{palavra} | {freq}")
