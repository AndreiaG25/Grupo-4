import re
from collections import Counter

# Lista expandida de stop words comuns em português
stop_words = {
    "a", "à", "agora", "ainda", "algo", "algum", "alguma", "algumas", "alguns", "alguém",
    "ao", "aos", "apenas", "após", "aqui", "as", "assim", "até", "bem", "bom", "cada",
    "casa", "com", "como", "contra", "coisa", "coisas", "comigo", "conta", "da", "das",
    "de", "dela", "dele", "deles", "delas", "depois", "desde", "dia", "dias", "diz",
    "dizer", "do", "dos", "e", "ela", "elas", "ele", "eles", "em", "enquanto", "então",
    "entre", "era", "essa", "essas", "esse", "esses", "esta", "está", "estamos", "estão",
    "estas", "este", "estes", "eu", "faz", "fazer", "fez", "foi", "for", "foram", "forma",
    "há", "isso", "isto", "já", "lá", "lhe", "lhes", "mais", "mas", "me", "mesma",
    "mesmo", "meu", "meus", "minha", "minhas", "muito", "na", "não", "nas", "nem", "no",
    "nos", "nós", "nossa", "nossas", "nosso", "nossos", "num", "numa", "nunca", "o", "os",
    "onde", "ou", "para", "pela", "pelas", "pelo", "pelos", "perto", "pode", "podem",
    "por", "porque", "porquê", "posso", "pouco", "povo", "pra", "qual", "quando", "quanto",
    "que", "quem", "sabe", "se", "sem", "sempre", "ser", "seu", "seus", "só", "sou",
    "sua", "suas", "também", "tão", "tem", "tendo", "ter", "te", "teu", "teus", "tinha",
    "tive", "toda", "todas", "todo", "todos", "trabalho", "tu", "tua", "tuas", "um",
    "uma", "umas", "uns", "vai", "vão", "você", "vocês", "é", "era", "estava", "estar",
    "estive", "estou", "vai", "vamos", "vou", "somos", "fui", "sobre", "aí", "ali", "tudo"
}

# Palavras específicas da transcrição que devem ser removidas
palavras_excluir = {"entrevistadora", "entrevistado", "entrevistada"}

def ler_texto(caminho_ficheiro):
    with open(caminho_ficheiro, 'r', encoding='utf-8') as f:
        texto = f.read()
    return texto

def limpar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r'[^a-záàâãéèêíïóôõöúçñ0-9\s]', '', texto)
    return texto

def contar_palavras(texto_limpo):
    palavras = texto_limpo.split()
    palavras_filtradas = [p for p in palavras if p not in stop_words and p not in palavras_excluir]
    frequencia = Counter(palavras_filtradas)
    return frequencia

def salvar_frequencia(frequencia, caminho_saida):
    with open(caminho_saida, 'w', encoding='utf-8') as f:
        for palavra, freq in frequencia.most_common():
            f.write(f"{palavra}\t{freq}\n")

if __name__ == "__main__":
    caminho_entrada = "entrevista_vitoria.txt"
    caminho_saida = "frequencia_filtrada.txt"

    texto = ler_texto(caminho_entrada)
    texto_limpo = limpar_texto(texto)
    frequencia = contar_palavras(texto_limpo)
    salvar_frequencia(frequencia, caminho_saida)

    print(f"Análise de frequência filtrada concluída! Resultado salvo em {caminho_saida}")

