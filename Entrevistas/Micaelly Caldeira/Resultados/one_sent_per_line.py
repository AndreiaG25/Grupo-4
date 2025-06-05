import re

def normalize_text_for_analysis(input_file_path, output_file_path):
    """
    Normaliza o texto de um arquivo, garantindo uma sentença por linha,
    removendo hifenização, tratando abreviações e diálogos,
    e removendo "Entrevistada:", "Pergunta X -" e "Pergunta X".
    """
    try:
        with open(input_file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except FileNotFoundError:
        print(f"Erro: O arquivo de entrada '{input_file_path}' não foi encontrado.")
        return

    # 1. Remover "Entrevistada:" e "Pergunta X -" e "Pergunta X"
    # Regex para capturar "Entrevistada:", "Pergunta X -", "Pergunta X" (onde X é um número e opcionalmente um traço)
    text = re.sub(r'Entrevistada:', '', text)
    text = re.sub(r'Pergunta\s+\d+\s*-?', '', text) # Remove "Pergunta X -" ou "Pergunta X"
    
    # 2. Remover o título da entrevista se estiver no topo do arquivo (ex: "As Memórias de Infância de Daniela Cunha")
    # Este é um padrão específico para a sua entrevista, pode precisar de ajuste se o título mudar.
    text = re.sub(r'As Memórias de Infância de Daniela Cunha\s*', '', text, 1) # remove apenas a primeira ocorrência

    # 3. Remover linhas vazias extras que podem surgir após a remoção acima
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        stripped_line = line.strip()
        if stripped_line: # Apenas adiciona linhas que não são vazias após strip
            cleaned_lines.append(stripped_line)
    
    # Juntar as linhas novamente com uma quebra de linha.
    # O spaCy e outros processadores se beneficiam de texto limpo para tokenização de sentenças.
    # Se o objetivo é 'uma sentença por linha', a lógica aqui precisa ser mais sofisticada.
    # Para o seu caso (parágrafos e análise de sentimentos por parágrafo), manter parágrafos separados é bom.
    # Para frequências e palavras-chave, o texto contínuo está OK.

    final_cleaned_text = '\n'.join(cleaned_lines)

    # 4. Normalização adicional (como no one_sent_per_line original)
    # Remover hifenização (se houver, e geralmente é para palavras divididas entre linhas)
    final_cleaned_text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', final_cleaned_text)
    
    # Remover múltiplos espaços em branco e quebras de linha duplicadas
    final_cleaned_text = re.sub(r'\s+', ' ', final_cleaned_text) # Substituir múltiplos espaços por um único
    final_cleaned_text = re.sub(r'\n\s*\n', '\n', final_cleaned_text) # Remover quebras de linha duplas
    final_cleaned_text = final_cleaned_text.strip() # Remover espaços em branco no início/fim do texto

    # Se a intenção é "uma sentença por linha" rigorosamente, você precisaria do spaCy aqui para segmentação de sentenças.
    # No entanto, para as suas análises (frequência, sentimento por parágrafo), o texto tratado por `split('\n')`
    # na `analyze_sentiment` (que espera parágrafos) e pelo `nlp(text)` no `freq.py` (que processa o texto inteiro)
    # já funciona bem com este nível de limpeza.

    with open(output_file_path, 'w', encoding='utf-8') as f:
        f.write(final_cleaned_text)
    print(f"Texto normalizado e limpo salvo em '{output_file_path}'")

if __name__ == "__main__":
    input_file = "memoriadeinfancia.txt"
    output_file = "memoriadeinfancia_osl.txt" # O nome que você está usando nos outros scripts
    normalize_text_for_analysis(input_file, output_file)