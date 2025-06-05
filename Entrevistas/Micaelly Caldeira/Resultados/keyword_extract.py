import spacy
import pandas as pd # Importar pandas para exportar para Excel
from collections import Counter

# Carrega o modelo de linguagem português
try:
    nlp = spacy.load("pt_core_news_lg")
except OSError:
    print("O modelo 'pt_core_news_lg' não foi encontrado. A executar 'python -m spacy download pt_core_news_lg'")
    spacy.cli.download("pt_core_news_lg")
    nlp = spacy.load("pt_core_news_lg")

def extract_keywords(text_path, num_keywords=None): # Alterado para aceitar num_keywords=None para todas
    try:
        with open(text_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except FileNotFoundError:
        print(f"Erro: O arquivo '{text_path}' não foi encontrado.")
        return []

    doc = nlp(text)
    
    # Filtrar tokens para substantivos (NOUN), adjetivos (ADJ) e nomes próprios (PROPN)
    # Excluir stopwords (palavras muito comuns sem significado semântico forte) e pontuação
    keywords = [
        token.lemma_.lower() for token in doc 
        if token.pos_ in ["NOUN", "ADJ", "PROPN"] and not token.is_stop and not token.is_punct
    ]
    
    # Contar a frequência das palavras-chave
    keyword_counts = Counter(keywords)
    
    # Retornar as palavras-chave mais comuns (todas se num_keywords for None)
    if num_keywords:
        return [item[0] for item in keyword_counts.most_common(num_keywords)]
    else:
        return [item[0] for item in keyword_counts.most_common()] # Retorna todas se num_keywords não for especificado

if __name__ == "__main__":
    normalized_file = "memoriadeinfancia_osl.txt"
    
    # Podemos extrair todas as palavras-chave para o Excel
    all_keywords = extract_keywords(normalized_file) 
    
    # E para impressão no console, podemos mostrar apenas as top 10, por exemplo
    top_keywords_for_display = extract_keywords(normalized_file, num_keywords=10)
    
    print("\n--- Palavras-chave extraídas (Top 10 para visualização) ---")
    for kw in top_keywords_for_display:
        print(f"- {kw}")

    # --- NOVO: Exportar para Excel ---
    output_excel_file = "palavras_chave_danielacunha.xlsx"
    try:
        # Criar um DataFrame a partir da lista de palavras-chave e suas contagens
        df_keywords = pd.DataFrame(Counter(all_keywords).most_common(), columns=['Palavra-chave', 'Frequência'])
        df_keywords.to_excel(output_excel_file, index=False)
        print(f"\nResultados das palavras-chave exportados para '{output_excel_file}' com sucesso!")
    except Exception as e:
        print(f"\nErro ao exportar palavras-chave para Excel: {e}")