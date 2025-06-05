import spacy
import pandas as pd
from collections import Counter

# Carrega o modelo de linguagem português
try:
    nlp = spacy.load("pt_core_news_lg")
except OSError:
    print("O modelo 'pt_core_news_lg' não foi encontrado. A executar 'python -m spacy download pt_core_news_lg'")
    spacy.cli.download("pt_core_news_lg")
    nlp = spacy.load("pt_core_news_lg")

def analyze_morphosyntax(text_path):
    try:
        with open(text_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except FileNotFoundError:
        print(f"Erro: O arquivo '{text_path}' não foi encontrado.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame() # Retorna DataFrames vazios

    doc = nlp(text)
    
    pos_counts = Counter()
    token_data = []
    entity_data = []

    for token in doc:
        pos_counts[token.pos_] += 1 # Contagem de POS
        
        # Correção aqui: Tenta obter o ID do lema. Se não existir, define ranking como None.
        ranking = None
        if token.lemma_ in nlp.vocab.strings: # Esta é a maneira de verificar se a string existe no StringStore
            ranking = nlp.vocab.strings.as_int(token.lemma_)

        token_data.append({
            'Palavra': token.text,
            'Lema': token.lemma_,
            'POS': token.pos_,
            'Tag Detalhada': token.tag_, # Tag morfológica mais detalhada
            'Dependência': token.dep_,
            'Cabeça da Dependência': token.head.text, # A palavra da qual este token depende
            'Stopword': token.is_stop,
            'Pontuação': token.is_punct,
            'Número': token.like_num,
            'Espaço': token.is_space,
            'Ranking': ranking if ranking is not None else '' # Use o ranking corrigido
        })

    for ent in doc.ents:
        entity_data.append([ent.text, ent.label_])

    df_tokens = pd.DataFrame(token_data)
    df_pos_counts = pd.DataFrame(pos_counts.items(), columns=['POS', 'Contagem']).sort_values(by='Contagem', ascending=False)
    
    # Entidades, removendo duplicatas e contando os tipos
    df_entities = pd.DataFrame(entity_data, columns=['Entidade', 'Tipo']).drop_duplicates()
    
    # Contagem dos tipos de entidades
    df_entity_types_counts = df_entities['Tipo'].value_counts().reset_index()
    df_entity_types_counts.columns = ['Tipo de Entidade', 'Contagem']


    return df_tokens, df_pos_counts, df_entities, df_entity_types_counts # Retorna também a contagem de tipos de entidades

if __name__ == "__main__":
    normalized_file = "memoriadeinfancia_osl.txt" # Certifique-se de que este arquivo existe
    df_tokens, df_pos_counts, df_entities, df_entity_types_counts = analyze_morphosyntax(normalized_file)
    
    # --- Impressão no console (opcional, para visualização rápida) ---
    print("\n--- Contagem de POS (Classes Gramaticais) ---")
    print(df_pos_counts.to_string())
    
    print("\n--- Palavras organizadas por Ranking (Top 30 Menos Comuns - focado em lemas) ---")
    # Filtrar palavras com ranking, remover stopwords e pontuação, e ordenar pelos IDs de lema (maiores IDs = menos comuns)
    df_ranking = df_tokens[
        (df_tokens['Ranking'].apply(lambda x: isinstance(x, (int, float)) and x != '')) & # Garante que Ranking é numérico e não vazio
        (df_tokens['Stopword'] == False) & 
        (df_tokens['Pontuação'] == False)
    ].sort_values(by='Ranking', ascending=False).head(30)
    print(df_ranking[['Palavra', 'Lema', 'POS', 'Ranking']].to_string())

    print("\n--- Entidades Reconhecidas ---")
    print(df_entities.to_string())
    
    print("\n--- Contagem de Tipos de Entidades ---")
    print(df_entity_types_counts.to_string())


    # --- NOVO: Exportar para Excel com múltiplas abas ---
    output_excel_file = "analise_morfossintatica_danielacunha.xlsx"
    
    try:
        with pd.ExcelWriter(output_excel_file, engine='openpyxl') as writer:
            df_pos_counts.to_excel(writer, sheet_name='Contagem_POS', index=False)
            df_tokens.to_excel(writer, sheet_name='Detalhes_Tokens', index=False)
            df_entities.to_excel(writer, sheet_name='Entidades_Reconhecidas', index=False)
            df_entity_types_counts.to_excel(writer, sheet_name='Contagem_Tipos_Entidades', index=False)

        print(f"\nResultados da análise morfossintática exportados para '{output_excel_file}' com sucesso!")
    except Exception as e:
        print(f"\nErro ao exportar para Excel: {e}")