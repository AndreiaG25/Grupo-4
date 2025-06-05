import pandas as pd
from transformers import pipeline

# Carregar o modelo BERT pré-treinado para análise de sentimento
# Usaremos um modelo multilingual que pode lidar com português.
# 'lxyuan/distilbert-base-multilingual-cased-sentiment-pt' é um bom candidato se disponível.
# Se não estiver disponível ou for muito pesado, o 'nlptown/bert-base-multilingual-uncased-sentiment'
# mencionado no relatório também funciona, mas com a ressalva dos valores fixos.
# Vamos tentar um que possa dar resultados mais contínuos.
# Caso contrário, fallback para nlptown, adaptando a lógica.

try:
    # O modelo 'finiteautomata/bertweet-base-sentiment-analysis' que estava a usar tem limite de 128 tokens.
    # Vamos mudar para 'cardiffnlp/twitter-xlm-roberta-base-sentiment' que é multilingual e mais flexível,
    # ou o 'nlptown/bert-base-multilingual-uncased-sentiment' que também funciona.
    # O importante é adicionar truncation=True para lidar com textos longos.
    
    # Modelos alternativos que podem ser melhores para português:
    # 'cardiffnlp/twitter-xlm-roberta-base-sentiment' (Multilingue, treinado em tweets, geralmente bom para textos curtos/médios)
    # 'nlptown/bert-base-multilingual-uncased-sentiment' (Multilingue, 5 estrelas, funciona bem, mas precisa de mapeamento)
    
    # Vamos usar 'cardiffnlp/twitter-xlm-roberta-base-sentiment' como primeira opção, pois dá "POSITIVE", "NEGATIVE", "NEUTRAL"
    # e é mais genérico que o BERTweet para português.
    classifier = pipeline("sentiment-analysis", model="cardiffnlp/twitter-xlm-roberta-base-sentiment")

except Exception as e:
    print(f"Erro ao carregar o modelo BERT: {e}")
    print("Tentando um modelo alternativo ou verificando a conexão/instalação.")
    # Fallback para o modelo nlptown se o anterior falhar
    classifier = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
    # Se usar nlptown, você precisará adaptar a polaridade:
    # label_map = {'1 star': -1.0, '2 stars': -0.5, '3 stars': 0.0, '4 stars': 0.5, '5 stars': 1.0}


def analyze_sentiment(text_path):
    try:
        with open(text_path, 'r', encoding='utf-8') as f:
            full_text = f.read()
    except FileNotFoundError:
        print(f"Erro: O arquivo '{text_path}' não foi encontrado.")
        return pd.DataFrame()

    # Dividir o texto em parágrafos.
    # O arquivo one_sent_per_line.py já formata bem com quebras de linha para cada pergunta/resposta.
    paragraphs = [p.strip() for p in full_text.split('\n') if p.strip() and not p.startswith('Pergunta ')]
    
    # Se ainda tiver problemas com comprimento, pode ser necessário dividir os textos muito longos
    # (ex: Entrevistada:) em sub-sentenças antes de enviar ao classificador.
    # No entanto, truncation=True deve mitigar a maioria dos erros de índice.

    results = []
    for i, para in enumerate(paragraphs):
        if not para: # Ignora parágrafos vazios que podem surgir da divisão
            continue

        # Realiza a inferência de sentimento
        # Adicionar truncation=True para lidar com sequências mais longas que o max_sequence_length do modelo
        sentiment_result = classifier(para, truncation=True)[0]
        
        label = sentiment_result['label']
        score = sentiment_result['score']
        
        polaridade = 0.0
        classificacao = "NEUTRAL"

        # Lógica para modelos como 'cardiffnlp/twitter-xlm-roberta-base-sentiment'
        if label == "LABEL_2" or label.upper() == "POSITIVE": # Para modelos que retornam LABEL_0, LABEL_1, LABEL_2 ou Positive/Negative/Neutral
            polaridade = score # Ou 1.0 se quiser apenas o sinal
            classificacao = "POSITIVE"
        elif label == "LABEL_0" or label.upper() == "NEGATIVE":
            polaridade = -score # Ou -1.0
            classificacao = "NEGATIVE"
        else: # LABEL_1 ou NEUTRAL
            polaridade = 0.0
            classificacao = "NEUTRAL"

        # Lógica específica se o fallback para 'nlptown/bert-base-multilingual-uncased-sentiment' for ativado
        if classifier.model.name_or_path == "nlptown/bert-base-multilingual-uncased-sentiment":
            label_map = {'1 star': -1.0, '2 stars': -0.5, '3 stars': 0.0, '4 stars': 0.5, '5 stars': 1.0}
            polaridade = label_map.get(label, 0.0)
            if polaridade > 0: classificacao = "POSITIVE"
            elif polaridade < 0: classificacao = "NEGATIVE"
            else: classificacao = "NEUTRAL"
            # O score do nlptown já é o "star", o 'score' no resultado é a confiança nesse star.
            # Podemos manter a polaridade como o valor mapeado e ignorar o 'score' de confiança, ou vice-versa.
            # Aqui, mantivemos a polaridade mapeada do star.


        results.append([i + 1, polaridade, classificacao, para]) # Incluir o parágrafo para referência
    
    df_sentiment = pd.DataFrame(results, columns=['Parágrafo', 'polaridade', 'classificação', 'Conteúdo do Parágrafo'])
    return df_sentiment

if __name__ == "__main__":
    normalized_file = "memoriadeinfancia_osl.txt" # Certifique-se de que este arquivo existe
    df_sentiment = analyze_sentiment(normalized_file)
    
    print("\n--- Análise de Sentimento (BERT) ---")
    print(df_sentiment.to_string())

    # --- NOVO: Exportar para Excel ---
    output_excel_file = "analise_sentimento_danielacunha.xlsx"
    try:
        df_sentiment.to_excel(output_excel_file, index=False)
        print(f"\nResultados da análise de sentimento exportados para '{output_excel_file}' com sucesso!")
    except Exception as e:
        print(f"\nErro ao exportar análise de sentimento para Excel: {e}")