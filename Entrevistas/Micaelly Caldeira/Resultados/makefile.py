# Define o interpretador Python a ser usado.
# Você pode mudar para `python3` se o seu sistema usar `python` para Python 2.x
PYTHON = python

# Nome do arquivo de entrada original
INPUT_TEXT = memoriadeinfancia.txt
# Nome do arquivo normalizado (saída de one_sent_per_line.py)
NORMALIZED_TEXT = memoriadeinfancia_osl.txt

# Nomes dos arquivos de saída Excel
FREQ_EXCEL = frequencia_palavras_danielacunha.xlsx
SENTIMENT_EXCEL = analise_sentimento_danielacunha.xlsx
KEYWORDS_EXCEL = palavras_chave_danielacunha.xlsx

# Define o alvo padrão, que executará todas as análises
.PHONY: all clean normalize frequency_report morphosyntax_report keywords_report sentiment_analysis_report

all: $(FREQ_EXCEL) $(SENTIMENT_EXCEL) $(KEYWORDS_EXCEL) morphosyntax_report

# Regra para normalizar o texto
$(NORMALIZED_TEXT): $(INPUT_TEXT)
	@echo "Normalizando o texto..."
	$(PYTHON) one_sent_per_line.py

# Regra para gerar o relatório de frequência
$(FREQ_EXCEL): $(NORMALIZED_TEXT)
	@echo "Gerando relatório de frequência..."
	$(PYTHON) freq.py

# Regra para gerar o relatório de análise morfossintática (não gera Excel, apenas imprime no console)
morphosyntax_report: $(NORMALIZED_TEXT)
	@echo "Realizando análise morfossintática..."
	$(PYTHON) jjconll.py

# Regra para gerar o relatório de palavras-chave
$(KEYWORDS_EXCEL): $(NORMALIZED_TEXT)
	@echo "Gerando relatório de palavras-chave..."
	$(PYTHON) keyword_extract.py

# Regra para gerar o relatório de análise de sentimentos
$(SENTIMENT_EXCEL): $(NORMALIZED_TEXT)
	@echo "Gerando relatório de análise de sentimentos..."
	$(PYTHON) sentiment_analysis.py

# Regra para limpar todos os arquivos gerados
clean:
	@echo "Limpando arquivos gerados..."
	@rm -f $(NORMALIZED_TEXT) $(FREQ_EXCEL) $(SENTIMENT_EXCEL) $(KEYWORDS_EXCEL)
	@echo "Limpeza concluída."