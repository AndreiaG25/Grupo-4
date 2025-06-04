# Normalização do texto
entrevistaosl.txt: entrevista_vitoria.txt
	onesentpl -x entrevista_vitoria.txt > entrevistaosl.txt

# Frequência das palavras e rácio
entrevista-freq.txt: entrevistaosl.txt
	txtfreq -r entrevistaosl.txt > entrevista-freq.txt

# Lemas e lemas sem pontuação
entrevistacon.txt: entrevistaosl.txt
	avdconll entrevistaosl.txt > entrevistacon.txt
	rg -v PUNCT entrevistacon.txt > entrevistaspunct.txt

# Extração de palavras-chave
palavra-chave.txt: entrevistaosl.txt
	python keyword_extract.py -n 5 entrevistaosl.txt > palavra-chave.txt

# Análise de sentimento
entrevista-sentimento.txt: entrevistaosl.txt
	sentimento entrevistaosl.txt > entrevista-sentimento.txt

# Análise de sentimento detalhada
entrevista-sentimento-detail.txt: entrevistaosl.txt
	sentimento -o entrevistaosl.txt > entrevista-sentimento-detail.txt

# Análise detalhada com rácio de palavras
entrevista-ratio.txt: entrevistaosl.txt
	sentimento_detalhado entrevistaosl.txt > entrevista-ratio.txt

# Análise detalhada em formato de tabela (TSV)
entrevista-summary.txt: entrevistaosl.txt
	sentimento_detalhado -s entrevistaosl.txt > entrevista-summary.txt
