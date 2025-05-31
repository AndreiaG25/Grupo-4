entrevistaosl.txt: entrevista.txt
	onesentpl -x entrevista.txt > entrevistaosl.txt

# Frequência das palavras e rácio
entrevista-freq.txt: entrevistaosl.txt
	txtfreq -r entrevistaosl.txt > entrevista-freq.txt

# Lemas e lemas sem pontuação
entrevistacon.txt: entrevistaosl.txt
	avdconll entrevistaosl.txt > entrevistacon.txt
	rg -v PUNCT entrevistacon.txt > entrevistaspunct.txt

# ANÁLISE DE SENTIMENTO:

# Análise sentimento
entrevista-sentimento.txt: entrevistaosl.txt
	sentimento entrevistaosl.txt > entrevista-sentimento.txt

# Análise sentimento detalhada
entrevista-sentimento-detail.txt: entrevistaosl.txt
	sentimento -o entrevistaosl.txt > entrevista-sentimento-detail.txt

# ANÁLISE DETALHADA COM RÁCIO DE PALAVRAS:

# Análise detalhada 
entrevista-ratio.txt: entrevistaosl.txt
	sentimento_detalhado entrevistaosl.txt > entrevista-ratio.txt

# Análise detalhada em formato tabela (TSV) 
entrevista-summary.txt: entrevistaosl.txt
	sentimento_detalhado -s entrevistaosl.txt > entrevista-summary.txt