# Prevendo o índice Ibovespa com Prophet no Python

*FIAP: Pós Tech - Data Analytics Tech Challenge #02*

Grupo 32 - Autores:
- Cristiane Aline Fischer
- Pedro Baldini
- Vinícius Prado Lima

**[Link do artigo](COLOCAR LINK MEDIUM)**

---

### Contexto
O [Ibovespa](https://www.b3.com.br/pt_br/market-data-e-indices/indices/indices-amplos/ibovespa.htm) (Índice da Bolsa de Valores de São Paulo), fundado em 1968, reflete o desempenho médio das cotações dos ativos mais negociados e representativos do mercado de ações brasileiro. O indicador é composto por uma carteira teórica periodicamente ajustada pela [B3](https://www.b3.com.br), na tentativa de incorporar os ativos de maior relevância no cenário financeiro do país.

**O problema:** criar um modelo de série temporal para prever diariamente o fechamento índice do Ibovespa, com acurácia mínima de 70%.

---

### Arquivos Importantes

**main/**

- `0_dados_EDA.ipynb`: coleta e análise exploratória de dados
- `1_ibovespa_forecasting.ipynb`: construção do modelo de previsão com Prophet
- `utils.py`: funções úteis para a construção do modelo

**output/**

- `ibovespa_forecasting.json`: modelo final para previsão
