# 📊 SaaS de Marketing Science & Predição de Mídia

Este projeto é uma plataforma end-to-end de Marketing Science construída para analisar, prever e otimizar campanhas de tráfego pago utilizando algoritmos de Machine Learning e inferência causal.

O objetivo da ferramenta é substituir análises manuais em planilhas por um ecossistema automatizado que responde a perguntas complexas de negócios (como o impacto real da fadiga de anúncios e a previsão de ROAS).

## 🚀 Arquitetura e Funcionalidades

A aplicação foi dividida em 4 módulos principais:

* **1. Visão Geral (Business Intelligence):** * Cálculo automático das principais métricas de mídia (CTR, CPC, CPM, CPA, ROAS) consumidas de um banco de dados relacional.
    * Visualização interativa da distribuição de performance por *Placement* e segmentação.
* **2. Simulador Preditivo (Machine Learning):** * Utiliza um modelo de Regressão treinado com automação (`LazyPredict` testando +30 algoritmos) para prever o ROAS e a Receita Estimada de campanhas futuras.
    * O usuário insere parâmetros como Orçamento, Frequência e Público, e a IA calcula o risco financeiro antes da campanha ir ao ar.
* **3. Top Creatives (Automação de Insights):** * Módulo lógico que varre a base de dados em tempo real buscando anúncios que superam a média global de alta performance.
    * Gera recomendações automáticas de escala (Escalar, Testar, Pausar) baseadas em padrões matemáticos encontrados nos "campeões".
* **4. Marketing Science Lab (Estatística Avançada):**
    * **Causalidade (Uplift):** Medição da diferença de retorno ao transferir o controle para a IA da plataforma de anúncios (Advantage+).
    * **Análise de Fadiga (OLS):** Gráfico de dispersão com linha de tendência de mínimos quadrados (OLS) cruzando Frequência vs. CPA para identificar o ponto exato de saturação do público.
    * **Projeção (Time Series):** Previsão direcional de ROAS para os próximos 30 dias.

## 🛠️ Stack Tecnológico

* **Linguagem:** Python 3
* **Engenharia de Dados:** Pandas, SQLite3 (Banco de Dados embutido)
* **Machine Learning / Estatística:** Scikit-Learn, LazyPredict, Statsmodels
* **Frontend / Deploy:** Streamlit, Plotly (Gráficos Interativos)

## 💻 Como rodar este projeto localmente

```bash
# Clone o repositório
git clone [https://github.com/Gui34567/marketing-science-saas.git](https://github.com/Gui34567/marketing-science-saas.git)

# Instale as dependências
pip install -r requirements.txt

# Execute a aplicação
streamlit run app.py
