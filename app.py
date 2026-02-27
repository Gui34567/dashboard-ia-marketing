import streamlit as st
import pandas as pd
import sqlite3
import joblib

# 1. Configuração visual
st.set_page_config(page_title="Nexos AI - Inteligência de Receita", layout="wide")

# 2. Carregar o Modelo e as Colunas (O Cérebro treinado)
@st.cache_resource
def carregar_modelo():
    modelo = joblib.load('modelo_marketing.pkl')
    colunas = joblib.load('colunas_treino.pkl')
    return modelo, colunas

modelo, colunas_treino = carregar_modelo()

# 3. Carregar os Dados do SQL
@st.cache_data
def carregar_dados():
    conn = sqlite3.connect('banco_dados_marketing.db')
    df = pd.read_sql_query("SELECT * FROM clientes_marketing", conn)
    conn.close()
    return df

df = carregar_dados()

# 4. Construindo a Interface
st.title("🚀 Nexos AI - Plataforma de Inteligência de Receita")

aba1, aba2 = st.tabs(["📊 Dashboard SQL", "🔮 Simulador da IA"])

with aba1:
    st.header("Visão Geral da Base de Dados")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total de Clientes", len(df))
    col2.metric("Ticket Médio", f"R$ {df['Valor_Venda'].mean():.2f}")
    col3.metric("CAC Médio", f"R$ {df['CAC'].mean():.2f}")
    
    st.subheader("Amostra dos Dados (SQLite)")
    st.dataframe(df.head(10))

with aba2:
    st.header("Simulador de Risco (Churn)")
    st.write("Altere as variáveis. A IA fará a predição em tempo real se o cliente vai nos abandonar.")
    
    col_esq, col_dir = st.columns(2)
    
    with col_esq:
        valor_venda = st.number_input("Valor da Venda (R$)", min_value=100.0, max_value=10000.0, value=1500.0)
        cac = st.number_input("Custo de Aquisição (CAC - R$)", min_value=10.0, max_value=1000.0, value=150.0)
        score_engajamento = st.slider("Score de Engajamento (0-100)", 0, 100, 20)
        dias_ultima_compra = st.number_input("Dias Desde Última Compra", min_value=1, max_value=365, value=75)
        frequencia = st.number_input("Frequência (Compras/Ano)", min_value=1, max_value=50, value=2)

    with col_dir:
        produto = st.selectbox("Produto", df['Produto'].unique())
        categoria = st.selectbox("Categoria", df['Categoria'].unique())
        origem_lead = st.selectbox("Origem do Lead", df['Origem_Lead'].unique())
        r_score = st.selectbox("Recência (R_Score 1-5)", [1, 2, 3, 4, 5], index=0)
        f_score = st.selectbox("Frequência (F_Score 1-5)", [1, 2, 3, 4, 5], index=1)
        m_score = st.selectbox("Valor (M_Score 1-5)", [1, 2, 3, 4, 5], index=2)

    if st.button("🔮 Prever Risco agora", type="primary"):
        # 1. Pega os dados que o usuário digitou
        input_dict = {
            'Valor_Venda': [valor_venda], 'CAC': [cac], 'Score_Engajamento': [score_engajamento],
            'Dias_Desde_Ultima_Compra': [dias_ultima_compra], 'Frequencia': [frequencia],
            'Produto': [produto], 'Categoria': [categoria], 'Origem_Lead': [origem_lead],
            'R_Score': [r_score], 'F_Score': [f_score], 'M_Score': [m_score]
        }
        input_df = pd.DataFrame(input_dict)
        
        # 2. Transforma os textos em números (exatamente como no treinamento)
        input_dummies = pd.get_dummies(input_df)
        
        # 3. Alinha as colunas para a IA não se perder (preenche o que faltar com 0)
        input_final = input_dummies.reindex(columns=colunas_treino, fill_value=0)
        
        # 4. A IA faz a previsão!
        resultado = modelo.predict(input_final)[0]
        
        st.divider()
        if resultado == 1:
            st.error("🚨 ALERTA: Este perfil tem Alto Risco de Cancelamento!")
            st.info("💡 Sugestão: Enviar cupom de desconto de 15% imediatamente.")
        else:
            st.success("✅ Cliente Seguro. Baixo risco de cancelamento.")
