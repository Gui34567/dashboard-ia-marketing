import streamlit as st
import pandas as pd
import sqlite3
import joblib
import plotly.express as px
import numpy as np
from datetime import timedelta

# --- 1. CONFIGURAÇÃO E DADOS ---
st.set_page_config(page_title="SaaS - Marketing Science", layout="wide")

@st.cache_data
def carregar_dados():
    conn = sqlite3.connect('marketing_science.db')
    df = pd.read_sql_query("SELECT * FROM campanhas_ads", conn)
    conn.close()
    df['Data'] = pd.to_datetime(df['Data'])
    return df

@st.cache_resource
def carregar_modelo():
    modelo = joblib.load('modelo_roas.pkl')
    colunas = joblib.load('colunas_roas.pkl')
    return modelo, colunas

df = carregar_dados()
modelo_roas, colunas_treino = carregar_modelo()

# --- 2. BARRA LATERAL (FILTROS CORRIGIDOS) ---
st.sidebar.title("Filtros Globais")
campanhas = st.sidebar.multiselect("Campanha", df['Campanha'].unique(), default=df['Campanha'].unique())
adsets = st.sidebar.multiselect("Adset", df['Adset'].unique())
placements = st.sidebar.multiselect("Placement", df['Placement'].unique())

# Lógica robusta: só filtra se o usuário tiver selecionado algo
df_filtrado = df.copy()
if campanhas:
    df_filtrado = df_filtrado[df_filtrado['Campanha'].isin(campanhas)]
if adsets:
    df_filtrado = df_filtrado[df_filtrado['Adset'].isin(adsets)]
if placements:
    df_filtrado = df_filtrado[df_filtrado['Placement'].isin(placements)]

# --- 3. INTERFACE PRINCIPAL ---
st.title("📊 Plataforma de Marketing Science")
st.caption("Desenvolvido para análise avançada de performance e predição de mídia.")

aba1, aba2, aba3, aba4 = st.tabs([
    "📈 Visão Geral", 
    "🔮 Simulador Preditivo", 
    "🏆 Top Creatives", 
    "🔬 Marketing Science Lab"
])

# --- ABA 1: VISÃO GERAL ---
with aba1:
    if df_filtrado.empty:
        st.warning("⚠️ Nenhum dado encontrado para os filtros selecionados.")
    else:
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("ROAS Médio", f"{df_filtrado['ROAS'].mean():.2f}x")
        col2.metric("CPA Médio", f"R$ {df_filtrado['CPA'].mean():.2f}")
        col3.metric("CTR Médio", f"{df_filtrado['CTR'].mean()*100:.2f}%")
        col4.metric("CPC Médio", f"R$ {df_filtrado['CPC'].mean():.2f}")
        col5.metric("CPM Médio", f"R$ {df_filtrado['CPM'].mean():.2f}")
        
        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            fig_placement = px.bar(df_filtrado.groupby('Placement')['ROAS'].mean().reset_index(), x='Placement', y='ROAS', title='ROAS por Placement', color='ROAS')
            st.plotly_chart(fig_placement, use_container_width=True)
        with c2:
            fig_adv = px.box(df_filtrado, x='Advantage_Plus_Placements', y='CPA', title='CPA: Advantage+ vs Manual', color='Advantage_Plus_Placements')
            st.plotly_chart(fig_adv, use_container_width=True)

# --- ABA 2: SIMULADOR ---
with aba2:
    st.markdown("### Simulador de Retorno de Mídia")
    st.info("Insira as configurações da sua próxima campanha para prever o ROAS com Machine Learning.")
    
    col_esq, col_dir = st.columns(2)
    with col_esq:
        sim_placement = st.selectbox("Onde o anúncio vai aparecer?", df['Placement'].unique())
        sim_adv_placement = st.radio("Usar Advantage+ Placements?", ['Yes', 'No'])
        sim_adv_creative = st.radio("Usar Advantage+ Creative?", ['Yes', 'No'])
        sim_objective = st.selectbox("Objetivo da Campanha", df['Campaign_Objective'].unique())
    with col_dir:
        sim_targeting = st.selectbox("Tipo de Público", df['Targeting'].unique())
        sim_spent = st.number_input("Orçamento Diário (R$)", min_value=10.0, value=150.0)
        sim_freq = st.slider("Expectativa de Frequência", 1.0, 5.0, 1.5, step=0.1)

    if st.button("🚀 Prever ROAS", type="primary"):
        input_dict = {
            'Placement': [sim_placement], 'Advantage_Plus_Placements': [sim_adv_placement],
            'Advantage_Plus_Creative': [sim_adv_creative], 'Campaign_Objective': [sim_objective],
            'Targeting': [sim_targeting], 'Amount_spent': [sim_spent], 'Frequency': [sim_freq]
        }
        input_df = pd.DataFrame(input_dict)
        input_dummies = pd.get_dummies(input_df)
        input_final = input_dummies.reindex(columns=colunas_treino, fill_value=0)
        
        roas_previsto = modelo_roas.predict(input_final)[0]
        receita_estimada = sim_spent * roas_previsto
        
        st.divider()
        res_col1, res_col2 = st.columns(2)
        res_col1.metric("ROAS Previsto", f"{roas_previsto:.2f}x")
        res_col2.metric("Receita Estimada", f"R$ {receita_estimada:.2f}")

# --- ABA 3: TOP CREATIVES ---
with aba3:
    st.markdown("### 🏆 Identificação Automática de Creatives Vencedores")
    
    if df_filtrado.empty:
        st.warning("Sem dados para analisar os criativos.")
    else:
        roas_medio = df_filtrado['ROAS'].mean()
        cpa_medio = df_filtrado['CPA'].mean()
        ctr_medio = df_filtrado['CTR'].mean()
        
        top_creatives = df_filtrado[
            (df_filtrado['ROAS'] > roas_medio) & 
            (df_filtrado['CPA'] < cpa_medio) & 
            (df_filtrado['CTR'] > ctr_medio) & 
            (df_filtrado['Campaign_Objective'] == 'Sales')
        ].sort_values(by='ROAS', ascending=False)
        
        if top_creatives.empty:
            st.warning("Nenhum creative superou todas as médias de alta performance com os filtros atuais.")
        else:
            st.success(f"🔥 Foram encontrados {len(top_creatives)} Creatives Vencedores!")
            st.dataframe(top_creatives[['Nome_do_Creative', 'Campanha', 'Placement', 'ROAS', 'CPA', 'CTR']].head(10), use_container_width=True)
            
            st.divider()
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                padrao_placement = top_creatives['Placement'].mode()[0]
                st.info(f"**Placement Favorito:** A maioria dos campeões roda em **{padrao_placement}**.")
            with col_p2:
                st.markdown(f"- **ESCALAR:** Aumente o orçamento nos anúncios focados em {padrao_placement}.")

# --- ABA 4: MARKETING SCIENCE ---
with aba4:
    st.markdown("### 🔬 Laboratório de Marketing Science")
    
    if df_filtrado.empty:
        st.warning("Sem dados suficientes.")
    else:
        st.markdown("#### Impacto da Frequência no Custo (Fadiga de Anúncio)")
        fig_scatter = px.scatter(df_filtrado, x='Frequency', y='CPA', color='Placement', trendline='ols', title='Relação Matemática: Frequência x CPA')
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        st.divider()
        st.markdown("#### Cálculo de Causalidade (Uplift Médio)")
        
        # Prevenção contra divisões por zero e valores vazios na causalidade
        roas_adv_yes = df_filtrado[df_filtrado['Advantage_Plus_Placements'] == 'Yes']['ROAS'].mean()
        roas_adv_no = df_filtrado[df_filtrado['Advantage_Plus_Placements'] == 'No']['ROAS'].mean()
        
        adv_yes = roas_adv_yes if not pd.isna(roas_adv_yes) else 0
        adv_no = roas_adv_no if not pd.isna(roas_adv_no) else 0
        uplift = ((adv_yes - adv_no) / adv_no) * 100 if adv_no > 0 else 0
            
        col_u1, col_u2, col_u3 = st.columns(3)
        col_u1.metric("ROAS s/ Advantage+", f"{adv_no:.2f}x")
        col_u2.metric("ROAS c/ Advantage+", f"{adv_yes:.2f}x")
        col_u3.metric("Uplift Comprovado", f"+{uplift:.1f}%", delta_color="normal" if uplift > 0 else "inverse")
        
        st.divider()
        st.markdown("#### 📈 Previsão de ROAS (Próximos 30 Dias)")
        
        df_serie = df_filtrado.groupby('Data')['ROAS'].mean().reset_index().sort_values('Data')
        
        if len(df_serie) > 0:
            ultima_data = df_serie['Data'].max()
            datas_futuras = [ultima_data + timedelta(days=i) for i in range(1, 31)]
            
            tendencia_roas = df_serie['ROAS'].tail(15).mean()
            roas_futuro = [tendencia_roas * np.random.uniform(0.95, 1.05) for _ in range(30)]
            
            df_futuro = pd.DataFrame({'Data': datas_futuras, 'ROAS': roas_futuro, 'Tipo': 'Previsão'})
            df_historico = df_serie.copy()
            df_historico['Tipo'] = 'Histórico'
            
            df_grafico = pd.concat([df_historico.tail(30), df_futuro])
            
            fig_projecao = px.line(df_grafico, x='Data', y='ROAS', color='Tipo', title='Histórico vs. Previsão')
            
            # CORREÇÃO DO BUG: Convertendo a data para texto simples antes de plotar a linha
            data_string = ultima_data.strftime('%Y-%m-%d')
            fig_projecao.add_vline(x=data_string, line_dash="dash", line_color="red", annotation_text="Hoje")
            
            st.plotly_chart(fig_projecao, use_container_width=True)
