import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import statsmodels.api as sm
import openpyxl
import io
from PIL import Image

# Configuração da página
st.set_page_config(page_title="Análise Estatística - Pediatria", 
                   layout="wide", 
                   page_icon="👶")

# Título com estilo
st.markdown("""
    <style>
        .title {
            font-size: 36px !important;
            color: #1f77b4 !important;
            text-align: center;
            padding-bottom: 20px;
        }
        .section-header {
            font-size: 24px !important;
            color: #2ca02c !important;
            border-bottom: 2px solid #2ca02c;
            padding-bottom: 10px;
            margin-top: 30px !important;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="title">Análise de Dados de Pediatria (Arango, 2001)</h1>', unsafe_allow_html=True)

# Constantes
DATA_PATH = "data/dadosPediatria.xlsx"
VARS_NUMERICAS = ["T_GEST", "PESO", "ESTATURA", "PC", "PT"]
VARS_CATEGORICAS = ["SEXO", "SANGUE", "RH", "ANOMALIA"]
COLORS = px.colors.qualitative.Plotly

# ==============================================
# FUNÇÕES AUXILIARES PARA VISUALIZAÇÃO
# ==============================================


def create_pretty_boxplot(df, x, y, title, color_map=None):
    """Cria um boxplot estilizado"""
    fig = px.box(df, x=x, y=y, color=x if color_map is None else None,
                 color_discrete_map=color_map,
                 points="all",
                 title=title,
                 template="plotly_white")
    
    fig.update_layout(
        hovermode="x unified",
        showlegend=False,
        font=dict(size=12),
        plot_bgcolor='rgba(240,240,240,0.8)',
        paper_bgcolor='rgba(240,240,240,0.8)',
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

def create_animated_scatter(df, x, y, color_col, title, size=None):
    """Cria um gráfico de dispersão animado"""
    fig = px.scatter(df, x=x, y=y, color=color_col,
                    size=size,
                    animation_frame=color_col if df[color_col].nunique() <= 10 else None,
                    title=title,
                    template="plotly_white",
                    color_discrete_sequence=COLORS,
                    hover_name=color_col)
    
    fig.update_layout(
        transition={'duration': 1000},
        font=dict(size=12),
        plot_bgcolor='rgba(240,240,240,0.8)',
        paper_bgcolor='rgba(240,240,240,0.8)'
    )
    return fig

def create_radar_chart(df, categories, title):
    """Cria um gráfico de radar para comparação"""
    fig = go.Figure()

    for sexo in df['SEXO'].unique():
        df_sexo = df[df['SEXO'] == sexo]
        fig.add_trace(go.Scatterpolar(
            r=df_sexo[categories].mean().values,
            theta=categories,
            fill='toself',
            name=sexo,
            line_color=COLORS[0] if sexo == 'M' else COLORS[1]
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, df[categories].max().max() * 1.1]
            )),
        showlegend=True,
        title=title,
        template="plotly_white"
    )
    return fig

def create_correlation_heatmap(df, vars, title):
    """Cria uma matriz de correlação visual"""
    corr = df[vars].corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.index,
        colorscale='RdBu',
        zmin=-1,
        zmax=1,
        hoverongaps=False,
        text=corr.round(2).values,
        texttemplate="%{text}"
    ))
    
    fig.update_layout(
        title=title,
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        yaxis_autorange='reversed',
        template="plotly_white"
    )
    return fig

def create_distribution_grid(df, vars, color_col=None):
    """Cria uma grade de gráficos de distribuição"""
    n_cols = 2
    n_rows = (len(vars) + 1) // n_cols
    
    fig = make_subplots(rows=n_rows, cols=n_cols, 
                       subplot_titles=[f"Distribuição de {var}" for var in vars])
    
    for i, var in enumerate(vars):
        row = (i // n_cols) + 1
        col = (i % n_cols) + 1
        
        if color_col and df[color_col].nunique() <= 5:
            for j, category in enumerate(df[color_col].unique()):
                fig.add_trace(
                    go.Histogram(
                        x=df[df[color_col] == category][var],
                        name=f"{color_col}={category}",
                        marker_color=COLORS[j],
                        opacity=0.7,
                        showlegend=(i == 0)
                    ),
                    row=row, col=col
                )
        else:
            fig.add_trace(
                go.Histogram(
                    x=df[var],
                    marker_color=COLORS[0],
                    opacity=0.7,
                    showlegend=False
                ),
                row=row, col=col
            )
    
    fig.update_layout(
        height=300 * n_rows,
        title_text="Distribuição das Variáveis",
        template="plotly_white",
        barmode='overlay'
    )
    fig.update_traces(marker_line_width=1, marker_line_color="white")
    
    return fig

# ==============================================
# FUNÇÕES PRINCIPAIS DE ANÁLISE
# ==============================================

def load_data():
    """Carrega e prepara os dados"""
    try:
        df = pd.read_excel(DATA_PATH)
        
        # Adicionar coluna de categoria de peso para visualização
        df['CATEGORIA_PESO'] = pd.cut(df['PESO'], 
                                     bins=[0, 2500, 3000, 3500, 4000, 10000],
                                     labels=['<2500g', '2500-3000g', '3000-3500g', '3500-4000g', '>4000g'])
        return df
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        return None

def show_data_overview(df):
    """Mostra visão geral dos dados"""
    st.markdown('<h2 class="section-header">Visão Geral dos Dados</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total de Registros", len(df))
        st.metric("Variáveis Numéricas", len(VARS_NUMERICAS))
        st.metric("Variáveis Categóricas", len(VARS_CATEGORICAS))
    
    with col2:
        st.metric("Média de Peso", f"{df['PESO'].mean():.1f}g")
        st.metric("Média de Estatura", f"{df['ESTATURA'].mean():.1f}cm")
        st.metric("Tempo Gestacional Médio", f"{df['T_GEST'].mean():.1f} dias")
    
    # Gráfico de distribuição de sexo e anomalias
    fig = px.sunburst(df, path=['SEXO', 'ANOMALIA'], 
                     color='SEXO', color_discrete_map={'M': COLORS[0], 'F': COLORS[1]},
                     title="Distribuição por Sexo e Anomalias")
    st.plotly_chart(fig, use_container_width=True)

def show_descriptive_analysis(df):
    """Mostra análise descritiva com visualizações"""
    st.markdown('<h2 class="section-header">Análise Descritiva</h2>', unsafe_allow_html=True)
    
    # Grade de distribuições
    st.plotly_chart(create_distribution_grid(df, VARS_NUMERICAS, 'SEXO'), 
                   use_container_width=True)
    
    # Boxplots comparativos
    st.markdown("### Comparação por Sexo")
    cols = st.columns(2)
    for i, var in enumerate(VARS_NUMERICAS):
        with cols[i % 2]:
            fig = create_pretty_boxplot(df, 'SEXO', var, 
                                      f"Distribuição de {var} por Sexo",
                                      color_map={'M': COLORS[0], 'F': COLORS[1]})
            st.plotly_chart(fig, use_container_width=True)
    
    # Gráfico de radar para comparação multivariada
    st.markdown("### Comparação Multivariada por Sexo")
    st.plotly_chart(create_radar_chart(df, ['PESO', 'ESTATURA', 'PC', 'PT'], 
                                      "Comparação de Médias por Sexo"), 
                   use_container_width=True)

def show_comparative_analysis(df):
    """Mostra análise comparativa entre grupos"""
    st.markdown('<h2 class="section-header">Análise Comparativa</h2>', unsafe_allow_html=True)
    
    # Matriz de correlação
    st.markdown("### Correlação entre Variáveis Numéricas")
    st.plotly_chart(create_correlation_heatmap(df, VARS_NUMERICAS, 
                                             "Matriz de Correlação"), 
                   use_container_width=True)
    
    # Scatter plot animado
    st.markdown("### Relação entre Peso e Outras Variáveis")
    fig = create_animated_scatter(df, 'PESO', 'ESTATURA', 'CATEGORIA_PESO', 
                                 "Relação Peso x Estatura por Categoria", 'PT')
    st.plotly_chart(fig, use_container_width=True)

def show_statistical_tests(df):
    """Mostra resultados de testes estatísticos"""
    st.markdown('<h2 class="section-header">Testes Estatísticos</h2>', unsafe_allow_html=True)
    
    # Teste T com visualização
    st.markdown("### Comparação de Médias (Teste T)")
    for var in VARS_NUMERICAS:
        col1, col2 = st.columns([1, 2])
        with col1:
            group_m = df[df['SEXO'] == 'M'][var].dropna()
            group_f = df[df['SEXO'] == 'F'][var].dropna()
            t, p = stats.ttest_ind(group_m, group_f)
            
            st.metric(f"Diferença de {var} entre sexos", 
                     f"{group_m.mean() - group_f.mean():.2f}")
            st.metric("Valor-p", f"{p:.4f}", 
                     "Significativo" if p < 0.05 else "Não Significativo")
        
        with col2:
            fig = go.Figure()
            fig.add_trace(go.Box(y=group_m, name='Masculino', marker_color=COLORS[0]))
            fig.add_trace(go.Box(y=group_f, name='Feminino', marker_color=COLORS[1]))
            fig.update_layout(title=f"Distribuição de {var} por Sexo",
                            template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
    
    # Teste Qui-quadrado
    st.markdown("### Associação entre Variáveis Categóricas")
    cont_table = pd.crosstab(df['SEXO'], df['ANOMALIA'])
    chi2, p, _, _ = stats.chi2_contingency(cont_table)
    
    fig = px.bar(cont_table.reset_index(), x='SEXO', y=['SIM', 'NÃO'], 
                barmode='group', title="Distribuição de Anomalias por Sexo",
                color_discrete_sequence=[COLORS[3], COLORS[4]])
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Qui-quadrado", f"{chi2:.2f}")
    with col2:
        st.metric("Valor-p", f"{p:.4f}", 
                 "Significativo" if p < 0.05 else "Não Significativo")
        
def show_confidence_intervals(df):
    st.markdown("### Intervalos de Confiança (95%) para Variáveis Numéricas")
    
    for var in ['PESO', 'ESTATURA', 'PC', 'PT']:
        data = df[var].dropna()
        mean = data.mean()
        sem = stats.sem(data)
        ci = stats.t.interval(0.95, len(data)-1, loc=mean, scale=sem)

        st.write(f"**{var}**:")
        st.write(f"Média: {mean:.2f}")
        st.write(f"IC 95%: ({ci[0]:.2f}, {ci[1]:.2f})")

def show_proportional_analysis(df):
    st.markdown("### Proporção de RCs (Sexo Feminino, Sangue Tipo O, RH+, com Anomalia)")
    
    total_rc = len(df)
    filtro = (df['SEXO'] == 'F') & (df['SANGUE'] == 'O') & (df['RH'] == 'POS') & (df['ANOMALIA'] == 'SIM')
    count = df[filtro].shape[0]
    prop = (count / total_rc) * 100

    st.write(f"Número de RCs com essas características: {count}")
    st.write(f"Proporção: {prop:.2f}%")

def show_normality_tests(df):
    st.markdown("### Teste de Normalidade (Shapiro-Wilk)")
    
    for var in ['PESO', 'ESTATURA', 'PC', 'PT']:
        data = df[var].dropna()
        stat, p = stats.shapiro(data)
        st.write(f"**{var}**: Estatística W = {stat:.4f}, p-valor = {p:.4f}")
        if p > 0.05:
            st.success(f"A distribuição de {var} **é normal** (p > 0.05)")
        else:
            st.warning(f"A distribuição de {var} **não é normal** (p ≤ 0.05)")


def show_regression_analysis(df):
    """Mostra análise de regressão"""
    st.markdown('<h2 class="section-header">Análise de Regressão</h2>', unsafe_allow_html=True)
    
    # Seleção de variáveis
    col1, col2 = st.columns(2)
    with col1:
        y_var = st.selectbox("Variável Dependente (Y):", VARS_NUMERICAS)
    with col2:
        x_var = st.selectbox("Variável Independente (X):", 
                            [v for v in VARS_NUMERICAS if v != y_var])
    
    if st.button("Executar Regressão"):
        # Gráfico de dispersão com linha de regressão
        fig = px.scatter(df, x=x_var, y=y_var, 
                        trendline="ols",
                        color='SEXO',
                        color_discrete_map={'M': COLORS[0], 'F': COLORS[1]},
                        title=f"Regressão Linear: {y_var} ~ {x_var}",
                        template="plotly_white")
        
        # Cálculo da regressão
        X = sm.add_constant(df[x_var])
        model = sm.OLS(df[y_var], X, missing='drop').fit()
        
        # Exibição dos resultados
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.metric("Coeficiente de Correlação (r)", 
                     f"{np.sqrt(model.rsquared):.3f}")
            st.metric("R² Ajustado", f"{model.rsquared_adj:.3f}")
            st.metric("Intercepto", f"{model.params[0]:.2f}")
            st.metric(f"Coeficiente para {x_var}", 
                     f"{model.params[1]:.2f}",
                     f"p = {model.pvalues[1]:.4f}")
        
        # Resumo completo em expansor
        with st.expander("Ver detalhes do modelo"):
            st.text(model.summary())

# ==============================================
# INTERFACE PRINCIPAL
# ==============================================

def main():
    """Função principal da aplicação"""
    # Sidebar
    st.sidebar.title("👶 Painel de Controle")
    analysis_option = st.sidebar.radio(
        "Selecione a Análise:",
        ["Visão Geral", "Análise Descritiva", "Análise Comparativa", 
         "Testes Estatísticos", "Análise de Regressão",
         "Intervalos de Confiança Peso,Estatura,PC,PT",
         "Proporção de (RC) feminino com sangue tipo O, RH+ e portadores de anomalia",
         "Teste de Normalidade (Shapiro-Wilk)"]
    )
    
    # Carregar dados
    df = load_data()
    
    if df is not None:
        # Executar análise selecionada
        if analysis_option == "Visão Geral":
            show_data_overview(df)
        elif analysis_option == "Análise Descritiva":
            show_descriptive_analysis(df)
        elif analysis_option == "Análise Comparativa":
            show_comparative_analysis(df)
        elif analysis_option == "Testes Estatísticos":
            show_statistical_tests(df)
        elif analysis_option == "Análise de Regressão":
            show_regression_analysis(df)
        elif analysis_option == "Intervalos de Confiança Peso,Estatura,PC,PT":
            show_confidence_intervals(df)
        elif analysis_option == "Proporção de (RC) feminino com sangue tipo O, RH+ e portadores de anomalia":
            show_proportional_analysis(df)
        elif analysis_option == "Teste de Normalidade (Shapiro-Wilk)":
            show_normality_tests(df)
        
        # Rodapé
        st.sidebar.markdown("---")
        st.sidebar.markdown("""
            **Análise Estatística**  
            Dados de Pediatria (Arango, 2001)  
            Desenvolvido com Python e Streamlit
        """)


if __name__ == "__main__":
    main()