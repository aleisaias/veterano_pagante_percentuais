# app.py
# ================================================================
# "Veterano - Pagante | 5 Dispersões + 2 Heatmaps + Barras (IC95%)"
# Streamlit + Plotly — usa percentuais (P_CONV, P_REND100, …)
# ================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from math import atanh, tanh, sqrt
from scipy.stats import pearsonr, spearmanr, norm

st.set_page_config(page_title="Veterano - Pagante", layout="wide")

ROXO = "#9900FF"
NUM_COLS = ["P_CONV","P_REND100","P_MENOR40","P_REPROV","P_NAO_AI","P_MIX_INAD","AUM","D_DEV_BOLSA"]
ALL_COLS = ["MARCA"] + NUM_COLS

st.title("Veterano - Pagante • 5 Dispersões + 2 Heatmaps + Barras (IC95%)")
st.caption("Interativo (Streamlit + Plotly). Upload de CSV opcional; parser aceita vírgula decimal e sufixos %/pp.")

# ---------------------------
# Helpers
# ---------------------------
def to_float(x):
    """Converte '71,1%' / '-18,3 pp' → float (71.1 / -18.3)."""
    if pd.isna(x) or x == "":
        return np.nan
    s = str(x).lower().replace("%","").replace("pp","").replace(",",".").strip()
    try:
        return float(s)
    except:
        return np.nan

def fisher_ci(r, n, alpha=0.05):
    """IC 95% do r de Pearson (Fisher)."""
    if n <= 3 or not np.isfinite(r) or abs(r) >= 1:
        return (np.nan, np.nan)
    z = atanh(r); se = 1.0 / sqrt(n - 3); zcrit = norm.ppf(1 - alpha/2.0)
    return tanh(z - zcrit*se), tanh(z + zcrit*se)

def add_quadrant_lines(fig, x0, x1, y0, y1):
    fig.add_hline(y=0, line_dash="dash", opacity=0.6)
    fig.add_vline(x=0, line_dash="dash", opacity=0.6)
    fig.update_xaxes(range=[x0, x1])
    fig.update_yaxes(range=[y0, y1])

def scatter_corr(df, y_col, y_label, title):
    x = df["P_CONV"]; y = df[y_col]
    dx = (x.max() - x.min())*0.1 if np.isfinite(x.max()-x.min()) else 1
    dy = (y.max() - y.min())*0.1 if np.isfinite(y.max()-y.min()) else 1
    x0, x1 = x.min()-dx, x.max()+dx
    y0, y1 = y.min()-dy, y.max()+dy

    fig = px.scatter(
        df, x="P_CONV", y=y_col, text="MARCA",
        color="AUM", color_continuous_scale="Viridis_r",
        labels={"P_CONV":"% Conversão (nível atual)", y_col:y_label, "AUM":"Δ % Aumento Percebido"},
        title=title
    )
    fig.update_traces(marker=dict(size=12, line=dict(width=0.6, color="black")),
                      textposition="top center")
    add_quadrant_lines(fig, x0, x1, y0, y1)
    fig.update_layout(margin=dict(l=10,r=10,t=50,b=10))
    return fig

def heatmap_corr(df, method="pearson"):
    corr = df[NUM_COLS].corr(method=method)
    fig = go.Figure(data=go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index,
        zmin=-1, zmax=1, colorscale="RdBu", reversescale=True,
        colorbar=dict(title="Correlação")
    ))
    # anotações por célula
    for i, row in enumerate(corr.index):
        for j, col in enumerate(corr.columns):
            fig.add_annotation(
                x=col, y=row, text=f"{corr.iloc[i,j]:.2f}",
                showarrow=False, font=dict(size=12, color="black"),
                xanchor="center", yanchor="middle",
                bgcolor="rgba(255,255,255,0.7)"
            )
    fig.update_layout(
        title=f"Heatmap de Correlação — {method.capitalize()}",
        xaxis=dict(tickangle=45),
        margin=dict(l=80,r=20,t=60,b=120)
    )
    return fig

def barras_correlacoes(df):
    var_map = {
        "P_REND100": "REND",
        "P_MENOR40": "Média <40",
        "P_REPROV" : "Reprovação",
        "P_NAO_AI" : "NAO_AI",
        "P_MIX_INAD":"Inadimplência",
        "AUM"      : "AUM"
    }
    x_var = "P_CONV"
    labels, rvals, lo_list, hi_list = [], [], [], []
    for col, lbl in var_map.items():
        sub = df[[x_var, col]].dropna(); n = len(sub)
        if n < 3:
            r, lo, hi = np.nan, np.nan, np.nan
        else:
            r, _ = pearsonr(sub[x_var], sub[col]); lo, hi = fisher_ci(r, n)
        labels.append(lbl); rvals.append(r); lo_list.append(lo); hi_list.append(hi)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=rvals,
        marker_color=ROXO, marker_line=dict(color="black", width=0.8),
        name="Correlação (Pearson)"
    ))
    y_err_plus  = [ (hi - r) if np.isfinite(hi) and np.isfinite(r) else 0 for r, hi in zip(rvals, hi_list) ]
    y_err_minus = [ (r - lo) if np.isfinite(lo) and np.isfinite(r) else 0 for r, lo in zip(rvals, lo_list) ]
    fig.update_traces(error_y=dict(
        type='data', symmetric=False, array=y_err_plus, arrayminus=y_err_minus,
        thickness=1.2, width=5, color="black"
    ))
    fig.add_hline(y=0, line_dash="dash", opacity=0.7)
    fig.update_layout(
        title="Correlações com Conversão e IC 95%",
        yaxis_title="Coeficiente de Correlação (Pearson)",
        margin=dict(l=40,r=20,t=60,b=40)
    )
    return fig

# ---------------------------
# Base padrão (se não subir CSV)
# ---------------------------
raw_default = [
 ["ÂNIMA BR","90,3%","86,1%","6,4%","13,2%","22,5%","12,0%","90,2%","-18,3 pp"],
 ["AGES","91,1%","90,3%","3,8%","9,2%","13,2%","12,2%","101,6%","-22,4 pp"],
 ["UNIFG - BA","90,5%","90,9%","3,8%","8,5%","15,2%","13,7%","91,1%","-26,5 pp"],
 ["UNF","88,6%","80,9%","8,9%","18,7%","16,4%","13,4%","112,4%","-18,4 pp"],
 ["UNP","90,5%","87,9%","5,5%","11,4%","19,5%","15,1%","119,9%","-13,5 pp"],
 ["FPB","87,7%","84,7%","7,3%","14,2%","25,5%","14,8%","98,7%","-7,8 pp"],
 ["UNIFG - PE","87,1%","82,7%","9,0%","16,3%","31,5%","10,7%","114,4%","-14,6 pp"],
 ["UAM","91,1%","86,9%","6,3%","12,3%","28,6%","9,6%","101,4%","0,0 pp"],
 ["USJT","90,6%","86,7%","6,4%","12,6%","30,0%","10,9%","67,2%","0,0 pp"],
 ["UNA","90,9%","87,5%","5,7%","11,8%","20,7%","14,1%","61,1%","0,0 pp"],
 ["UNIBH","88,6%","87,1%","5,9%","12,3%","16,1%","11,9%","81,8%","0,0 pp"],
 ["IBMR","89,1%","82,6%","7,3%","16,7%","26,7%","6,6%","149,7%","-22,2 pp"],
 ["FASEH","90,2%","91,6%","4,2%","8,2%","16,1%","15,5%","74,6%","0,0 pp"],
 ["MIL. CAMPOS","89,5%","61,8%","8,5%","37,9%","0,0%","14,7%","32,5%","0,0 pp"],
 ["UNISUL","92,5%","86,4%","7,3%","13,2%","19,5%","11,6%","82,7%","0,0 pp"],
 ["UNICURITIBA","89,0%","84,1%","6,6%","15,3%","26,7%","12,7%","68,4%","0,0 pp"],
 ["UNISOCIESC","91,7%","89,4%","5,1%","9,6%","22,8%","11,8%","67,9%","0,0 pp"],
 ["UNR","90,0%","82,7%","7,2%","16,8%","22,3%","10,9%","108,4%","-23,7 pp"],
 ["FAD","89,5%","82,1%","8,8%","17,5%","23,7%","8,5%","141,5%","-16,1 pp"]
]
df_default = pd.DataFrame(raw_default, columns=ALL_COLS)
for c in NUM_COLS:
    df_default[c] = df_default[c].apply(to_float)

# ---------------------------
# Sidebar: Upload CSV (opcional)
# ---------------------------
st.sidebar.header("Dados")
st.sidebar.write("CSV esperado com colunas:")
st.sidebar.code(", ".join(ALL_COLS), language="text")
uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded:
    try:
        df = pd.read_csv(uploaded, dtype=str)
        missing = [c for c in ALL_COLS if c not in df.columns]
        if missing:
            st.sidebar.error(f"Faltam colunas no CSV: {missing}. Usando base de exemplo.")
            df = df_default.copy()
        else:
            for c in NUM_COLS:
                df[c] = df[c].apply(to_float)
    except Exception as e:
        st.sidebar.error(f"Erro ao ler CSV: {e}. Usando base de exemplo.")
        df = df_default.copy()
else:
    df = df_default.copy()

# ---------------------------
# Abas e gráficos
# ---------------------------
tab1, tab2, tab3 = st.tabs(["Dispersões", "Heatmaps", "Barras (IC95%)"])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            scatter_corr(df, "P_MENOR40", "% com Média < 40", "% Conversão (X) vs % Média < 40 (Y)"),
            use_container_width=True
        )
    with c2:
        st.plotly_chart(
            scatter_corr(df, "P_REPROV", "% Reprovado", "% Conversão (X) vs % Reprovado (Y)"),
            use_container_width=True
        )
    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(
            scatter_corr(df, "P_REND100", "% Rendimento 100%", "% Conversão (X) vs % Rendimento 100% (Y)"),
            use_container_width=True
        )
    with c4:
        st.plotly_chart(
            scatter_corr(df, "P_MIX_INAD", "% Mix Inadimplência", "% Conversão (X) vs % Mix Inadimplência (Y)"),
            use_container_width=True
        )
    st.plotly_chart(
        scatter_corr(df, "P_NAO_AI", "% Não Realizou AI", "% Conversão (X) vs % Não Realizou AI (Y)"),
        use_container_width=True
    )

with tab2:
    colA, colB = st.columns(2)
    with colA:
        st.plotly_chart(heatmap_corr(df, method="pearson"), use_container_width=True)
    with colB:
        st.plotly_chart(heatmap_corr(df, method="spearman"), use_container_width=True)

with tab3:
    st.plotly_chart(barras_correlacoes(df), use_container_width=True)

st.markdown("---")
st.caption(f"Barras em roxo {ROXO}. Dispersões com escala Viridis invertida (cor = Δ % AUM).")
