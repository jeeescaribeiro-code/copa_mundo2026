"""
=============================================================
COPA DO MUNDO 2026
=============================================================
"""

import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import os
from pathlib import Path

from ingestao_dados import RENAME_JOGOS

BASE_DIR = Path(__file__).resolve().parent
OUTPUT = BASE_DIR / "output"
os.makedirs(OUTPUT, exist_ok=True)

#Estilo global dos gráficos
plt.rcParams.update({
    "figure.facecolor" : "white",
    "axes.facecolor"   : "#F9F9F9",
    "axes.grid"        : True,
    "grid.alpha"       : 0.3,
    "grid.linestyle"   : "--",
    "font.family"      : "sans-serif",
    "axes.spines.top"  : False,
    "axes.spines.right": False,
})
AZUL= "#1A5276"
VERDE= "#1E8449"
LARANJA= "#D85A30"
CINZA= "#7F8C8D"
AMARELO= "#D4AC0D"

#Primeiro, carregamos o dataset de jogos processado na etapa 1. Ele já deve conter as colunas principais, mas faremos ajustes adicionais para garantir consistência e criar novas colunas derivadas.
jogos = pd.read_csv(os.path.join(OUTPUT, "jogos.csv"))
print(f"  Shape original: {jogos.shape}")
#Renomeia apenas as colunas que existirem (seguro para ambas as versões)
jogos = jogos.rename(columns={k: v for k, v in RENAME_JOGOS.items() if k in jogos.columns})
print(f"  Colunas padronizadas: OK")

#Extrair ano do tournament_id
jogos["year"] = jogos["tournament_id"].str.extract(r"(\d{4})").astype(int)

#Garantir tipos numéricos nos placares
jogos["home_score"] = pd.to_numeric(jogos["home_score"], errors="coerce").fillna(0).astype(int)
jogos["away_score"] = pd.to_numeric(jogos["away_score"], errors="coerce").fillna(0).astype(int)

#Colunas derivadas
jogos["total_gols"] = jogos["home_score"] + jogos["away_score"]
jogos["saldo_home"] = jogos["home_score"] - jogos["away_score"]

#Fase do torneio: ordinal e flag knockout
FASE_ORDINAL = {
    "group stage" : 1,
    "second group stage" : 2,
    "final round" : 2,
    "round of 16": 3,
    "quarter-finals": 4,
    "semi-finals": 5,
    "third-place match" : 6,
    "final" : 7,
}
jogos["fase_ordinal"]  = jogos["stage_name"].str.lower().map(FASE_ORDINAL).fillna(1).astype(int)
jogos["fase_knockout"] = (jogos["fase_ordinal"] >= 3).astype(int)

print(f"Shape final:{jogos.shape}")
print(f"Colunas: {jogos.columns.tolist()}\n")

"""
Em resumo, o dataset de jogos foi carregado e processado para garantir consistência nos nomes das colunas, tipos de dados adequados e a criação de novas colunas que facilitam a análise. Com isso, estamos prontos para avançar para a etapa de análise exploratória dos dados (EDA), onde vamos extrair insights e criar visualizações para entender melhor os padrões e tendências históricas da Copa do Mundo FIFA.
"""

#Segundo, vamos realizar uma análise exploratória dos dados (EDA) para entender as principais características do dataset, identificar padrões e destacar insights relevantes sobre os jogos da Copa do Mundo FIFA entre 1930 e 2022.
print("=" * 55)
print("ESTATÍSTICAS GERAIS")
print("=" * 55)
print(f"Total de partidas analisadas: {len(jogos)}")
print(f"Edições (1930–2022): {jogos['year'].nunique()}")
print(f"Total de gols marcados : {jogos['total_gols'].sum()}")
print(f"Média de gols por partida : {jogos['total_gols'].mean():.2f}")
print(f"Partida com mais gols : {jogos['total_gols'].max()} gols")
print(f"Partidas com prorrogação : {int(jogos['extra_time'].sum())}")
print(f"Partidas com pênaltis: {int(jogos['penalty_shootout'].sum())}")

#Distribuição de resultados
print("\nDISTRIBUIÇÃO DE RESULTADOS")
print("-" * 40)
dist_resultado = jogos["result"].value_counts()
for res, count in dist_resultado.items():
    pct = count / len(jogos) * 100
    print(f"  {res:<20} {count:>4} partidas  ({pct:.1f}%)")

#Gols por edição
print("\nGOLS POR EDIÇÃO")
print("-" * 40)
por_ano = (jogos.groupby("year")
           .agg(partidas=("total_gols","count"),
                total=("total_gols","sum"),
                media=("total_gols","mean"))
           .round(2))
print(por_ano.to_string())

#Top 10 seleções — vitórias
print("\nTOP 10 SELEÇÕES — VITÓRIAS TOTAIS")
print("-" * 40)
vitorias_casa  = jogos[jogos["home_team_win"] == 1]["home_team"].value_counts()
vitorias_fora  = jogos[jogos["away_team_win"] == 1]["away_team"].value_counts()
vitorias_total = (vitorias_casa
                  .add(vitorias_fora, fill_value=0)
                  .astype(int)
                  .sort_values(ascending=False))
print(vitorias_total.head(10).to_string())

#Empates por fase
print("\nEMPATES POR FASE DO TORNEIO")
print("-" * 40)
empates_fase = (jogos.groupby("stage_name")["draw"]
                .agg(["sum","count"])
                .rename(columns={"sum":"empates","count":"partidas"}))
empates_fase["pct_empate"] = (empates_fase["empates"] / empates_fase["partidas"] * 100).round(1)
print(empates_fase.sort_values("pct_empate", ascending=False).to_string())

#Vantagem do mandante
print("\nVANTAGEM DO MANDANTE POR FASE")
print("-" * 40)
grupos   = jogos[jogos["group_stage"] == 1]
knockout = jogos[jogos["knockout_stage"] == 1]
for nome, subset in [("Grupos", grupos), ("Knockout", knockout), ("Geral", jogos)]:
    v_casa = (subset["home_team_win"] == 1).mean() * 100
    empate = (subset["draw"] == 1).mean() * 100
    v_fora = (subset["away_team_win"] == 1).mean() * 100
    print(f"  {nome:<10} | Casa: {v_casa:.1f}%  Empate: {empate:.1f}%  Fora: {v_fora:.1f}%")

#Nulos
print("\nVALORES NULOS")
print("-" * 40)
nulos = jogos.isnull().sum()
nulos = nulos[nulos > 0]
print("  Nenhum valor nulo!" if nulos.empty else nulos.to_string())

"""
Em resumo, fizemos uma análise exploratória dos dados (EDA) para entender as principais características do dataset de jogos da Copa do Mundo FIFA entre 1930 e 2022.
Identificamos a distribuição de resultados, a média de gols por edição, as seleções com mais vitórias, a taxa de empates por fase do torneio e a vantagem do mandante. 
Além disso, verificamos a presença de valores nulos no dataset. 
Com esses insights em mãos, estamos prontos para avançar para a etapa de visualização dos dados, onde vamos criar gráficos para ilustrar essas descobertas de forma mais clara e impactante.
"""

#Terceiro, vamos criar visualizações para ilustrar os insights obtidos na análise exploratória dos dados (EDA). Faremos gráficos que destacam a distribuição de resultados, a média de gols por edição, a vantagem do mandante e o desempenho das seleções ao longo da história da Copa do Mundo FIFA.
media_por_ano = jogos.groupby("year")["total_gols"].mean()
#Gráfico 1: Visão geral
fig = plt.figure(figsize=(18, 11))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

# Distribuição de resultados
ax1 = fig.add_subplot(gs[0, 0])
labels_res = {"home team win": "Vitória\nCasa", "draw": "Empate", "away team win": "Vitória\nFora"}
cores_res  = {"home team win": AZUL, "draw": AMARELO, "away team win": VERDE}
vals_res   = dist_resultado.reindex(["home team win","draw","away team win"]).fillna(0)
bars = ax1.bar(
    [labels_res[k] for k in vals_res.index],
    vals_res.values,
    color=[cores_res[k] for k in vals_res.index],
    width=0.5, edgecolor="white", linewidth=1.5
)
for bar, v in zip(bars, vals_res.values):
    ax1.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 8,
             f"{int(v)}\n({v/len(jogos)*100:.1f}%)",
             ha="center", va="bottom", fontsize=9, fontweight="bold")
ax1.set_title("Distribuição de Resultados\n(964 partidas, 1930–2022)", fontweight="bold", fontsize=11)
ax1.set_ylabel("Nº de partidas")
ax1.set_ylim(0, vals_res.max() * 1.25)
ax1.set_facecolor("white")

# Média de gols por edição
ax2 = fig.add_subplot(gs[0, 1:])
ax2.plot(media_por_ano.index, media_por_ano.values,
         marker="o", color=AZUL, linewidth=2.2, markersize=6, zorder=3)
ax2.fill_between(media_por_ano.index, media_por_ano.values, alpha=0.12, color=AZUL)
ax2.axhline(media_por_ano.mean(), color=LARANJA, linestyle="--", alpha=0.8, linewidth=1.5,
            label=f"Média geral: {media_por_ano.mean():.2f} gols/partida")
pico_ano = media_por_ano.idxmax()
pico_val = media_por_ano.max()
ax2.annotate(f"Pico: {pico_val:.2f}\n({pico_ano})",
             xy=(pico_ano, pico_val),
             xytext=(pico_ano + 3, pico_val + 0.3),
             arrowprops=dict(arrowstyle="->", color=CINZA),
             fontsize=8.5, color=CINZA)
ax2.set_title("Média de Gols por Partida — por Edição", fontweight="bold", fontsize=11)
ax2.set_ylabel("Gols / partida")
ax2.set_xlabel("Ano")
ax2.legend(fontsize=9)
ax2.set_facecolor("white")

# Histograma de gols
ax3 = fig.add_subplot(gs[1, 0])
ax3.hist(jogos["total_gols"], bins=range(0, 14),
         color=AZUL, edgecolor="white", linewidth=1.2, alpha=0.85)
ax3.axvline(jogos["total_gols"].mean(), color=LARANJA, linestyle="--",
            linewidth=1.8, label=f"Média: {jogos['total_gols'].mean():.2f}")
ax3.axvline(jogos["total_gols"].median(), color=VERDE, linestyle=":",
            linewidth=1.8, label=f"Mediana: {int(jogos['total_gols'].median())}")
ax3.set_title("Distribuição: Total de Gols por Partida", fontweight="bold", fontsize=11)
ax3.set_xlabel("Gols na partida")
ax3.set_ylabel("Frequência")
ax3.legend(fontsize=9)
ax3.set_facecolor("white")

# Top 10 vitórias
ax4 = fig.add_subplot(gs[1, 1])
top10 = vitorias_total.head(10)
cores_top = [AZUL if t == "Brazil" else
             (VERDE if t in ["Argentina","Germany","West Germany"] else CINZA)
             for t in top10.index]
ax4.barh(top10.index[::-1], top10.values[::-1],
         color=cores_top[::-1], edgecolor="white", linewidth=1)
for i, v in enumerate(top10.values[::-1]):
    ax4.text(v + 0.3, i, str(int(v)), va="center", fontsize=9, fontweight="bold")
ax4.set_title("Top 10 Seleções — Vitórias Totais\n(1930–2022)", fontweight="bold", fontsize=11)
ax4.set_xlabel("Vitórias")
ax4.set_facecolor("white")

# Vantagem do mandante por fase
ax5 = fig.add_subplot(gs[1, 2])
fases_plot = ["Grupos\n(676)", "Knockout\n(246)", "Geral\n(964)"]
pct_casa = [(grupos["home_team_win"]==1).mean()*100,
            (knockout["home_team_win"]==1).mean()*100,
            (jogos["home_team_win"]==1).mean()*100]
pct_emp  = [(grupos["draw"]==1).mean()*100,
            (knockout["draw"]==1).mean()*100,
            (jogos["draw"]==1).mean()*100]
pct_fora = [(grupos["away_team_win"]==1).mean()*100,
            (knockout["away_team_win"]==1).mean()*100,
            (jogos["away_team_win"]==1).mean()*100]
x = np.arange(len(fases_plot))
w = 0.25
ax5.bar(x - w, pct_casa, w, label="Casa",   color=AZUL,    edgecolor="white")
ax5.bar(x,     pct_emp,  w, label="Empate", color=AMARELO, edgecolor="white")
ax5.bar(x + w, pct_fora, w, label="Fora",   color=VERDE,   edgecolor="white")
ax5.set_xticks(x)
ax5.set_xticklabels(fases_plot, fontsize=9)
ax5.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
ax5.set_title("Resultados por Fase\n(% das partidas)", fontweight="bold", fontsize=11)
ax5.legend(fontsize=9)
ax5.set_facecolor("white")

fig.suptitle("EDA — Copa do Mundo FIFA 1930–2022",
             fontsize=14, fontweight="bold", y=1.01)
path1 = os.path.join(OUTPUT, "eda_visao_geral.png")
plt.savefig(path1, dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print(f"\n[Gráfico 1 salvo]: {path1}")

#Gráfico 2: Gols
fig2, (ax6, ax7) = plt.subplots(1, 2, figsize=(14, 5))
fig2.patch.set_facecolor("white")

gols_fase = (jogos.groupby("stage_name")["total_gols"]
             .mean().sort_values(ascending=False))
ax6.barh(gols_fase.index[::-1], gols_fase.values[::-1], color=AZUL, edgecolor="white")
ax6.axvline(gols_fase.mean(), color=LARANJA, linestyle="--",
            linewidth=1.5, label=f"Média: {gols_fase.mean():.2f}")
ax6.set_title("Média de Gols por Fase do Torneio", fontweight="bold", fontsize=11)
ax6.set_xlabel("Média de gols / partida")
ax6.legend(fontsize=9)
ax6.set_facecolor("white")

partidas_ano = jogos.groupby("year").size()
ax7.bar(partidas_ano.index, partidas_ano.values,
        color=AZUL, edgecolor="white", linewidth=0.8, alpha=0.85, width=2.5)
for ano, n in partidas_ano.items():
    if n in [18, 52, 64]:
        ax7.text(ano, n + 0.8, str(n), ha="center", fontsize=8, color=CINZA)
ax7.set_title("Nº de Partidas por Edição", fontweight="bold", fontsize=11)
ax7.set_xlabel("Ano")
ax7.set_ylabel("Partidas")
ax7.set_facecolor("white")
ax7.spines["top"].set_visible(False)
ax7.spines["right"].set_visible(False)

fig2.suptitle("EDA — Análise de Gols e Partidas", fontsize=13, fontweight="bold")
fig2.tight_layout()
path2 = os.path.join(OUTPUT, "eda_gols_detalhado.png")
plt.savefig(path2, dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print(f"[Gráfico 2 salvo]: {path2}")

#Gráfico 3: Top 15 seleções
fig3, axes = plt.subplots(1, 3, figsize=(18, 6))
fig3.patch.set_facecolor("white")

top15_v = vitorias_total.head(15)
axes[0].barh(top15_v.index[::-1], top15_v.values[::-1], color=AZUL, edgecolor="white")
axes[0].set_title("Vitórias Totais", fontweight="bold")
axes[0].set_facecolor("white")
axes[0].spines["top"].set_visible(False)
axes[0].spines["right"].set_visible(False)

gols_casa = jogos.groupby("home_team")["home_score"].sum()
gols_fora = jogos.groupby("away_team")["away_score"].sum()
gols_time = gols_casa.add(gols_fora, fill_value=0).sort_values(ascending=False).head(15)
axes[1].barh(gols_time.index[::-1], gols_time.values[::-1], color=VERDE, edgecolor="white")
axes[1].set_title("Gols Marcados Totais", fontweight="bold")
axes[1].set_facecolor("white")
axes[1].spines["top"].set_visible(False)
axes[1].spines["right"].set_visible(False)

part_c = jogos.groupby("home_team").size()
part_f = jogos.groupby("away_team").size()
part_t = part_c.add(part_f, fill_value=0).sort_values(ascending=False).head(15)
axes[2].barh(part_t.index[::-1], part_t.values[::-1], color=LARANJA, edgecolor="white")
axes[2].set_title("Partidas Disputadas", fontweight="bold")
axes[2].set_facecolor("white")
axes[2].spines["top"].set_visible(False)
axes[2].spines["right"].set_visible(False)

fig3.suptitle("Top 15 Seleções — Histórico Completo (1930–2022)",
              fontsize=13, fontweight="bold")
fig3.tight_layout()
path3 = os.path.join(OUTPUT, "eda_selecoes.png")
plt.savefig(path3, dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print(f"[Gráfico 3 salvo]: {path3}")

#Quarto, vamos salvar o dataset processado em um novo arquivo CSV para uso nas próximas etapas do projeto, como modelagem preditiva e análise de desempenho.
jogos.to_csv(os.path.join(OUTPUT, "jogos_processado.csv"), index=False)
print(f"\nDataset processado salvo: jogos_processado.csv  ({jogos.shape})")

# Quinto, vamos gerar um relatório final com o resumo dos dados carregados e as principais estatísticas obtidas na análise exploratória, para documentar os insights e facilitar a comunicação dos resultados.
print("\n" + "="*55)
print("RESUMO DA EDA")
print("="*55)
print(f"  Partidas analisadas  : {len(jogos)}")
print(f"  Edições              : {jogos['year'].nunique()}  ({jogos['year'].min()}–{jogos['year'].max()})")
print(f"  Total de gols        : {jogos['total_gols'].sum()}")
print(f"  Média gols/partida   : {jogos['total_gols'].mean():.2f}")
print(f"  Vitória mandante     : {(jogos['home_team_win']==1).mean()*100:.1f}%")
print(f"  Empates              : {(jogos['draw']==1).mean()*100:.1f}%")
print(f"  Vitória visitante    : {(jogos['away_team_win']==1).mean()*100:.1f}%")
print(f"  Seleções únicas      : {pd.concat([jogos['home_team'], jogos['away_team']]).nunique()}")
print(f"  Valores nulos        : {jogos.isnull().sum().sum()}")
print(f"\n  Arquivos gerados em: {OUTPUT}")
print(f"    eda_visao_geral.png")
print(f"    eda_gols_detalhado.png")
print(f"    eda_selecoes.png")
print(f"    jogos_processado.csv")
print("\n[Etapa 2 concluída]")
