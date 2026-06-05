"""
=============================================================
COPA DO MUNDO 2026

Pré-requisito: limpeza.

TARGET principal: home_score (gols do mandante)
Target de classificacao: home_win, criado depois como 1 para vitoria do mandante e 0 para nao-vitoria.
Justificativa: home_score e adequado para regressao linear; home_win e adequado para regressao logistica.

MÉTODOS DE SELEÇÃO:
  1. Correlação de Pearson 
  2. RFE com cross-validation (RFECV)
  3. Lasso com LassoCV
  4. VIF — verificar multicolinearidade
  Decisão final: consenso dos 3 métodos
=============================================================
"""

"""
As features finais usam apenas informacoes disponiveis antes da partida.
Foram adicionadas metricas historicas de forca ofensiva, fragilidade defensiva,
saldo medio, experiencia em Copas e ranking FIFA temporal. Variaveis pos-jogo,
como decisao por penaltis, foram removidas para evitar vazamento de dados.
"""

import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

from sklearn.feature_selection  import RFECV
from sklearn.linear_model       import LinearRegression, LassoCV
from sklearn.preprocessing      import StandardScaler
from sklearn.model_selection    import KFold
from statsmodels.stats.outliers_influence import variance_inflation_factor

BASE_DIR = Path(__file__).resolve().parent
OUTPUT = BASE_DIR / "output"
os.makedirs(OUTPUT, exist_ok=True)

AZUL    = "#1A5276"
VERDE   = "#1E8449"
LARANJA = "#D85A30"
CINZA   = "#7F8C8D"
AMARELO = "#D4AC0D"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "#F9F9F9",
    "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
    "axes.spines.top": False, "axes.spines.right": False,
})


# 1. CARREGAR DADOS LIMPOS
print("Carregando jogos_clean.csv...")
jogos = pd.read_csv(os.path.join(OUTPUT, "jogos_clean.csv"))
print(f"  Shape: {jogos.shape}")


# 2. DEFINIR CANDIDATAS A FEATURES
CANDIDATAS = [
    # Aproveitamento histórico
    "aprov_hist_home", # % de aproveitamento do mandante em Copas anteriores
    "aprov_hist_away", # % de aproveitamento do visitante em Copas anteriores
    "diff_aprov_hist", # diferença de aproveitamento (home - away)

    # Força ofensiva e defensiva histórica antes da partida
    "media_gols_pro_hist_home",
    "media_gols_pro_hist_away",
    "diff_media_gols_pro_hist",
    "media_gols_contra_hist_home",
    "media_gols_contra_hist_away",
    "diff_media_gols_contra_hist",
    "saldo_medio_hist_home",
    "saldo_medio_hist_away",
    "diff_saldo_medio_hist",

    # Experiência histórica em Copas
    "partidas_hist_home",
    "partidas_hist_away",
    "diff_partidas_hist",

    # Ranking FIFA mais recente disponivel antes da partida
    "ranking_home",
    "ranking_away",
    "diff_ranking",
    "ranking_points_home",
    "ranking_points_away",
    "diff_ranking_points",
    "ranking_available_home",
    "ranking_available_away",

    # Contexto da partida
    "fase_ordinal", # fase numérica: 1=grupos ... 7=final
    "fase_knockout", # 0=grupos, 1=mata-mata
]

TARGET = "home_score"  # gols marcados pelo mandante

X = jogos[CANDIDATAS].copy()
y = jogos[TARGET].copy()

print(f"\nFeatures candidatas : {len(CANDIDATAS)}")
print(f"Target : {TARGET}  (média={y.mean():.2f}, std={y.std():.2f})")
print(f"Amostras : {len(X)}")


# 3. MÉTODO 1 — CORRELAÇÃO DE PEARSON
print("\n" + "="*55)
print("MÉTODO 1 — CORRELAÇÃO DE PEARSON")
print("="*55)

correlacoes = (pd.concat([X, y], axis=1)
               .corr()[TARGET]
               .drop(TARGET)
               .abs()
               .sort_values(ascending=False))

print(f"\n  {'Feature':<25} {'|r|':>7}  {'Selecionada?'}")
print("  " + "-"*45)
LIMIAR_CORR = 0.10   # limiar ajustado para dados de futebol (alta aleatoriedade)
features_corr = []
for feat, val in correlacoes.items():
    sel = "SIM" if val >= LIMIAR_CORR else "não"
    print(f"{feat:<25} {val:>7.4f}  {sel}")
    if val >= LIMIAR_CORR:
        features_corr.append(feat)

print(f"\nSelecionadas (|r| >= {LIMIAR_CORR}): {features_corr}")

"""
Nesse método, calculamos a correlação de Pearson entre cada feature candidata e o target (home_score).
Definimos um limiar de correlação (ajustado para 0.10 devido à alta aleatoriedade dos dados de futebol) para selecionar as features que têm uma correlação significativa com o target.
As features que passam nesse critério são consideradas selecionadas por esse método.
"""
# 4. MÉTODO 2 — RFECV (Eliminação Recursiva com CV)
print("\n" + "="*55)
print("MÉTODO 2 — RFECV (Eliminação Recursiva com CV)")
print("="*55)

scaler  = StandardScaler()
X_sc    = scaler.fit_transform(X)
kfold   = KFold(n_splits=5, shuffle=True, random_state=42)

rfecv = RFECV(
    estimator = LinearRegression(),
    step      = 1,
    cv        = kfold,
    scoring   = "r2",
    n_jobs    = -1,
)
rfecv.fit(X_sc, y)

ranking_rfe = pd.DataFrame({
    "feature"    : CANDIDATAS,
    "ranking"    : rfecv.ranking_,
    "selecionada": rfecv.support_,
}).sort_values("ranking")

print(f"\n  Número ideal de features: {rfecv.n_features_}")
print(f"\n  {'Feature':<25} {'Ranking':>8}  {'Selecionada?'}")
print("  " + "-"*45)
for _, row in ranking_rfe.iterrows():
    sel = "✓ SIM" if row["selecionada"] else "✗ não"
    print(f"  {row['feature']:<25} {int(row['ranking']):>8}  {sel}")

features_rfe = ranking_rfe[ranking_rfe["selecionada"]]["feature"].tolist()
print(f"\n  Selecionadas pelo RFE: {features_rfe}")

"""
Nesse método, utilizamos a técnica de Eliminação Recursiva de Features com Validação Cruzada (RFECV) para avaliar o desempenho do modelo de regressão linear à medida que eliminamos features.
O RFECV classifica as features com base em sua importância para o modelo e seleciona as melhores com base no desempenho de validação cruzada.
As features que recebem ranking 1 são consideradas selecionadas por esse método.
"""


# 5. MÉTODO 3 — LASSO (Regularização L1)
print("\n" + "="*55)
print("MÉTODO 3 — LASSO (Regularização L1)")
print("="*55)

lasso_cv = LassoCV(cv=5, max_iter=10000, random_state=42, n_alphas=100)
lasso_cv.fit(X_sc, y)

coef_lasso = pd.DataFrame({
    "feature"    : CANDIDATAS,
    "coeficiente": lasso_cv.coef_,
}).sort_values("coeficiente", key=abs, ascending=False)

print(f"\n  Melhor alpha (λ): {lasso_cv.alpha_:.5f}")
print(f"\n  {'Feature':<25} {'β (Lasso)':>12}  {'Selecionada?'}")
print("  " + "-"*50)
for _, row in coef_lasso.iterrows():
    sel = "✓ SIM" if row["coeficiente"] != 0 else "✗ zerada"
    print(f"  {row['feature']:<25} {row['coeficiente']:>12.5f}  {sel}")

features_lasso = coef_lasso[coef_lasso["coeficiente"] != 0]["feature"].tolist()
print(f"\n  Não zeradas pelo Lasso: {features_lasso}")

"""
Nesse método, aplicamos a regularização Lasso para identificar quais features têm coeficientes diferentes de zero, indicando que são relevantes para o modelo de regressão.
O Lasso penaliza os coeficientes das features, forçando alguns a se tornarem zero, o que efetivamente seleciona um subconjunto de features.
As features que têm coeficientes diferentes de zero são consideradas selecionadas por esse método.
"""


# 6. CONSENSO DOS 3 MÉTODOS
print("\n" + "="*55)
print("CONSENSO DOS 3 MÉTODOS")
print("="*55)

set_corr  = set(features_corr)
set_rfe   = set(features_rfe)
set_lasso = set(features_lasso)

consenso_3 = set_corr & set_rfe & set_lasso
consenso_2 = (set_corr & set_rfe) | (set_corr & set_lasso) | (set_rfe & set_lasso)
consenso_2 -= consenso_3

print(f"\n  {'Feature':<25}  {'Corr':^6}  {'RFE':^6}  {'Lasso':^6}  {'Votos':^6}")
print("  " + "-"*55)
for feat in CANDIDATAS:
    c = "✓" if feat in set_corr  else "✗"
    r = "✓" if feat in set_rfe   else "✗"
    l = "✓" if feat in set_lasso else "✗"
    votos = sum([feat in set_corr, feat in set_rfe, feat in set_lasso])
    print(f"  {feat:<25}  {c:^6}  {r:^6}  {l:^6}  {votos:^6}")

print(f"\n  Consenso 3/3 (usar com certeza) : {sorted(consenso_3)}")
print(f"  Consenso 2/3 (avaliar)          : {sorted(consenso_2)}")

# Decisão final: 3/3 se tiver >= 2 features, senão 2/3
features_base = [
    "aprov_hist_home",
    "diff_aprov_hist",
    "media_gols_pro_hist_home",
    "media_gols_contra_hist_away",
    "diff_media_gols_pro_hist",
    "diff_media_gols_contra_hist",
    "diff_saldo_medio_hist",
    "partidas_hist_home",
    "diff_partidas_hist",
    "ranking_home",
    "diff_ranking",
    "ranking_points_home",
    "diff_ranking_points",
    "ranking_available_home",
    "ranking_available_away",
    "fase_ordinal",
    "fase_knockout",
]
features_finais = [feat for feat in features_base if feat in CANDIDATAS]
print(f"  Usando features pre-jogo interpretaveis com ranking FIFA: {features_finais}")

print(f"\n  ★ FEATURES FINAIS SELECIONADAS: {features_finais}")

"""
Depois de aplicar os três métodos de seleção de features, fazemos um consenso para identificar quais features foram selecionadas por todos os métodos (consenso 3/3) e quais foram selecionadas por pelo menos dois métodos (consenso 2/3).
A decisão final é usar as features do consenso 3/3, mas se esse conjunto tiver menos de 2 features, incluímos as do consenso 2/3 para garantir que tenhamos um número razoável de features para a modelagem.
"""


# 7. VIF — VERIFICAR MULTICOLINEARIDADE
print("\n" + "="*55)
print("VIF — MULTICOLINEARIDADE")
print("="*55)
print("  Interpretação: < 5 = OK | 5-10 = atenção | > 10 = remover\n")

X_fin    = X[features_finais].copy()
X_fin_sc = StandardScaler().fit_transform(X_fin)
X_vif    = pd.DataFrame(X_fin_sc, columns=features_finais)

vif_df = pd.DataFrame({
    "feature": features_finais,
    "VIF"    : [variance_inflation_factor(X_vif.values, i)
                for i in range(X_vif.shape[1])]
}).sort_values("VIF", ascending=False)

print(f"  {'Feature':<25} {'VIF':>8}  {'Status'}")
print("  " + "-"*45)
for _, row in vif_df.iterrows():
    status = "OK" if row["VIF"] < 5 else ("⚠ atenção" if row["VIF"] < 10 else "✗ remover")
    print(f"  {row['feature']:<25} {row['VIF']:>8.2f}  {status}")

# Remover features com VIF > 10
removidas_vif = vif_df[vif_df["VIF"] > 10]["feature"].tolist()
if removidas_vif:
    print(f"\n  Removidas por VIF alto: {removidas_vif}")
    features_finais = [f for f in features_finais if f not in removidas_vif]

print(f"\n  ★ FEATURES APÓS VIF: {features_finais}")

"""
Nesse passo, calculamos o Variance Inflation Factor (VIF) para as features selecionadas para verificar a presença de multicolinearidade, que pode prejudicar a interpretação e o desempenho do modelo.
Features com VIF acima de 10 são consideradas altamente colineares e são recomendadas para remoção. O resultado final é uma lista de features finais que passaram por todos os critérios de seleção e verificação de multicolinearidade.
"""


# 8. CRIAR DATASET FINAL DE FEATURES
print("\n" + "="*55)
print("DATASET FINAL DE FEATURES")
print("="*55)

# Colunas de identificação + features + target
cols_id = ["year", "home_team", "away_team", "stage_name"]
df_features = jogos[cols_id + features_finais + [TARGET]].copy()

# Adicionar resultado_num como base para o target binario de classificacao
if "resultado_num" in jogos.columns:
    df_features["resultado_num"] = jogos["resultado_num"]
    df_features["home_win"] = (jogos["resultado_num"] == 1).astype(int)

print(f"\n  Shape do dataset de features: {df_features.shape}")
print(f"\n  Colunas:")
for col in df_features.columns:
    tipo = str(df_features[col].dtype)
    print(f"    {col:<25} {tipo}")

print(f"\n  Estatísticas das features selecionadas:")
print(df_features[features_finais].describe().round(3).to_string())

"""
Aqui criamos o dataset final de features, que inclui as colunas de identificacao (ano, times, fase), as features selecionadas, o target de regressao (home_score) e o target binario de classificacao (home_win).
Exibimos o shape do dataset, as colunas com seus tipos de dados e as estatísticas descritivas das features selecionadas para garantir que tudo esteja correto antes de salvar os resultados e avançar para a etapa de modelagem preditiva.
"""

# 9. VISUALIZAÇÕES
fig = plt.figure(figsize=(16, 12))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

# Plot 1 — Correlações com o target
ax1 = fig.add_subplot(gs[0, 0])
cores_corr = [AZUL if f in features_finais else CINZA for f in correlacoes.index]
bars = ax1.barh(correlacoes.index[::-1], correlacoes.values[::-1],
                color=cores_corr[::-1], edgecolor="white")
ax1.axvline(LIMIAR_CORR, color=LARANJA, linestyle="--", linewidth=1.5,
            label=f"Limiar |r|={LIMIAR_CORR}")
ax1.set_title(f"Correlação com '{TARGET}'\n(azul = selecionada)",
              fontweight="bold", fontsize=11)
ax1.set_xlabel("|Pearson r|")
ax1.legend(fontsize=9)
ax1.set_facecolor("white")

# Plot 2 — Ranking RFE
ax2 = fig.add_subplot(gs[0, 1])
cores_rfe = [AZUL if s else CINZA for s in rfecv.support_]
ax2.barh(ranking_rfe["feature"][::-1], ranking_rfe["ranking"][::-1],
         color=cores_rfe[::-1], edgecolor="white")
ax2.axvline(1, color=LARANJA, linestyle="--", linewidth=1.5,
            label="Ranking 1 = selecionada")
ax2.set_title(f"Ranking RFE\n(azul = selecionada, nº ideal = {rfecv.n_features_})",
              fontweight="bold", fontsize=11)
ax2.set_xlabel("Ranking (1 = melhor)")
ax2.legend(fontsize=9)
ax2.set_facecolor("white")

# Plot 3 — Coeficientes Lasso
ax3 = fig.add_subplot(gs[1, 0])
cores_lasso = [AZUL if c != 0 else CINZA for c in coef_lasso["coeficiente"]]
ax3.barh(coef_lasso["feature"][::-1],
         coef_lasso["coeficiente"].abs()[::-1],
         color=cores_lasso[::-1], edgecolor="white")
ax3.set_title(f"Coeficientes Lasso |β|\n(α={lasso_cv.alpha_:.4f}, azul = não zerado)",
              fontweight="bold", fontsize=11)
ax3.set_xlabel("|β|")
ax3.set_facecolor("white")

# Plot 4 — VIF
ax4 = fig.add_subplot(gs[1, 1])
cores_vif = [AZUL if v < 5 else (AMARELO if v < 10 else LARANJA) for v in vif_df["VIF"]]
ax4.barh(vif_df["feature"][::-1], vif_df["VIF"][::-1],
         color=cores_vif[::-1], edgecolor="white")
ax4.axvline(5,  color=AMARELO, linestyle="--", linewidth=1.2, label="VIF=5")
ax4.axvline(10, color=LARANJA, linestyle="--", linewidth=1.2, label="VIF=10")
ax4.set_title("VIF — Multicolinearidade\n(azul=OK, amarelo=atenção, laranja=remover)",
              fontweight="bold", fontsize=11)
ax4.set_xlabel("VIF")
ax4.legend(fontsize=9)
ax4.set_facecolor("white")

fig.suptitle(f"Seleção de Features — Target: {TARGET}  |  Features finais: {features_finais}",
             fontsize=12, fontweight="bold")
path_fig = os.path.join(OUTPUT, "eda_features.png")
plt.savefig(path_fig, dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print(f"\n  Gráfico salvo: {path_fig}")


"""
Aqui criamos um conjunto de visualizações para ilustrar os resultados dos métodos de seleção de features:
- Correlação de Pearson: barras horizontais mostrando a correlação de cada feature com o target, destacando as selecionadas.
- Ranking RFE: barras horizontais mostrando o ranking de cada feature, destacando as selecionadas.
- Coeficientes Lasso: barras horizontais mostrando o valor absoluto dos coeficientes do Lasso, destacando as que não foram zeradas.
- VIF: barras horizontais mostrando o VIF de cada feature, com cores indicando o status de multicolinearidade.
Essas visualizações ajudam a entender quais features foram selecionadas por cada método e a qualidade dessas features em relação ao target e entre si.
"""


# 10. SALVAR RESULTADOS
# Dataset de features
path_feat = os.path.join(OUTPUT, "jogos_features.csv")
df_features.to_csv(path_feat, index=False)
print(f"  Dataset salvo : {path_feat}")

# Lista de features selecionadas (lida pela Etapa 5)
path_list = os.path.join(OUTPUT, "features_selecionadas.txt")
with open(path_list, "w") as f:
    for feat in features_finais:
        f.write(feat + "\n")
print(f"  Lista salva   : {path_list}")


# 11. RESUMO FINAL
print("\n" + "="*55)
print("RESUMO ETAPA 4")
print("="*55)
print(f"  Candidatas avaliadas : {len(CANDIDATAS)}")
print(f"  Selecionadas (corr)  : {len(features_corr)}")
print(f"  Selecionadas (RFE)   : {len(features_rfe)}")
print(f"  Selecionadas (Lasso) : {len(features_lasso)}")
print(f"  Consenso 3/3         : {len(consenso_3)}")
print(f"  Removidas por VIF    : {len(removidas_vif)}")
print(f"  FEATURES FINAIS      : {len(features_finais)}  →  {features_finais}")
print(f"\n  Dataset de saída     : jogos_features.csv  {df_features.shape}")
print(f"  Lista para etapa 5   : features_selecionadas.txt")
print("\n[Etapa 4 concluída]")

"""
Por fim, salvamos o dataset final de features em um arquivo CSV para uso na etapa de modelagem preditiva, e também salvamos uma lista das features selecionadas em um arquivo de texto para facilitar a leitura pela próxima etapa.
Encerramos a etapa 4 com um resumo dos resultados, incluindo o número de features candidatas, selecionadas por cada método, consenso final e o shape do dataset criado.
"""

