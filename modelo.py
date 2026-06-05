"""
=============================================================
COPA DO MUNDO 2026
Etapa 5: Treinamento, Avaliação e Predição para 2026

Entrada : output/jogos_features.csv  (histórico 1930-2022)
Tabela : dataset/tabela_oficial_copa_2026_grupos.csv  (jogos da Copa 2026)
Saída   : output/modelo_copa.pkl
          output/metricas.json
          output/predicoes_2026.csv
          output/avaliacao_modelo.png
          output/predicoes_2026.png

FEATURES FINAIS (definidas na Etapa 4):
  - aproveitamento histórico
  - médias históricas de gols feitos/sofridos
  - saldo médio histórico
  - experiência em partidas de Copa
  - contexto da fase

Observação: decidido_penaltis foi removida por ser variável pós-jogo.

TARGET : home_score (gols do mandante)
=============================================================
"""

import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import json, joblib, os, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

from sklearn.linear_model    import LinearRegression
from sklearn.pipeline        import Pipeline
from sklearn.preprocessing   import StandardScaler
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.metrics         import mean_squared_error, mean_absolute_error, r2_score

# ── Caminhos ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
OUTPUT        = BASE_DIR / "output"
PLANILHA_2026 = BASE_DIR / "tabela-copa-do-mundo-fifa-2026.xlsx"
TABELA_OFICIAL_2026 = BASE_DIR / "dataset" / "tabela_oficial_copa_2026_grupos.csv"
os.makedirs(OUTPUT, exist_ok=True)

# ── Paleta ────────────────────────────────────────────────
AZUL    = "#1A5276"
VERDE   = "#1E8449"
LARANJA = "#D85A30"
CINZA   = "#7F8C8D"
AMARELO = "#D4AC0D"
ROXO    = "#6C3483"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "#F9F9F9",
    "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
    "axes.spines.top": False, "axes.spines.right": False,
})


# ══════════════════════════════════════════════════════════
# 1. CARREGAR DADOS HISTÓRICOS E FEATURES
# ══════════════════════════════════════════════════════════
print("Carregando dados históricos (jogos_features.csv)...")
df = pd.read_csv(os.path.join(OUTPUT, "jogos_features.csv"))

feat_path = os.path.join(OUTPUT, "features_selecionadas.txt")
with open(feat_path) as f:
    FEATURES = [l.strip() for l in f if l.strip()]

TARGET = "home_score"

print(f"  Shape    : {df.shape}")
print(f"  Features : {FEATURES}")
print(f"  Target   : {TARGET}  (média={df[TARGET].mean():.2f}, std={df[TARGET].std():.2f})")
print(f"  Amostras : {len(df)}")

X = df[FEATURES]
y = df[TARGET]

"""
NOTA SOBRE aprov_hist_home / diff_aprov_hist
---------------------------------------------
Essas features são calculadas em limpeza.py pela função calcular_historico(),
que percorre TODOS os jogos históricos (1930-2022) em ordem cronológica.
Para cada partida, o aproveitamento registrado é o acumulado das edições
ANTERIORES àquele jogo — evitando vazamento de dados (data leakage).

Fórmula:  aprov = pontos_acumulados / (partidas_acumuladas × 3)
  - Vitória  = 3 pts  |  Empate = 1 pt  |  Derrota = 0 pts
  - Seleções sem histórico recebem 0.5 (prior neutro)

Portanto, aprov_hist_home mede o desempenho geral da seleção em todas
as Copas anteriores à partida sendo avaliada — não apenas a última edição.
"""


# ══════════════════════════════════════════════════════════
# 2. DIVISÃO TREINO / TESTE  (80 % / 20 %)
# ══════════════════════════════════════════════════════════
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
print(f"\n  Treino : {len(X_train)} amostras")
print(f"  Teste  : {len(X_test)}  amostras")


# ══════════════════════════════════════════════════════════
# 3. PIPELINE: NORMALIZAÇÃO + REGRESSÃO LINEAR
# ══════════════════════════════════════════════════════════
# O StandardScaler é ajustado SOMENTE nos dados de treino
# em cada fold → evita data leakage na cross-validation
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("modelo", LinearRegression()),
])


# ══════════════════════════════════════════════════════════
# 4. CROSS-VALIDATION (5 folds)
# ══════════════════════════════════════════════════════════
print("\n" + "="*55)
print("CROSS-VALIDATION — 5 FOLDS")
print("="*55)

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

cv_results = cross_validate(
    pipe, X, y,
    cv=kfold,
    scoring=["r2",
             "neg_root_mean_squared_error",
             "neg_mean_absolute_error"],
    return_train_score=True,
)

r2_treino = cv_results["train_r2"]
r2_teste  = cv_results["test_r2"]
rmse_cv   = -cv_results["test_neg_root_mean_squared_error"]
mae_cv    = -cv_results["test_neg_mean_absolute_error"]

print(f"\n  {'Fold':<6} {'R² Treino':>10} {'R² Teste':>10} {'RMSE':>8} {'MAE':>8}")
print("  " + "-"*46)
for i in range(5):
    print(f"  {i+1:<6} {r2_treino[i]:>10.4f} {r2_teste[i]:>10.4f} "
          f"{rmse_cv[i]:>8.4f} {mae_cv[i]:>8.4f}")
print("  " + "-"*46)
print(f"  {'Média':<6} {r2_treino.mean():>10.4f} {r2_teste.mean():>10.4f} "
      f"{rmse_cv.mean():>8.4f} {mae_cv.mean():>8.4f}")
print(f"  {'Desvio':<6} {r2_treino.std():>10.4f} {r2_teste.std():>10.4f} "
      f"{rmse_cv.std():>8.4f} {mae_cv.std():>8.4f}")

gap = r2_treino.mean() - r2_teste.mean()
print(f"\n  Gap R² (treino − teste): {gap:.4f}")
if gap > 0.15:
    print("  [AVISO] Possível overfitting")
else:
    print("  [OK] Modelo generaliza bem — gap dentro do esperado")


# ══════════════════════════════════════════════════════════
# 5. TREINAR MODELO FINAL  (80 % dos dados)
# ══════════════════════════════════════════════════════════
print("\n" + "="*55)
print("MODELO FINAL — AVALIAÇÃO NO CONJUNTO DE TESTE")
print("="*55)

pipe.fit(X_train, y_train)

y_pred_teste  = pipe.predict(X_test)
y_pred_treino = pipe.predict(X_train)

r2_final   = r2_score(y_test, y_pred_teste)
rmse_final = np.sqrt(mean_squared_error(y_test, y_pred_teste))
mae_final  = mean_absolute_error(y_test, y_pred_teste)
residuos   = y_test.values - y_pred_teste

print(f"\n  R²   = {r2_final:.4f}  →  {r2_final*100:.1f}% da variância explicada")
print(f"  RMSE = {rmse_final:.4f}  →  erro médio de ±{rmse_final:.2f} gols")
print(f"  MAE  = {mae_final:.4f}  →  desvio absoluto médio de {mae_final:.2f} gols")


# ══════════════════════════════════════════════════════════
# 6. EQUAÇÃO DA REGRESSÃO LINEAR
# ══════════════════════════════════════════════════════════
print("\n" + "="*55)
print("EQUAÇÃO DA REGRESSÃO LINEAR")
print("="*55)

coef       = pipe.named_steps["modelo"].coef_
intercepto = pipe.named_steps["modelo"].intercept_

print(f"\n  home_score = {intercepto:.4f}")
for feat, beta in zip(FEATURES, coef):
    sinal = "+" if beta >= 0 else "-"
    print(f"             {sinal} {abs(beta):.4f} × {feat}")

coef_df = pd.DataFrame({"feature": FEATURES, "beta": coef}).sort_values("beta", key=abs, ascending=False)
print(f"\n  {'Feature':<25} {'β':>10}  Interpretação")
print("  " + "-"*55)
for _, row in coef_df.iterrows():
    direcao = "↑ gols" if row["beta"] > 0 else "↓ gols"
    print(f"  {row['feature']:<25} {row['beta']:>10.4f}  {direcao}")


# ══════════════════════════════════════════════════════════
# 7. ANÁLISE DOS RESÍDUOS
# ══════════════════════════════════════════════════════════
print("\n" + "="*55)
print("ANÁLISE DOS RESÍDUOS")
print("="*55)
print(f"  Média dos resíduos : {residuos.mean():.4f}  (ideal: ≈ 0)")
print(f"  Desvio dos resíduos: {residuos.std():.4f}")
print(f"  Resíduo mín/máx    : {residuos.min():.2f} / {residuos.max():.2f}")


# ══════════════════════════════════════════════════════════
# 8. CONSTRUIR APROVEITAMENTO HISTÓRICO ACUMULADO 1930-2022
#    (base para calcular aprov das seleções na Copa 2026)
# ══════════════════════════════════════════════════════════
print("\n" + "="*55)
print("CALCULANDO APROVEITAMENTO HISTÓRICO 1930-2022")
print("="*55)

"""
Para prever jogos da Copa 2026, precisamos do aproveitamento histórico
de cada seleção acumulado ao longo de TODAS as Copas anteriores (1930-2022).
Reconstruímos esse cálculo aqui a partir do dataset histórico completo
(jogos_clean.csv), que contém resultado_num e já tem os nomes padronizados.
"""

historico_base = pd.read_csv(os.path.join(OUTPUT, "jogos_clean.csv"))
historico_base = historico_base.sort_values(["year"]).reset_index(drop=True)

# Acumular pontos, partidas e gols para cada seleção ao longo de todo o histórico
hist_acumulado = {}  # {team: pontos, partidas, gols_pro, gols_contra}

for _, row in historico_base.iterrows():
    ta  = row["home_team"]
    tb  = row["away_team"]
    res = row["resultado_num"]

    ha = hist_acumulado.get(ta, {"pontos": 0, "partidas": 0, "gols_pro": 0, "gols_contra": 0})
    hb = hist_acumulado.get(tb, {"pontos": 0, "partidas": 0, "gols_pro": 0, "gols_contra": 0})

    pts_a = 3 if res == 1 else (1 if res == 0 else 0)
    pts_b = 3 if res == -1 else (1 if res == 0 else 0)
    gols_a = int(row["home_score"])
    gols_b = int(row["away_score"])

    hist_acumulado[ta] = {
        "pontos": ha["pontos"] + pts_a,
        "partidas": ha["partidas"] + 1,
        "gols_pro": ha["gols_pro"] + gols_a,
        "gols_contra": ha["gols_contra"] + gols_b,
    }
    hist_acumulado[tb] = {
        "pontos": hb["pontos"] + pts_b,
        "partidas": hb["partidas"] + 1,
        "gols_pro": hb["gols_pro"] + gols_b,
        "gols_contra": hb["gols_contra"] + gols_a,
    }

def aprov_selecao(team):
    """Aproveitamento acumulado de 1930-2022. Seleção sem histórico → 0.5."""
    h = hist_acumulado.get(team, None)
    if h is None or h["partidas"] == 0:
        return 0.5
    return h["pontos"] / (h["partidas"] * 3)

def historico_selecao(team):
    """Features históricas acumuladas até 2022. Seleção sem histórico recebe prior neutro."""
    h = hist_acumulado.get(team, None)
    if h is None or h["partidas"] == 0:
        return {
            "aprov": 0.5,
            "media_gols_pro": 0.0,
            "media_gols_contra": 0.0,
            "saldo_medio": 0.0,
            "partidas": 0,
        }

    partidas = h["partidas"]
    return {
        "aprov": h["pontos"] / (partidas * 3),
        "media_gols_pro": h["gols_pro"] / partidas,
        "media_gols_contra": h["gols_contra"] / partidas,
        "saldo_medio": (h["gols_pro"] - h["gols_contra"]) / partidas,
        "partidas": partidas,
    }

# Relatório: top 15 aproveitamentos históricos
ranking_hist = (
    pd.DataFrame([
        {"selecao": k, "partidas": v["partidas"],
         "pontos": v["pontos"],
         "aprov": v["pontos"] / (v["partidas"] * 3)}
        for k, v in hist_acumulado.items()
    ])
    .sort_values("aprov", ascending=False)
)
print("\n  Top 15 aproveitamentos históricos (1930-2022):")
print(f"  {'Seleção':<25} {'Partidas':>9} {'Pontos':>7} {'Aprov':>7}")
print("  " + "-"*50)
for _, r in ranking_hist.head(15).iterrows():
    print(f"  {r['selecao']:<25} {r['partidas']:>9} {r['pontos']:>7} {r['aprov']:>7.3f}")


# ══════════════════════════════════════════════════════════
# 9. EXTRAIR JOGOS DA COPA 2026 DA PLANILHA
# ══════════════════════════════════════════════════════════
print("\n" + "="*55)
print("EXTRAINDO JOGOS DA COPA 2026")
print("="*55)

# Mapeamento: nomes em PT (planilha) → nomes em EN (histórico)
NOME_PT_PARA_EN = {
    "México"           : "Mexico",
    "Coreia do Sul"    : "South Korea",
    "África do Sul"    : "South Africa",
    "República Tcheca" : "Czech Republic",
    "Canadá"           : "Canada",
    "Catar"            : "Qatar",
    "Bósnia"           : "Bosnia",
    "Suíça"            : "Switzerland",
    "Brasil"           : "Brazil",
    "Marrocos"         : "Morocco",
    "Haiti"            : "Haiti",
    "Escócia"          : "Scotland",
    "Estados Unidos"   : "USA",
    "Paraguai"         : "Paraguay",
    "Austrália"        : "Australia",
    "Turquia"          : "Turkey",
    "Alemanha"         : "Germany",
    "Costa do Marfim"  : "Cote d'Ivoire",
    "Curaçao"          : "Curacao",
    "Equador"          : "Ecuador",
    "Holanda"          : "Netherlands",
    "Japão"            : "Japan",
    "Suécia"           : "Sweden",
    "Tunísia"          : "Tunisia",
    "Bélgica"          : "Belgium",
    "Egito"            : "Egypt",
    "Irã"              : "Iran",
    "Nova Zelândia"    : "New Zealand",
    "Espanha"          : "Spain",
    "Arábia Saudita"   : "Saudi Arabia",
    "Cabo Verde"       : "Cape Verde",
    "Uruguai"          : "Uruguay",
    "França"           : "France",
    "Senegal"          : "Senegal",
    "Iraque"           : "Iraq",
    "Noruega"          : "Norway",
    "Argentina"        : "Argentina",
    "Argélia"          : "Algeria",
    "Áustria"          : "Austria",
    "Jordânia"         : "Jordan",
    "Portugal"         : "Portugal",
    "Colômbia"         : "Colombia",
    "RD Congo"         : "DR Congo",
    "Uzbequistão"      : "Uzbekistan",
    "Inglaterra"       : "England",
    "Croácia"          : "Croatia",
    "Gana"             : "Ghana",
    "Panamá"           : "Panama",
}

if TABELA_OFICIAL_2026.exists():
    jogos_2026 = pd.read_csv(TABELA_OFICIAL_2026)
    print(f"  Fonte usada: {TABELA_OFICIAL_2026}")
else:
    xl = pd.read_excel(PLANILHA_2026, sheet_name=None)
    grupos_sheets = [s for s in xl.keys() if s.startswith("Gr-")]

    jogos_2026 = []
    for sheet in grupos_sheets:
        grupo = sheet.replace("Gr-", "Grupo ")
        df_g  = xl[sheet]
        # Linhas 3-8 (0-indexed) contêm os 6 jogos do grupo
        # col 0 = Rodada, col 1 = Mandante, col 7 = Visitante
        for i in range(3, 9):
            rodada    = df_g.iloc[i, 0]
            mandante  = df_g.iloc[i, 1]
            visitante = df_g.iloc[i, 7]
            if pd.notna(mandante) and pd.notna(visitante):
                jogos_2026.append({
                    "grupo"     : grupo,
                    "rodada"    : rodada,
                    "home_pt"   : mandante,
                    "away_pt"   : visitante,
                    "home_team" : NOME_PT_PARA_EN.get(mandante, mandante),
                    "away_team" : NOME_PT_PARA_EN.get(visitante, visitante),
                })

    jogos_2026 = pd.DataFrame(jogos_2026)
print(f"  Total de jogos extraídos: {len(jogos_2026)}")

# Verificar seleções sem histórico
sem_hist = set()
for t in pd.concat([jogos_2026["home_team"], jogos_2026["away_team"]]).unique():
    if t not in hist_acumulado:
        sem_hist.add(t)
if sem_hist:
    print(f"\n  Seleções sem histórico em Copas (prior=0.5): {sorted(sem_hist)}")
else:
    print("  Todas as seleções têm histórico registrado.")


# ══════════════════════════════════════════════════════════
# 10. CALCULAR FEATURES PARA OS JOGOS DE 2026
# ══════════════════════════════════════════════════════════
jogos_2026["aprov_hist_home"] = jogos_2026["home_team"].apply(aprov_selecao)
jogos_2026["aprov_hist_away"] = jogos_2026["away_team"].apply(aprov_selecao)
jogos_2026["diff_aprov_hist"] = jogos_2026["aprov_hist_home"] - jogos_2026["aprov_hist_away"]

hist_home_2026 = jogos_2026["home_team"].apply(historico_selecao)
hist_away_2026 = jogos_2026["away_team"].apply(historico_selecao)

jogos_2026["media_gols_pro_hist_home"] = hist_home_2026.apply(lambda h: h["media_gols_pro"])
jogos_2026["media_gols_pro_hist_away"] = hist_away_2026.apply(lambda h: h["media_gols_pro"])
jogos_2026["diff_media_gols_pro_hist"] = jogos_2026["media_gols_pro_hist_home"] - jogos_2026["media_gols_pro_hist_away"]

jogos_2026["media_gols_contra_hist_home"] = hist_home_2026.apply(lambda h: h["media_gols_contra"])
jogos_2026["media_gols_contra_hist_away"] = hist_away_2026.apply(lambda h: h["media_gols_contra"])
jogos_2026["diff_media_gols_contra_hist"] = jogos_2026["media_gols_contra_hist_home"] - jogos_2026["media_gols_contra_hist_away"]

jogos_2026["saldo_medio_hist_home"] = hist_home_2026.apply(lambda h: h["saldo_medio"])
jogos_2026["saldo_medio_hist_away"] = hist_away_2026.apply(lambda h: h["saldo_medio"])
jogos_2026["diff_saldo_medio_hist"] = jogos_2026["saldo_medio_hist_home"] - jogos_2026["saldo_medio_hist_away"]

jogos_2026["partidas_hist_home"] = hist_home_2026.apply(lambda h: h["partidas"])
jogos_2026["partidas_hist_away"] = hist_away_2026.apply(lambda h: h["partidas"])
jogos_2026["diff_partidas_hist"] = jogos_2026["partidas_hist_home"] - jogos_2026["partidas_hist_away"]

# A tabela oficial usada aqui contém apenas fase de grupos.
jogos_2026["fase_ordinal"] = 1
jogos_2026["fase_knockout"] = 0


# ══════════════════════════════════════════════════════════
# 11. TREINAR MODELO FINAL COM 100 % DOS DADOS HISTÓRICOS
# ══════════════════════════════════════════════════════════
print("\nTreinando modelo final com 100% dos dados históricos (1930-2022)...")
pipe.fit(X, y)


# ══════════════════════════════════════════════════════════
# 12. PREDIÇÃO — JOGOS DA COPA 2026
# ══════════════════════════════════════════════════════════
print("\n" + "="*55)
print("PREDIÇÕES — COPA DO MUNDO 2026")
print("="*55)

X_2026 = jogos_2026[FEATURES]

# Predições para home e away (modelo foi treinado para home_score;
# para away_score, invertemos mandante/visitante)
jogos_2026["pred_home"] = pipe.predict(X_2026).clip(0)

# Predição do away: montar features espelhadas
X_2026_inv = pd.DataFrame({
    f: jogos_2026["aprov_hist_away"] if f == "aprov_hist_home"
       else (-jogos_2026["diff_aprov_hist"] if f == "diff_aprov_hist"
             else jogos_2026[f])
    for f in FEATURES
})
jogos_2026["pred_away"] = pipe.predict(X_2026_inv).clip(0)

# Resultado previsto
def resultado_previsto(row):
    if row["pred_home"] > row["pred_away"]:
        return f"{row['home_pt']} vence"
    elif row["pred_away"] > row["pred_home"]:
        return f"{row['away_pt']} vence"
    else:
        return "Empate"

jogos_2026["resultado_previsto"] = jogos_2026.apply(resultado_previsto, axis=1)

# Exibir tabela completa de predições
print(f"\n  {'Grupo':<9} {'Rodada':<12} {'Mandante':<20} {'Pred':>5}  {'Visitante':<20} {'Pred':>5}  Resultado")
print("  " + "-"*90)
for _, row in jogos_2026.iterrows():
    print(f"  {row['grupo']:<9} {row['rodada']:<12} {row['home_pt']:<20} "
          f"{row['pred_home']:>5.2f}  {row['away_pt']:<20} "
          f"{row['pred_away']:>5.2f}  {row['resultado_previsto']}")


# ══════════════════════════════════════════════════════════
# 13. APROVEITAMENTO HISTÓRICO DAS SELEÇÕES 2026
# ══════════════════════════════════════════════════════════
print("\n" + "="*55)
print("APROVEITAMENTO HISTÓRICO DAS SELEÇÕES (base para predição)")
print("="*55)

todas_selecoes = sorted(set(
    jogos_2026["home_team"].tolist() + jogos_2026["away_team"].tolist()
))
tabela_aprov = pd.DataFrame([
    {"Seleção (EN)": t,
     "Seleção (PT)": next((k for k, v in NOME_PT_PARA_EN.items() if v == t), t),
     "Partidas": hist_acumulado.get(t, {}).get("partidas", 0),
     "Pontos":   hist_acumulado.get(t, {}).get("pontos", 0),
     "Aprov.":   aprov_selecao(t)}
    for t in todas_selecoes
]).sort_values("Aprov.", ascending=False)

print(f"\n  {'Seleção (PT)':<20} {'Partidas':>9} {'Pontos':>7} {'Aprov.':>8}")
print("  " + "-"*48)
for _, r in tabela_aprov.iterrows():
    print(f"  {r['Seleção (PT)']:<20} {r['Partidas']:>9} {r['Pontos']:>7} {r['Aprov.']:>8.3f}")


# ══════════════════════════════════════════════════════════
# 14. VISUALIZAÇÕES
# ══════════════════════════════════════════════════════════

# ── Gráfico 1: Avaliação do modelo (igual ao original) ───
fig = plt.figure(figsize=(16, 12))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.38)

ax1 = fig.add_subplot(gs[0, 0])
folds = [f"Fold {i+1}" for i in range(5)]
x = np.arange(5); w = 0.35
ax1.bar(x - w/2, r2_treino, w, label="Treino", color=AZUL,  alpha=0.85, edgecolor="white")
ax1.bar(x + w/2, r2_teste,  w, label="Teste",  color=VERDE, alpha=0.85, edgecolor="white")
ax1.axhline(r2_teste.mean(), color=LARANJA, linestyle="--", linewidth=1.5,
            label=f"Média teste: {r2_teste.mean():.3f}")
ax1.set_xticks(x); ax1.set_xticklabels(folds, fontsize=8)
ax1.set_title("R² por Fold\n(treino vs teste)", fontweight="bold", fontsize=11)
ax1.set_ylabel("R²"); ax1.legend(fontsize=8); ax1.set_facecolor("white")

ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter(y_test, y_pred_teste, alpha=0.4, s=20, color=AZUL, zorder=3)
lim_max = max(y_test.max(), y_pred_teste.max()) + 0.5
ax2.plot([0, lim_max], [0, lim_max], "--", color=LARANJA, linewidth=1.5, alpha=0.8, label="Previsão perfeita")
ax2.set_xlabel("Gols reais"); ax2.set_ylabel("Gols previstos")
ax2.set_title("Real vs Previsto\n(conjunto de teste)", fontweight="bold", fontsize=11)
ax2.legend(fontsize=8); ax2.set_facecolor("white")

ax3 = fig.add_subplot(gs[0, 2])
cores_coef = [AZUL if b > 0 else LARANJA for b in coef_df["beta"]]
bars3 = ax3.barh(coef_df["feature"][::-1], coef_df["beta"][::-1],
                 color=cores_coef[::-1], edgecolor="white", height=0.5)
ax3.axvline(0, color=CINZA, linewidth=0.8)
for bar, val in zip(bars3, coef_df["beta"][::-1]):
    ax3.text(val + (0.005 if val >= 0 else -0.005), bar.get_y() + bar.get_height()/2,
             f"{val:+.4f}", va="center", ha="left" if val >= 0 else "right", fontsize=9)
ax3.set_title("Coeficientes β\n(azul=↑gols, laranja=↓gols)", fontweight="bold", fontsize=11)
ax3.set_xlabel("β (escala normalizada)"); ax3.set_facecolor("white")

ax4 = fig.add_subplot(gs[1, 0])
ax4.hist(residuos, bins=20, color=AZUL, edgecolor="white", alpha=0.85)
ax4.axvline(0, color=LARANJA, linestyle="--", linewidth=1.8, label="Zero (ideal)")
ax4.axvline(residuos.mean(), color=VERDE, linestyle=":", linewidth=1.5,
            label=f"Média: {residuos.mean():.3f}")
ax4.set_title("Distribuição dos Resíduos\n(ideal: normal centrada em 0)", fontweight="bold", fontsize=11)
ax4.set_xlabel("Resíduo (real − previsto)"); ax4.set_ylabel("Frequência")
ax4.legend(fontsize=8); ax4.set_facecolor("white")

ax5 = fig.add_subplot(gs[1, 1])
ax5.scatter(y_pred_teste, residuos, alpha=0.4, s=20, color=VERDE)
ax5.axhline(0, color=LARANJA, linestyle="--", linewidth=1.5)
ax5.set_xlabel("Gols previstos"); ax5.set_ylabel("Resíduo")
ax5.set_title("Resíduos vs Previstos\n(sem padrão = bom ajuste)", fontweight="bold", fontsize=11)
ax5.set_facecolor("white")

ax6 = fig.add_subplot(gs[1, 2])
ax6.axis("off")
dados_tabela = [
    ["R² (CV média)",    f"{r2_teste.mean():.4f}"],
    ["R² (teste)",       f"{r2_final:.4f}"],
    ["RMSE (CV)",        f"{rmse_cv.mean():.4f}"],
    ["RMSE (teste)",     f"{rmse_final:.4f}"],
    ["MAE (CV)",         f"{mae_cv.mean():.4f}"],
    ["MAE (teste)",      f"{mae_final:.4f}"],
    ["Gap treino-teste", f"{gap:.4f}"],
    ["Amostras treino",  f"{len(X_train)}"],
    ["Amostras teste",   f"{len(X_test)}"],
]
tbl = ax6.table(cellText=dados_tabela, colLabels=["Métrica", "Valor"],
                cellLoc="center", loc="center", colWidths=[0.6, 0.35])
tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 1.6)
for j in range(2):
    tbl[0, j].set_facecolor(AZUL)
    tbl[0, j].set_text_props(color="white", fontweight="bold")
ax6.set_title("Resumo de Métricas", fontweight="bold", fontsize=11, pad=20)

fig.suptitle(
    f"Regressão Linear — Copa do Mundo  |  "
    f"home_score = {intercepto:.2f}"
    + "".join([f" {'+'if b>=0 else '-'} {abs(b):.2f}×{f}" for f, b in zip(FEATURES, coef)]),
    fontsize=10, fontweight="bold"
)
path_fig = os.path.join(OUTPUT, "avaliacao_modelo.png")
plt.savefig(path_fig, dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print(f"\n  Gráfico avaliação salvo: avaliacao_modelo.png")


# ── Gráfico 2: Predições Copa 2026 ───────────────────────
grupos_unicos = jogos_2026["grupo"].unique()
ncols = 3
nrows = int(np.ceil(len(grupos_unicos) / ncols))

fig2, axes = plt.subplots(nrows, ncols, figsize=(18, nrows * 4.2))
axes = axes.flatten()

for ax_i, grupo in enumerate(sorted(grupos_unicos)):
    ax = axes[ax_i]
    sub = jogos_2026[jogos_2026["grupo"] == grupo].reset_index(drop=True)

    labels   = [f"{r['home_pt']}\nvs\n{r['away_pt']}" for _, r in sub.iterrows()]
    x_pos    = np.arange(len(sub))
    w        = 0.35

    bars_h = ax.bar(x_pos - w/2, sub["pred_home"], w, label="Mandante", color=AZUL,   alpha=0.85, edgecolor="white")
    bars_a = ax.bar(x_pos + w/2, sub["pred_away"], w, label="Visitante", color=LARANJA, alpha=0.85, edgecolor="white")

    for bar in bars_h:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=7, color=AZUL)
    for bar in bars_a:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=7, color=LARANJA)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_title(grupo, fontweight="bold", fontsize=11)
    ax.set_ylabel("Gols previstos")
    ax.set_ylim(0, max(sub[["pred_home", "pred_away"]].max().max() + 0.5, 2.5))
    ax.legend(fontsize=8); ax.set_facecolor("white")

# Ocultar axes extras
for ax_i in range(len(grupos_unicos), len(axes)):
    axes[ax_i].set_visible(False)

fig2.suptitle("Copa do Mundo 2026 — Predição de Gols por Jogo (Fase de Grupos)",
              fontsize=14, fontweight="bold")
plt.tight_layout()
path_fig2 = os.path.join(OUTPUT, "predicoes_2026.png")
plt.savefig(path_fig2, dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print(f"  Gráfico predições salvo: predicoes_2026.png")


# ══════════════════════════════════════════════════════════
# 15. SALVAR MODELO E ARTEFATOS
# ══════════════════════════════════════════════════════════
joblib.dump(pipe,    os.path.join(OUTPUT, "modelo_copa.pkl"))
joblib.dump(FEATURES,os.path.join(OUTPUT, "features_lista.pkl"))

cols_salvar = ["grupo", "rodada", "data", "horario_et", "horario_local",
               "estadio", "cidade", "home_pt", "away_pt",
               "home_team", "away_team",
               "aprov_hist_home", "aprov_hist_away", "diff_aprov_hist",
               "media_gols_pro_hist_home", "media_gols_pro_hist_away", "diff_media_gols_pro_hist",
               "media_gols_contra_hist_home", "media_gols_contra_hist_away", "diff_media_gols_contra_hist",
               "saldo_medio_hist_home", "saldo_medio_hist_away", "diff_saldo_medio_hist",
               "partidas_hist_home", "partidas_hist_away", "diff_partidas_hist",
               "fase_ordinal", "fase_knockout",
               "pred_home", "pred_away", "resultado_previsto"]
cols_salvar = [col for col in cols_salvar if col in jogos_2026.columns]
jogos_2026[cols_salvar].to_csv(os.path.join(OUTPUT, "predicoes_2026.csv"), index=False)

metricas = {
    "features"         : FEATURES,
    "target"           : TARGET,
    "intercepto"       : float(intercepto),
    "coeficientes"     : dict(zip(FEATURES, [float(c) for c in coef])),
    "r2_cv_media"      : float(r2_teste.mean()),
    "r2_cv_desvio"     : float(r2_teste.std()),
    "rmse_cv_media"    : float(rmse_cv.mean()),
    "mae_cv_media"     : float(mae_cv.mean()),
    "r2_teste"         : float(r2_final),
    "rmse_teste"       : float(rmse_final),
    "mae_teste"        : float(mae_final),
    "gap_overfitting"  : float(gap),
    "n_amostras_total" : int(len(X)),
    "n_amostras_treino": int(len(X_train)),
    "n_amostras_teste" : int(len(X_test)),
}
with open(os.path.join(OUTPUT, "metricas.json"), "w", encoding="utf-8") as f:
    json.dump(metricas, f, indent=2, ensure_ascii=False)

print(f"\n  Modelo salvo      : modelo_copa.pkl")
print(f"  Features salvas   : features_lista.pkl")
print(f"  Métricas salvas   : metricas.json")
print(f"  Predições salvas  : predicoes_2026.csv")


# ══════════════════════════════════════════════════════════
# 16. RESUMO FINAL
# ══════════════════════════════════════════════════════════
print("\n" + "="*55)
print("RESUMO ETAPA 5")
print("="*55)

eq_linhas = "\n".join([
    f"             {'+' if c >= 0 else '-'} {abs(c):.4f} × {f}"
    for f, c in zip(FEATURES, coef)
])

print(f"""
  EQUAÇÃO FINAL:
  ══════════════════════════════════════════
  home_score = {intercepto:.4f}
{eq_linhas}
  ══════════════════════════════════════════

  DESEMPENHO:
    R² médio (CV)  : {r2_teste.mean():.4f}
    RMSE médio(CV) : {rmse_cv.mean():.4f} gols
    MAE médio (CV) : {mae_cv.mean():.4f} gols

  NOTA SOBRE O R²:
    R²={r2_teste.mean():.3f} é esperado para futebol.
    Resultados de futebol têm alta aleatoriedade —
    mesmo modelos profissionais ficam nessa faixa.
    O modelo captura a tendência geral corretamente.

  NOTA SOBRE aprov_hist:
    Calculado a partir de TODOS os jogos históricos
    (1930-2022) em limpeza.py (calcular_historico).
    Seleções estreantes na Copa recebem prior 0.5.
    O mesmo cálculo é refeito aqui (passo 8) para
    garantir consistência na predição dos jogos 2026.

  SAÍDAS GERADAS:
    modelo_copa.pkl        → modelo treinado
    metricas.json          → métricas de desempenho
    predicoes_2026.csv     → predições fase de grupos
    avaliacao_modelo.png   → gráficos de avaliação
    predicoes_2026.png     → gráficos de predição
""")
print("[Etapa 5 concluída]")
