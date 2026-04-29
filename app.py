“””
PME Finance Tracker
Application de collecte et d’analyse descriptive des données financières d’une PME
Cours : INF 232 EC2 — Analyse de données
Langage : Python | Framework : Streamlit
“””

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker
from datetime import date, datetime
import io

# ─────────────────────────────────────────────────────────────

# CONFIGURATION MATPLOTLIB (thème sombre)

# ─────────────────────────────────────────────────────────────

plt.rcParams.update({
“figure.facecolor”: “#0e1117”,
“axes.facecolor”:   “#1a1f2e”,
“axes.edgecolor”:   “#2d3748”,
“axes.labelcolor”:  “#a0aec0”,
“xtick.color”:      “#a0aec0”,
“ytick.color”:      “#a0aec0”,
“text.color”:       “#e2e8f0”,
“grid.color”:       “#2d3748”,
“grid.alpha”:       0.5,
“legend.facecolor”: “#1a1f2e”,
“legend.edgecolor”: “#2d3748”,
})

# ─────────────────────────────────────────────────────────────

# CONFIGURATION STREAMLIT

# ─────────────────────────────────────────────────────────────

st.set_page_config(
page_title=“PME Finance Tracker”,
page_icon=“🏦”,
layout=“wide”,
initial_sidebar_state=“expanded”
)

# ─────────────────────────────────────────────────────────────

# CONSTANTES

# ─────────────────────────────────────────────────────────────

MOIS = [
“Janvier”,“Février”,“Mars”,“Avril”,“Mai”,“Juin”,
“Juillet”,“Août”,“Septembre”,“Octobre”,“Novembre”,“Décembre”
]

CATEGORIES = {
“Revenu”: [
“Ventes de produits”,
“Prestations de services”,
“Subventions reçues”,
“Investissements reçus”,
“Autres revenus”
],
“Dépense”: [
“Salaires & charges”,
“Loyer & immobilier”,
“Matières premières”,
“Équipements & matériel”,
“Marketing & publicité”,
“Transport & logistique”,
“Impôts & taxes”,
“Services informatiques”,
“Frais bancaires”,
“Autres dépenses”
]
}

PAIEMENTS = [
“Espèces”, “Virement bancaire”,
“Mobile Money”, “Chèque”, “Carte bancaire”
]

COLONNES = [
“Date”, “Entreprise”, “Type”, “Catégorie”,
“Montant (FCFA)”, “Mode de paiement”,
“Mois”, “Responsable”, “Description”
]

# ─────────────────────────────────────────────────────────────

# SESSION STATE

# ─────────────────────────────────────────────────────────────

if “df” not in st.session_state:
demo = [
[“2025-01-10”,“SARL Kongo Commerce”,“Revenu”, “Ventes de produits”,     850000,“Virement bancaire”,“Janvier”, “J. Mbala”, “Lot A”],
[“2025-01-15”,“SARL Kongo Commerce”,“Dépense”,“Salaires & charges”,      300000,“Virement bancaire”,“Janvier”, “J. Mbala”, “”],
[“2025-01-20”,“SARL Kongo Commerce”,“Dépense”,“Loyer & immobilier”,      120000,“Espèces”,          “Janvier”, “J. Mbala”, “”],
[“2025-02-05”,“SARL Kongo Commerce”,“Revenu”, “Prestations de services”, 420000,“Mobile Money”,     “Février”, “A. Nzinga”,””],
[“2025-02-12”,“SARL Kongo Commerce”,“Dépense”,“Matières premières”,      180000,“Espèces”,          “Février”, “A. Nzinga”,””],
[“2025-03-01”,“SARL Kongo Commerce”,“Revenu”, “Ventes de produits”,      960000,“Virement bancaire”,“Mars”,    “J. Mbala”, “”],
[“2025-03-10”,“SARL Kongo Commerce”,“Dépense”,“Marketing & publicité”,    85000,“Carte bancaire”,   “Mars”,    “A. Nzinga”,””],
[“2025-03-20”,“SARL Kongo Commerce”,“Dépense”,“Salaires & charges”,      300000,“Virement bancaire”,“Mars”,    “J. Mbala”, “”],
[“2025-04-08”,“SARL Kongo Commerce”,“Revenu”, “Ventes de produits”,     1100000,“Virement bancaire”,“Avril”,   “J. Mbala”, “Commande importante”],
[“2025-04-15”,“SARL Kongo Commerce”,“Dépense”,“Équipements & matériel”,  250000,“Virement bancaire”,“Avril”,   “J. Mbala”, “Achat imprimante”],
[“2025-05-03”,“SARL Kongo Commerce”,“Revenu”, “Ventes de produits”,      730000,“Mobile Money”,     “Mai”,     “A. Nzinga”,””],
[“2025-05-18”,“SARL Kongo Commerce”,“Dépense”,“Transport & logistique”,   95000,“Espèces”,          “Mai”,     “A. Nzinga”,””],
]
st.session_state.df = pd.DataFrame(demo, columns=COLONNES)

# ─────────────────────────────────────────────────────────────

# FONCTIONS UTILITAIRES

# ─────────────────────────────────────────────────────────────

def get_df() -> pd.DataFrame:
return st.session_state.df.copy()

def fmt(n: float) -> str:
“”“Formate un montant en FCFA avec séparateurs.”””
return f”{int(n):,} FCFA”.replace(”,”, “ “)

def calc_stats(serie: pd.Series) -> pd.DataFrame:
“”“Retourne un tableau de statistiques descriptives.”””
if serie.empty:
return pd.DataFrame()
return pd.DataFrame({
“Indicateur”: [
“N (observations)”, “Somme”, “Moyenne”, “Médiane”,
“Minimum”, “Maximum”, “Écart-type”, “Variance”,
“Coeff. de variation (%)”, “1er Quartile Q1”, “3e Quartile Q3”
],
“Valeur”: [
int(len(serie)),
fmt(serie.sum()),
fmt(serie.mean()),
fmt(serie.median()),
fmt(serie.min()),
fmt(serie.max()),
fmt(serie.std()),
fmt(serie.var()),
f”{(serie.std() / serie.mean() * 100):.2f} %” if serie.mean() != 0 else “—”,
fmt(serie.quantile(0.25)),
fmt(serie.quantile(0.75)),
]
})

# ─────────────────────────────────────────────────────────────

# SIDEBAR

# ─────────────────────────────────────────────────────────────

with st.sidebar:
st.title(“🏦 PME Finance Tracker”)
st.caption(“INF 232 EC2 — Analyse de données”)
st.markdown(”—”)

```
page = st.radio("Navigation", [
    "📥 Saisie des données",
    "📋 Base de données",
    "📊 Analyse descriptive",
    "📈 Régression linéaire",
    "💾 Export des données"
])

st.markdown("---")
df_side = get_df()
if not df_side.empty:
    rev   = df_side[df_side["Type"] == "Revenu"]["Montant (FCFA)"].sum()
    dep   = df_side[df_side["Type"] == "Dépense"]["Montant (FCFA)"].sum()
    solde = rev - dep
    st.metric("Transactions",    len(df_side))
    st.metric("Total Revenus",   fmt(rev))
    st.metric("Total Dépenses",  fmt(dep))
    st.metric("Solde Net",       fmt(solde))
```

# ═════════════════════════════════════════════════════════════

# PAGE 1 — SAISIE DES DONNÉES

# ═════════════════════════════════════════════════════════════

if page == “📥 Saisie des données”:

```
st.title("📥 Saisie d'une transaction")
st.markdown("Remplissez le formulaire pour enregistrer une transaction financière dans la base de données.")
st.markdown("---")

with st.form(key="form_saisie", clear_on_submit=True):

    col1, col2 = st.columns(2)

    with col1:
        entreprise  = st.text_input("Entreprise / PME *",       placeholder="Ex : SARL Kongo Commerce")
        type_tx     = st.selectbox("Type de transaction *",      ["Revenu", "Dépense"])
        montant     = st.number_input("Montant (FCFA) *",        min_value=0, step=500, format="%d")
        mois        = st.selectbox("Mois de référence *",        MOIS, index=datetime.now().month - 1)

    with col2:
        date_tx     = st.date_input("Date de la transaction *",  value=date.today())
        categorie   = st.selectbox("Catégorie *",                CATEGORIES[type_tx])
        paiement    = st.selectbox("Mode de paiement",           PAIEMENTS)
        responsable = st.text_input("Responsable / Saisi par",   placeholder="Ex : Jean Mbala")

    description = st.text_area("Description / Note (optionnel)", placeholder="Détails complémentaires...")

    st.markdown(" ")
    soumettre = st.form_submit_button(
        "💾 Enregistrer la transaction",
        use_container_width=True,
        type="primary"
    )

    if soumettre:
        erreurs = []
        if not entreprise.strip():
            erreurs.append("Le champ **Entreprise** est obligatoire.")
        if montant <= 0:
            erreurs.append("Le **Montant** doit être supérieur à 0 FCFA.")

        if erreurs:
            for e in erreurs:
                st.error(e)
        else:
            nouvelle_ligne = pd.DataFrame([[
                str(date_tx), entreprise.strip(), type_tx, categorie,
                int(montant), paiement, mois, responsable.strip(), description.strip()
            ]], columns=COLONNES)

            st.session_state.df = pd.concat(
                [st.session_state.df, nouvelle_ligne],
                ignore_index=True
            )
            st.success(
                f"✅ Transaction enregistrée — {type_tx} de **{fmt(montant)}** | "
                f"Catégorie : {categorie} | {mois}"
            )
            st.balloons()
```

# ═════════════════════════════════════════════════════════════

# PAGE 2 — BASE DE DONNÉES

# ═════════════════════════════════════════════════════════════

elif page == “📋 Base de données”:

```
st.title("📋 Base de données")
st.markdown("Visualisez, filtrez et gérez l'ensemble des transactions.")
st.markdown("---")

df = get_df()

if df.empty:
    st.info("📭 Aucune donnée disponible. Commencez par saisir des transactions.")
else:
    col1, col2, col3 = st.columns(3)
    with col1:
        flt_type   = st.selectbox("Filtrer par type",  ["Tous", "Revenu", "Dépense"])
    with col2:
        flt_mois   = st.selectbox("Filtrer par mois",  ["Tous"] + MOIS)
    with col3:
        flt_search = st.text_input("Recherche libre",  placeholder="Mot-clé...")

    if flt_type != "Tous":
        df = df[df["Type"] == flt_type]
    if flt_mois != "Tous":
        df = df[df["Mois"] == flt_mois]
    if flt_search.strip():
        mask = df.apply(lambda row: flt_search.lower() in str(row).lower(), axis=1)
        df = df[mask]

    st.markdown(f"**{len(df)} transaction(s) affichée(s)**")
    st.dataframe(df.reset_index(drop=True), use_container_width=True, height=420)

    st.markdown("---")
    if st.button("🗑️ Supprimer toutes les données", type="secondary"):
        st.session_state.df = pd.DataFrame(columns=COLONNES)
        st.success("Base de données vidée.")
        st.rerun()
```

# ═════════════════════════════════════════════════════════════

# PAGE 3 — ANALYSE DESCRIPTIVE

# ═════════════════════════════════════════════════════════════

elif page == “📊 Analyse descriptive”:

```
st.title("📊 Analyse descriptive")
st.markdown("Indicateurs statistiques et visualisations graphiques de vos données financières.")
st.markdown("---")

df = get_df()

if len(df) < 2:
    st.warning("⚠️ Données insuffisantes. Enregistrez au moins 2 transactions.")
    st.stop()

revenus  = df[df["Type"] == "Revenu"]["Montant (FCFA)"]
depenses = df[df["Type"] == "Dépense"]["Montant (FCFA)"]
total_rev = revenus.sum()
total_dep = depenses.sum()
solde     = total_rev - total_dep
marge     = (solde / total_rev * 100) if total_rev > 0 else 0

# ── KPI globaux ──
c1, c2, c3, c4 = st.columns(4)
c1.metric("💰 Total Revenus",  fmt(total_rev))
c2.metric("📤 Total Dépenses", fmt(total_dep))
c3.metric("⚖️ Solde Net",      fmt(solde), delta=f"{solde:+,.0f}".replace(",", " "))
c4.metric("📉 Marge nette",    f"{marge:.1f} %")

st.markdown("---")

# ── Tableaux de statistiques descriptives ──
st.subheader("📐 Statistiques descriptives")
col_r, col_d = st.columns(2)
with col_r:
    st.markdown("**📗 Revenus**")
    st.dataframe(calc_stats(revenus), use_container_width=True, hide_index=True)
with col_d:
    st.markdown("**📕 Dépenses**")
    st.dataframe(calc_stats(depenses), use_container_width=True, hide_index=True)

st.markdown("---")

# ── Graphique 1 : Revenus vs Dépenses par mois ──
st.subheader("📊 Revenus et Dépenses par mois")

par_mois = df.groupby(["Mois", "Type"])["Montant (FCFA)"].sum().unstack(fill_value=0)
mois_presents = [m for m in MOIS if m in par_mois.index]
par_mois = par_mois.reindex(mois_presents)

fig1, ax1 = plt.subplots(figsize=(10, 4))
x = np.arange(len(mois_presents))
w = 0.35
if "Revenu" in par_mois.columns:
    ax1.bar(x - w/2, par_mois["Revenu"],  w, label="Revenus",  color="#10d47e", alpha=0.85)
if "Dépense" in par_mois.columns:
    ax1.bar(x + w/2, par_mois["Dépense"], w, label="Dépenses", color="#f05252", alpha=0.85)
ax1.set_xticks(x)
ax1.set_xticklabels(mois_presents, rotation=30, ha="right", fontsize=9)
ax1.yaxis.set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda v, _: f"{int(v):,}".replace(",", " "))
)
ax1.set_ylabel("Montant (FCFA)")
ax1.legend()
ax1.grid(axis="y", linestyle="--", alpha=0.4)
ax1.set_title("Comparaison mensuelle — Revenus / Dépenses", fontsize=12, pad=10)
plt.tight_layout()
st.pyplot(fig1)
plt.close(fig1)

st.markdown("---")

# ── Graphiques 2 & 3 côte à côte ──
col_g2, col_g3 = st.columns(2)

with col_g2:
    st.subheader("🥧 Répartition des dépenses par catégorie")
    dep_cat = df[df["Type"] == "Dépense"].groupby("Catégorie")["Montant (FCFA)"].sum()
    if dep_cat.empty:
        st.info("Aucune dépense enregistrée.")
    else:
        fig2, ax2 = plt.subplots(figsize=(5, 5))
        ax2.pie(
            dep_cat.values,
            labels=dep_cat.index,
            autopct="%1.1f%%",
            colors=plt.cm.tab10.colors,
            startangle=90,
            pctdistance=0.8
        )
        ax2.set_title("Dépenses par catégorie", fontsize=11)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

with col_g3:
    st.subheader("📈 Évolution du solde cumulé")
    df_s = df.sort_values("Date").reset_index(drop=True)
    df_s["Flux"]  = df_s.apply(
        lambda r: r["Montant (FCFA)"] if r["Type"] == "Revenu" else -r["Montant (FCFA)"], axis=1
    )
    df_s["Solde"] = df_s["Flux"].cumsum()

    fig3, ax3 = plt.subplots(figsize=(5, 5))
    ax3.plot(range(len(df_s)), df_s["Solde"],
             color="#f5c842", linewidth=2.5, marker="o", markersize=4)
    ax3.fill_between(range(len(df_s)), df_s["Solde"], alpha=0.12, color="#f5c842")
    ax3.axhline(0, color="#f05252", linestyle="--", linewidth=1, alpha=0.7, label="Seuil zéro")
    ax3.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{int(v):,}".replace(",", " "))
    )
    ax3.set_xlabel("N° transaction")
    ax3.set_ylabel("Solde cumulé (FCFA)")
    ax3.set_title("Évolution du solde dans le temps", fontsize=11)
    ax3.legend()
    ax3.grid(linestyle="--", alpha=0.4)
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)
```

# ═════════════════════════════════════════════════════════════

# PAGE 4 — RÉGRESSION LINÉAIRE SIMPLE

# ═════════════════════════════════════════════════════════════

elif page == “📈 Régression linéaire”:

```
st.title("📈 Régression linéaire simple")
st.markdown("Modélisation de la relation entre deux variables financières.")
st.markdown("---")

df = get_df()

if len(df) < 3:
    st.warning("⚠️ Il faut au moins 3 transactions pour effectuer une régression.")
    st.stop()

# Préparation des données
df_s = df.sort_values("Date").reset_index(drop=True)
df_s["Flux"]  = df_s.apply(
    lambda r: r["Montant (FCFA)"] if r["Type"] == "Revenu" else -r["Montant (FCFA)"], axis=1
)
df_s["Solde"] = df_s["Flux"].cumsum()
df_s["Index"] = range(1, len(df_s) + 1)

col1, col2 = st.columns(2)
with col1:
    choix_x = st.selectbox("Variable X (prédicteur)", [
        "Index (numéro de transaction)",
        "Montant de la transaction"
    ])
with col2:
    choix_y = st.selectbox("Variable Y (cible)", [
        "Solde cumulé",
        "Montant de la transaction"
    ])

X = df_s["Index"].values          if "Index"  in choix_x else df_s["Montant (FCFA)"].values
Y = df_s["Solde"].values          if "Solde"  in choix_y else df_s["Montant (FCFA)"].values

n     = len(X)
x_bar = X.mean()
y_bar = Y.mean()
Sxx   = np.sum((X - x_bar) ** 2)
Sxy   = np.sum((X - x_bar) * (Y - y_bar))

b1    = Sxy / Sxx
b0    = y_bar - b1 * x_bar
Y_hat = b0 + b1 * X

SS_tot = np.sum((Y - y_bar) ** 2)
SS_res = np.sum((Y - Y_hat) ** 2)
R2     = 1 - (SS_res / SS_tot) if SS_tot != 0 else 0
r      = np.sqrt(abs(R2)) * np.sign(b1)
rmse   = np.sqrt(SS_res / n)

# ── Graphique ──
fig_r, ax_r = plt.subplots(figsize=(10, 5))
ax_r.scatter(X, Y, color="#10d47e", alpha=0.8, s=60, zorder=3, label="Observations")
ax_r.plot(X, Y_hat, color="#f5c842", linewidth=2.5,
          label=f"Ŷ = {b0:,.0f} + {b1:.4f} · X".replace(",", " "))
ax_r.set_xlabel(choix_x)
ax_r.set_ylabel(choix_y)
ax_r.set_title("Nuage de points et droite de régression", fontsize=12, pad=10)
ax_r.legend()
ax_r.grid(linestyle="--", alpha=0.4)
ax_r.yaxis.set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda v, _: f"{int(v):,}".replace(",", " "))
)
plt.tight_layout()
st.pyplot(fig_r)
plt.close(fig_r)

st.markdown("---")

# ── Tableau des résultats ──
st.subheader("📐 Résultats numériques")

resultats = pd.DataFrame({
    "Paramètre": [
        "N (observations)",
        "Ordonnée à l'origine  β₀",
        "Pente  β₁",
        "Coefficient de corrélation  r",
        "Coefficient de détermination  R²",
        "RMSE (erreur quadratique moyenne)",
        "Qualité du modèle"
    ],
    "Valeur": [
        n,
        f"{b0:,.2f} FCFA".replace(",", " "),
        f"{b1:.4f}",
        f"{r:.4f}",
        f"{R2:.4f}",
        f"{rmse:,.2f} FCFA".replace(",", " "),
        "Excellent (R² > 0.8)" if R2 > 0.8 else "Modéré (0.5 < R² ≤ 0.8)" if R2 > 0.5 else "Faible (R² ≤ 0.5)"
    ]
})
st.dataframe(resultats, use_container_width=True, hide_index=True)

st.markdown("---")
st.subheader("📝 Interprétation")
direction = "positive ↗" if b1 > 0 else "négative ↘"
force_r   = "forte" if abs(r) > 0.7 else "modérée" if abs(r) > 0.4 else "faible"
st.markdown(f"""
```

- **Équation :** Ŷ = **{b0:,.0f}** + **{b1:.4f}** × X
- **Relation :** La relation entre X et Y est **{direction}**.
- **R² = {R2:.4f}** → Le modèle explique **{R2*100:.1f} %** de la variance de Y.
- **r = {r:.4f}** → Corrélation **{force_r}** entre les deux variables.
- **RMSE = {fmt(rmse)}** → Erreur de prédiction moyenne du modèle.
  “””.replace(”,”, “ “))

# ═════════════════════════════════════════════════════════════

# PAGE 5 — EXPORT DES DONNÉES

# ═════════════════════════════════════════════════════════════

elif page == “💾 Export des données”:

```
st.title("💾 Export des données")
st.markdown("Téléchargez vos données ou les scripts d'analyse prêts à l'emploi.")
st.markdown("---")

df = get_df()

if df.empty:
    st.info("📭 Aucune donnée à exporter.")
    st.stop()

# ── CSV ──
st.subheader("📄 Format CSV")
buf = io.StringIO()
df.to_csv(buf, index=False, encoding="utf-8-sig")
st.download_button(
    "⬇️ Télécharger CSV",
    data=buf.getvalue().encode("utf-8-sig"),
    file_name="pme_finance_data.csv",
    mime="text/csv",
    use_container_width=True
)

st.markdown("---")

# ── JSON ──
st.subheader("📦 Format JSON")
st.download_button(
    "⬇️ Télécharger JSON",
    data=df.to_json(orient="records", force_ascii=False, indent=2).encode("utf-8"),
    file_name="pme_finance_data.json",
    mime="application/json",
    use_container_width=True
)

st.markdown("---")

# ── Script R ──
st.subheader("📊 Script R — Analyse complète")
script_r = """\
```

# ═══════════════════════════════════════════════

# PME Finance Tracker — Analyse R

# INF 232 EC2

# ═══════════════════════════════════════════════

library(ggplot2)
library(dplyr)

# 1. Chargement des données

df <- read.csv(“pme_finance_data.csv”, encoding = “UTF-8”, stringsAsFactors = FALSE)
df$Montant..FCFA. <- as.numeric(df$Montant..FCFA.)
df$Date <- as.Date(df$Date)

# 2. Statistiques descriptives

cat(”\n=== STATISTIQUES DESCRIPTIVES ===\n”)
tapply(df$Montant..FCFA., df$Type, summary)

# 3. Solde cumulé

df <- df %>% arrange(Date) %>%
mutate(Flux  = ifelse(Type == “Revenu”, Montant..FCFA., -Montant..FCFA.),
Solde = cumsum(Flux),
Index = row_number())

# 4. Visualisation

ggplot(df, aes(x = Mois, y = Montant..FCFA., fill = Type)) +
geom_bar(stat = “sum”, position = “dodge”) +
coord_flip() +
labs(title = “Revenus vs Dépenses par mois”, x = “”, y = “FCFA”) +
theme_minimal()

ggplot(df, aes(x = Index, y = Solde)) +
geom_line(color = “#f5c842”, size = 1.2) +
geom_point(color = “#10d47e”, size = 2) +
geom_hline(yintercept = 0, linetype = “dashed”, color = “#f05252”) +
labs(title = “Evolution du solde cumulé”, x = “N° transaction”, y = “Solde FCFA”) +
theme_minimal()

# 5. Régression linéaire

model <- lm(Solde ~ Index, data = df)
cat(”\n=== REGRESSION LINEAIRE ===\n”)
print(summary(model))

ggplot(df, aes(x = Index, y = Solde)) +
geom_point(color = “#10d47e”) +
geom_smooth(method = “lm”, color = “#f5c842”, se = TRUE) +
labs(title = “Régression linéaire : Solde ~ Index”) +
theme_minimal()
“””
st.download_button(
“⬇️ Télécharger le script R”,
data=script_r.encode(“utf-8”),
file_name=“pme_analyse.R”,
mime=“text/plain”,
use_container_width=True
)

```
st.markdown("---")

# ── Script Python ──
st.subheader("🐍 Script Python — Analyse complète")
script_py = """\
```

# ═══════════════════════════════════════════════

# PME Finance Tracker — Analyse Python

# INF 232 EC2

# ═══════════════════════════════════════════════

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 1. Chargement des données

df = pd.read_csv(“pme_finance_data.csv”)
df[“Montant (FCFA)”] = pd.to_numeric(df[“Montant (FCFA)”])
df[“Date”] = pd.to_datetime(df[“Date”])
df = df.sort_values(“Date”).reset_index(drop=True)

# 2. Statistiques descriptives

print(”=== STATISTIQUES DESCRIPTIVES ===”)
print(df.groupby(“Type”)[“Montant (FCFA)”].describe())

# 3. Solde cumulé

df[“Flux”]  = df.apply(lambda r: r[“Montant (FCFA)”] if r[“Type”]==“Revenu” else -r[“Montant (FCFA)”], axis=1)
df[“Solde”] = df[“Flux”].cumsum()
df[“Index”] = range(1, len(df)+1)

# 4. Visualisation

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
grp = df.groupby([“Mois”,“Type”])[“Montant (FCFA)”].sum().unstack(fill_value=0)
grp.plot(kind=“bar”, ax=axes[0], color=[”#f05252”,”#10d47e”], alpha=0.85)
axes[0].set_title(“Revenus vs Dépenses par mois”)
axes[0].set_ylabel(“FCFA”)
axes[1].plot(df[“Index”], df[“Solde”], color=”#f5c842”, linewidth=2)
axes[1].axhline(0, color=”#f05252”, linestyle=”–”, alpha=0.7)
axes[1].fill_between(df[“Index”], df[“Solde”], alpha=0.1, color=”#f5c842”)
axes[1].set_title(“Evolution du solde cumulé”)
axes[1].set_ylabel(“FCFA”)
plt.tight_layout()
plt.show()

# 5. Régression linéaire simple

X = df[“Index”].values.reshape(-1, 1)
y = df[“Solde”].values
model = LinearRegression().fit(X, y)
print(f”\n=== RÉGRESSION LINÉAIRE ===”)
print(f”beta_0 (intercept) : {model.intercept_:.2f}”)
print(f”beta_1 (pente)     : {model.coef_[0]:.4f}”)
print(f”R²                 : {model.score(X, y):.4f}”)

plt.figure(figsize=(8, 5))
plt.scatter(df[“Index”], df[“Solde”], color=”#10d47e”, label=“Observations”)
plt.plot(df[“Index”], model.predict(X), color=”#f5c842”, linewidth=2, label=“Droite de régression”)
plt.title(“Régression linéaire : Solde ~ Index”)
plt.xlabel(“N° transaction”)
plt.ylabel(“Solde FCFA”)
plt.legend()
plt.tight_layout()
plt.show()
“””
st.download_button(
“⬇️ Télécharger le script Python”,
data=script_py.encode(“utf-8”),
file_name=“pme_analyse.py”,
mime=“text/plain”,
use_container_width=True
)

```
st.markdown("---")
st.subheader("👁️ Aperçu des données")
st.dataframe(df, use_container_width=True, height=300)
```
