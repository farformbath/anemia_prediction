import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_curve, roc_auc_score,
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, ConfusionMatrixDisplay
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, learning_curve, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import warnings
warnings.filterwarnings("ignore")

# ─── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Deteksi Risiko Anemia — Kelompok 8",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Main background */
.main { background-color: #f8f9fb; }

/* Title card */
.title-card {
    background: linear-gradient(135deg, #1a3a5c 0%, #2563a8 60%, #4c9be8 100%);
    padding: 2rem 2.5rem;
    border-radius: 16px;
    color: white;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 20px rgba(37,99,168,0.25);
}
.title-card h1 { margin: 0; font-size: 1.9rem; font-weight: 800; letter-spacing: -0.5px; }
.title-card p  { margin: 0.3rem 0 0; opacity: 0.85; font-size: 0.95rem; }

/* Metric cards */
.metric-card {
    background: white;
    border-radius: 12px;
    padding: 1.1rem 1.3rem;
    text-align: center;
    box-shadow: 0 2px 10px rgba(0,0,0,0.07);
    border-top: 4px solid;
    transition: transform .2s;
}
.metric-card:hover { transform: translateY(-3px); }
.metric-value { font-size: 1.9rem; font-weight: 800; margin: 0.2rem 0; }
.metric-label { font-size: 0.78rem; color: #6b7280; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }
.metric-delta { font-size: 0.8rem; margin-top: 0.2rem; }

/* Best model badge */
.best-badge {
    background: linear-gradient(135deg, #10b981, #059669);
    color: white; border-radius: 20px;
    padding: 0.2rem 0.7rem;
    font-size: 0.75rem; font-weight: 700;
    display: inline-block; margin-left: 0.4rem;
}

/* Section headers */
.section-header {
    font-size: 1.1rem; font-weight: 700;
    color: #1e3a5f; border-left: 4px solid #2563a8;
    padding-left: 0.8rem; margin: 1.5rem 0 1rem;
}

/* Prediction result */
.pred-anemia {
    background: linear-gradient(135deg, #fef2f2, #fee2e2);
    border: 2px solid #f87171;
    border-radius: 12px; padding: 1.2rem; text-align: center;
}
.pred-normal {
    background: linear-gradient(135deg, #f0fdf4, #dcfce7);
    border: 2px solid #4ade80;
    border-radius: 12px; padding: 1.2rem; text-align: center;
}

/* Sidebar styling */
.css-1d391kg { background: #1a3a5c !important; }

/* Table */
.styled-table { border-collapse: collapse; width: 100%; }
.styled-table th { background: #1a3a5c; color: white; padding: 0.6rem 1rem; text-align: center; }
.styled-table td { padding: 0.5rem 1rem; text-align: center; border-bottom: 1px solid #e5e7eb; }
.styled-table tr:hover td { background: #eff6ff; }
</style>
""", unsafe_allow_html=True)

# ─── Load models ────────────────────────────────────────────────────────────
# Naive Bayes  → Pipeline (StandardScaler sudah tertanam di dalamnya)
# Random Forest & Decision Tree → model biasa (tidak pakai scaling)
@st.cache_resource
def load_models():
    models = {}
    try:
        with open("naive_bayes_pipeline.pkl", "rb") as f:
            models["Naive Bayes"] = pickle.load(f)
        with open("random_forest_model.pkl", "rb") as f:
            models["Random Forest"] = pickle.load(f)
        with open("decision_tree_model.pkl", "rb") as f:
            models["Decision Tree"] = pickle.load(f)
        return models
    except FileNotFoundError as e:
        st.error(f"File model tidak ditemukan: {e}")
        return None

@st.cache_data
def load_data():
    """df_raw = anemia.csv (dengan duplikat) | df_clean = anemia_cleaned.csv (duplikat = 0)."""
    try:
        df_raw   = pd.read_csv("dataset/anemia.csv")
        df_clean = pd.read_csv("dataset/anemia_cleaned.csv")
        return df_raw, df_clean
    except FileNotFoundError as e:
        st.error(f"Dataset tidak ditemukan: {e}")
        return None, None

@st.cache_data
def load_results():
    """Load hasil evaluasi dari notebook — tidak ada retraining di app."""
    try:
        with open("results.pkl", "rb") as f:
            data = pickle.load(f)
        return data["results"], data["X_test"], data["y_test"]
    except FileNotFoundError:
        st.error("results.pkl tidak ditemukan! Jalankan notebook terlebih dahulu.")
        return None, None, None

models               = load_models()
df_raw, df           = load_data()          # df_raw = original, df = cleaned
results, X_test, y_test = load_results()

# ─── Helpers ────────────────────────────────────────────────────────────────
COLORS = {
    "Naive Bayes":    "#E85D75",
    "Random Forest":  "#3498db",
    "Decision Tree":  "#2ecc71",
}
PALETTE = ["#4C9BE8", "#E85D75"]
MODEL_NAMES = list(COLORS.keys())

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🩸 Navigasi")
    page = st.radio(
        "",
        ["🏠 Dashboard", "📊 EDA", "⚖️ Perbandingan Model",
         "📈 Evaluasi Detail", "🔮 Prediksi", "ℹ️ Info Dataset"],
        label_visibility="collapsed"
    )
    st.divider()
    st.markdown("**Kelompok 8 — Data Mining**")
    st.markdown("Deteksi Dini Risiko Anemia")
    st.markdown("*Naive Bayes · Random Forest · Decision Tree*")

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 1: DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════
if page == "🏠 Dashboard":
    st.markdown("""
    <div class='title-card'>
        <h1>🩸 Deteksi Dini Risiko Penyakit Anemia</h1>
        <p>Perbandingan Naive Bayes · Random Forest · Decision Tree &nbsp;|&nbsp; Kelompok 8 — Data Mining</p>
    </div>
    """, unsafe_allow_html=True)

    if df is not None and results is not None:
        # ── Dataset Overview ──
        st.markdown("<div class='section-header'>📁 Ringkasan Dataset</div>", unsafe_allow_html=True)
        dup_count = len(df_raw) - len(df)
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.markdown(f"""<div class='metric-card' style='border-color:#6b7280'>
                <div class='metric-label'>Data Raw</div>
                <div class='metric-value' style='color:#6b7280'>{len(df_raw):,}</div>
                <div class='metric-delta'>sebelum cleaning</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class='metric-card' style='border-color:#f59e0b'>
                <div class='metric-label'>Duplikat</div>
                <div class='metric-value' style='color:#f59e0b'>{dup_count:,}</div>
                <div class='metric-delta'>baris dihapus</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class='metric-card' style='border-color:#2563a8'>
                <div class='metric-label'>Data Cleaned</div>
                <div class='metric-value' style='color:#2563a8'>{len(df):,}</div>
                <div class='metric-delta'>dipakai training</div>
            </div>""", unsafe_allow_html=True)
        with c4:
            anemia_pct = df['Result'].mean() * 100
            st.markdown(f"""<div class='metric-card' style='border-color:#E85D75'>
                <div class='metric-label'>Kasus Anemia</div>
                <div class='metric-value' style='color:#E85D75'>{df['Result'].sum():,}</div>
                <div class='metric-delta'>{anemia_pct:.1f}% dari cleaned</div>
            </div>""", unsafe_allow_html=True)
        with c5:
            st.markdown(f"""<div class='metric-card' style='border-color:#10b981'>
                <div class='metric-label'>Tidak Anemia</div>
                <div class='metric-value' style='color:#10b981'>{(df['Result']==0).sum():,}</div>
                <div class='metric-delta'>{100-anemia_pct:.1f}% dari cleaned</div>
            </div>""", unsafe_allow_html=True)

        # ── Best model highlight ──
        st.markdown("<div class='section-header'>🏆 Performa Model</div>", unsafe_allow_html=True)

        best_name = max(results, key=lambda x: results[x]["CV Mean"])
        cols = st.columns(3)
        metrics_show = ["Accuracy", "F1-Score", "ROC-AUC"]
        border_c = {"Naive Bayes": "#E85D75", "Random Forest": "#3498db", "Decision Tree": "#2ecc71"}

        for i, name in enumerate(MODEL_NAMES):
            badge = "<span class='best-badge'>⭐ TERBAIK</span>" if name == best_name else ""
            with cols[i]:
                st.markdown(f"""<div class='metric-card' style='border-color:{border_c[name]}'>
                    <div class='metric-label'>{name}{badge}</div>""", unsafe_allow_html=True)
                for m in metrics_show:
                    val = results[name][m]
                    st.markdown(f"""<div style='margin:0.3rem 0'>
                        <span style='font-size:0.78rem;color:#6b7280'>{m}:</span>
                        <span style='font-weight:700;color:{border_c[name]};font-size:1.05rem'> {val*100:.2f}%</span>
                    </div>""", unsafe_allow_html=True)
                cv = results[name]["CV Mean"]
                cvs = results[name]["CV Std"]
                st.markdown(f"""<div style='margin:0.3rem 0'>
                    <span style='font-size:0.78rem;color:#6b7280'>CV 10-fold:</span>
                    <span style='font-weight:700;color:{border_c[name]};font-size:1.05rem'> {cv*100:.2f}% ±{cvs*100:.2f}%</span>
                </div></div>""", unsafe_allow_html=True)

        # ── Quick comparison bar ──
        st.markdown("<div class='section-header'>📊 Grafik Perbandingan Cepat</div>", unsafe_allow_html=True)
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        fig.patch.set_facecolor("#f8f9fb")

        metrics_bar = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
        x = np.arange(len(metrics_bar))
        w = 0.25
        for i, name in enumerate(MODEL_NAMES):
            vals = [results[name][m] * 100 for m in metrics_bar]
            axes[0].bar(x + i*w, vals, w, label=name, color=COLORS[name], alpha=0.88, edgecolor='white')
        axes[0].set_xticks(x + w)
        axes[0].set_xticklabels(metrics_bar, fontsize=9)
        axes[0].set_ylabel("Score (%)")
        axes[0].set_title("Perbandingan Metrik Evaluasi", fontweight="bold", fontsize=11)
        axes[0].legend(fontsize=8)
        axes[0].set_ylim(80, 102)
        axes[0].set_facecolor("#f8f9fb")
        axes[0].spines[["top","right"]].set_visible(False)

        # CV bar
        cv_means = [results[n]["CV Mean"]*100 for n in MODEL_NAMES]
        cv_stds  = [results[n]["CV Std"]*100  for n in MODEL_NAMES]
        bars = axes[1].bar(MODEL_NAMES, cv_means, color=[COLORS[n] for n in MODEL_NAMES],
                           alpha=0.88, edgecolor="white", width=0.5)
        axes[1].errorbar(range(3), cv_means, yerr=cv_stds, fmt="none", color="black", capsize=5, lw=2)
        for bar, val in zip(bars, cv_means):
            axes[1].text(bar.get_x()+bar.get_width()/2, val+0.3, f"{val:.2f}%",
                        ha="center", fontsize=9, fontweight="bold")
        axes[1].set_title("Cross-Validation 10-Fold", fontweight="bold", fontsize=11)
        axes[1].set_ylabel("Accuracy (%)")
        axes[1].set_ylim(85, 102)
        axes[1].set_facecolor("#f8f9fb")
        axes[1].spines[["top","right"]].set_visible(False)

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 2: EDA
# ═══════════════════════════════════════════════════════════════════════════
elif page == "📊 EDA":
    st.markdown("<div class='title-card'><h1>📊 Exploratory Data Analysis</h1><p>Eksplorasi dan visualisasi distribusi dataset anemia</p></div>", unsafe_allow_html=True)

    if df is not None:
        tab1, tab2, tab3, tab4 = st.tabs(["📋 Statistik", "📈 Distribusi", "🔗 Korelasi", "📦 Outlier"])

        with tab1:
            # ── Pilihan dataset ──
            dataset_choice = st.radio(
                "Pilih Dataset:",
                ["📂 Raw (anemia.csv)", "✅ Cleaned (anemia_cleaned.csv)"],
                horizontal=True
            )
            use_raw = dataset_choice.startswith("📂")
            df_view = df_raw if use_raw else df

            # Hitung duplikat dari selisih raw vs cleaned (bukan df_raw.duplicated()
            # karena duplikat sudah hanya bisa diketahui dari perbedaan jumlah baris)
            dup_count = len(df_raw) - len(df)

            col_i1, col_i2, col_i3 = st.columns(3)
            col_i1.metric("Total Baris", f"{len(df_view):,}")
            if use_raw:
                col_i2.metric("Duplikat", f"{dup_count:,}", delta="perlu dihapus", delta_color="inverse")
                col_i3.metric("Setelah Cleaning", f"{len(df):,}")
                st.warning(f"⚠️ Duplikat terdeteksi: **{dup_count} baris** ({len(df_raw):,} raw → {len(df):,} cleaned) — gunakan anemia_cleaned.csv untuk training")
            else:
                col_i2.metric("Duplikat", "0")
                col_i3.metric("Baris Dihapus", f"{dup_count:,}", delta="dari raw", delta_color="inverse")
                st.success(f"✅ Tidak ada duplikat — {dup_count} baris telah dihapus, dataset siap untuk training ({len(df):,} baris)")

            st.dataframe(df_view, use_container_width=True, height=280)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**Statistik Deskriptif ({'Raw' if use_raw else 'Cleaned'})**")
                st.dataframe(df_view.describe().round(3), use_container_width=True)
            with c2:
                st.markdown(f"**Info Kolom ({'Raw' if use_raw else 'Cleaned'})**")
                info_view = pd.DataFrame({
                    "Kolom":   df_view.columns,
                    "Tipe":    df_view.dtypes.values,
                    "Missing": df_view.isnull().sum().values,
                    "Unik":    df_view.nunique().values
                })
                st.dataframe(info_view, use_container_width=True, hide_index=True)

        with tab2:
            st.markdown("<div class='section-header'>Distribusi Kelas Target</div>", unsafe_allow_html=True)
            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            fig.patch.set_facecolor("#f8f9fb")
            label_map = {0: "Tidak Anemia", 1: "Anemia"}
            df_plot = df.copy(); df_plot["Result_Label"] = df_plot["Result"].map(label_map)
            counts = df_plot["Result_Label"].value_counts()
            axes[0].bar(counts.index, counts.values, color=PALETTE, width=0.5, edgecolor="white", linewidth=1.5)
            for i, v in enumerate(counts.values):
                axes[0].text(i, v+5, f"{v}\n({v/len(df)*100:.1f}%)", ha="center", fontsize=10)
            axes[0].set_title("Distribusi Kelas Target", fontweight="bold"); axes[0].spines[["top","right"]].set_visible(False)
            axes[0].set_facecolor("#f8f9fb")
            axes[1].pie(counts.values, labels=counts.index, colors=PALETTE, autopct="%1.1f%%",
                        startangle=90, wedgeprops={"edgecolor":"white","linewidth":2})
            axes[1].set_title("Proporsi Kelas", fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True); plt.close()

            st.markdown("<div class='section-header'>Distribusi Fitur Numerik per Kelas</div>", unsafe_allow_html=True)
            fitur_num = ["Hemoglobin", "MCH", "MCHC", "MCV"]
            fig, axes = plt.subplots(2, 2, figsize=(13, 8))
            fig.patch.set_facecolor("#f8f9fb")
            axes = axes.flatten()
            for i, feat in enumerate(fitur_num):
                for label, color in [(0, "#4C9BE8"), (1, "#E85D75")]:
                    data = df[df["Result"]==label][feat]
                    axes[i].hist(data, bins=25, alpha=0.6, color=color, label=label_map[label], edgecolor="white")
                axes[i].set_title(f"Distribusi {feat}", fontweight="bold", fontsize=11)
                axes[i].set_xlabel(feat); axes[i].set_ylabel("Frekuensi")
                axes[i].legend(); axes[i].spines[["top","right"]].set_visible(False)
                axes[i].set_facecolor("#f8f9fb")
            plt.suptitle("Distribusi Fitur Berdasarkan Kelas Anemia", fontweight="bold", y=1.01)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True); plt.close()

        with tab3:
            fig, ax = plt.subplots(figsize=(7, 5))
            fig.patch.set_facecolor("#f8f9fb")
            corr = df.corr()
            mask = np.triu(np.ones_like(corr, dtype=bool))
            sns.heatmap(corr, annot=True, fmt=".3f", cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                        mask=mask, ax=ax, linewidths=0.5, square=True, annot_kws={"size":10})
            ax.set_title("Heatmap Korelasi Antar Fitur", fontweight="bold", fontsize=13)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True); plt.close()

            st.markdown("**Korelasi terhadap Target (Result)**")
            target_corr = corr["Result"].drop("Result").sort_values(ascending=False)
            corr_df = pd.DataFrame({"Fitur": target_corr.index, "Korelasi": target_corr.values.round(4)})
            st.dataframe(corr_df, hide_index=True, use_container_width=True)

        with tab4:
            fig, axes = plt.subplots(2, 2, figsize=(13, 8))
            fig.patch.set_facecolor("#f8f9fb"); axes = axes.flatten()
            fitur_num = ["Hemoglobin", "MCH", "MCHC", "MCV"]
            for i, feat in enumerate(fitur_num):
                data_g = [df[df["Result"]==k][feat].values for k in [0,1]]
                bp = axes[i].boxplot(data_g, patch_artist=True, labels=["Tidak Anemia","Anemia"], notch=False)
                for patch, color in zip(bp["boxes"], PALETTE):
                    patch.set_facecolor(color); patch.set_alpha(0.7)
                axes[i].set_title(f"Boxplot {feat}", fontweight="bold"); axes[i].spines[["top","right"]].set_visible(False)
                axes[i].set_facecolor("#f8f9fb")
            plt.suptitle("Boxplot Fitur Hematologi per Kelas", fontweight="bold", y=1.01)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True); plt.close()

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 3: PERBANDINGAN MODEL
# ═══════════════════════════════════════════════════════════════════════════
elif page == "⚖️ Perbandingan Model":
    st.markdown("<div class='title-card'><h1>⚖️ Perbandingan Tiga Model</h1><p>Naive Bayes · Random Forest · Decision Tree</p></div>", unsafe_allow_html=True)

    if results is not None:
        # ── Summary table ──
        st.markdown("<div class='section-header'>📋 Tabel Ringkasan Performa</div>", unsafe_allow_html=True)
        cols_show = ["Accuracy","Precision","Recall","F1-Score","ROC-AUC","CV Mean","CV Std"]
        summary = pd.DataFrame({n: {c: results[n][c] for c in cols_show} for n in MODEL_NAMES}).T
        summary_pct = (summary * 100).round(2)

        best_name = summary["CV Mean"].idxmax()

        def color_best(val):
            return "background-color: #d1fae5; font-weight: bold" if val == summary_pct[val.name].max() else ""

        st.dataframe(
            summary_pct.style.apply(lambda col: ["background-color: #d1fae5; font-weight:bold"
                                                  if v == col.max() else "" for v in col], axis=0)
                              .format("{:.2f}%"),
            use_container_width=True
        )
        st.success(f"✅ **Model Terbaik: {best_name}** — CV Mean tertinggi ({results[best_name]['CV Mean']*100:.2f}%)")

        # ── Radar chart ──
        st.markdown("<div class='section-header'>🕸️ Radar Chart Perbandingan</div>", unsafe_allow_html=True)
        radar_metrics = ["Accuracy","Precision","Recall","F1-Score","ROC-AUC"]
        angles = np.linspace(0, 2*np.pi, len(radar_metrics), endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
        fig.patch.set_facecolor("#f8f9fb")
        ax.set_facecolor("#f0f4f8")
        for name in MODEL_NAMES:
            vals = [results[name][m]*100 for m in radar_metrics] + [results[name][radar_metrics[0]]*100]
            ax.plot(angles, vals, "o-", lw=2, color=COLORS[name], label=name)
            ax.fill(angles, vals, alpha=0.12, color=COLORS[name])
        ax.set_thetagrids(np.degrees(angles[:-1]), radar_metrics, fontsize=10)
        ax.set_ylim(80, 102)
        ax.set_title("Radar Chart Perbandingan Model", fontweight="bold", fontsize=12, pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.35,1.1), fontsize=9)
        st.pyplot(fig, use_container_width=True); plt.close()

        # ── ROC Curve all ──
        st.markdown("<div class='section-header'>📈 ROC Curve Semua Model</div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(8,5))
        fig.patch.set_facecolor("#f8f9fb"); ax.set_facecolor("#f8f9fb")
        for name in MODEL_NAMES:
            fpr, tpr = results[name]["fpr"], results[name]["tpr"]
            auc_v = results[name]["ROC-AUC"]
            ax.plot(fpr, tpr, label=f"{name} (AUC={auc_v:.4f})", color=COLORS[name], lw=2)
        ax.plot([0,1],[0,1],"k--",lw=1,label="Random (AUC=0.5)")
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve — Semua Model", fontweight="bold")
        ax.legend(loc="lower right"); ax.spines[["top","right"]].set_visible(False)
        st.pyplot(fig, use_container_width=True); plt.close()

        # ── Kesimpulan ──
        st.markdown("<div class='section-header'>💡 Kesimpulan Perbandingan</div>", unsafe_allow_html=True)
        for name in MODEL_NAMES:
            badge = "🏆 TERBAIK" if name == best_name else ""
            cv = results[name]["CV Mean"]; cvs = results[name]["CV Std"]
            f1 = results[name]["F1-Score"]; acc = results[name]["Accuracy"]
            with st.expander(f"{name}  {badge}"):
                c1, c2, c3 = st.columns(3)
                c1.metric("Accuracy",  f"{acc*100:.2f}%")
                c2.metric("F1-Score",  f"{f1*100:.2f}%")
                c3.metric("CV 10-fold",f"{cv*100:.2f}%", delta=f"±{cvs*100:.2f}%")
                if name == "Naive Bayes":
                    st.info("Naive Bayes mengasumsikan independensi antar fitur (Gaussian). Bekerja baik pada data hematologi yang memiliki distribusi mendekati normal.")
                elif name == "Random Forest":
                    st.info("Random Forest mengombinasikan banyak pohon keputusan (ensemble). Robust terhadap overfitting dan menangani fitur yang berkorelasi dengan baik.")
                else:
                    st.info("Decision Tree mudah diinterpretasi secara visual. Dengan pruning (max_depth=3, ccp_alpha=0.01) overfitting dapat dikurangi.")

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 4: EVALUASI DETAIL
# ═══════════════════════════════════════════════════════════════════════════
elif page == "📈 Evaluasi Detail":
    st.markdown("<div class='title-card'><h1>📈 Evaluasi Detail per Model</h1><p>Confusion Matrix · Learning Curve · Classification Report</p></div>", unsafe_allow_html=True)

    if results is not None:
        selected_model = st.selectbox("Pilih Model:", MODEL_NAMES)
        res = results[selected_model]

        # Metrics row
        st.markdown("<div class='section-header'>📊 Metrik Evaluasi</div>", unsafe_allow_html=True)
        c1,c2,c3,c4,c5 = st.columns(5)
        metrics_list = ["Accuracy","Precision","Recall","F1-Score","ROC-AUC"]
        cols = [c1,c2,c3,c4,c5]
        col_c = COLORS[selected_model]
        for ci, m in zip(cols, metrics_list):
            ci.markdown(f"""<div class='metric-card' style='border-color:{col_c}'>
                <div class='metric-label'>{m}</div>
                <div class='metric-value' style='color:{col_c}'>{res[m]*100:.2f}%</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("")
        tab1, tab2, tab3, tab4 = st.tabs(["🟦 Confusion Matrix", "📈 ROC Curve", "📚 Learning Curve", "📋 Report"])

        with tab1:
            fig, ax = plt.subplots(figsize=(5,4))
            fig.patch.set_facecolor("#f8f9fb"); ax.set_facecolor("#f8f9fb")
            cm = res["cm"]
            disp = ConfusionMatrixDisplay(cm, display_labels=["Tidak Anemia","Anemia"])
            disp.plot(ax=ax, colorbar=False, cmap="Blues")
            ax.set_title(f"Confusion Matrix — {selected_model}", fontweight="bold")
            plt.tight_layout()
            col_cm, _ = st.columns([1,1])
            with col_cm:
                st.pyplot(fig, use_container_width=True); plt.close()
            # Metrics from cm
            tn,fp,fn,tp = cm.ravel()
            st.markdown(f"""
            | Metrik | Nilai |
            |--------|-------|
            | True Positive  | {tp} |
            | True Negative  | {tn} |
            | False Positive | {fp} |
            | False Negative | {fn} |
            | Sensitivity (Recall) | {tp/(tp+fn)*100:.2f}% |
            | Specificity | {tn/(tn+fp)*100:.2f}% |
            """)

        with tab2:
            fig, ax = plt.subplots(figsize=(7,5))
            fig.patch.set_facecolor("#f8f9fb"); ax.set_facecolor("#f8f9fb")
            ax.plot(res["fpr"], res["tpr"], color=COLORS[selected_model], lw=2,
                    label=f"AUC = {res['ROC-AUC']:.4f}")
            ax.plot([0,1],[0,1],"k--",lw=1,label="Random")
            ax.fill_between(res["fpr"], res["tpr"], alpha=0.15, color=COLORS[selected_model])
            ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
            ax.set_title(f"ROC Curve — {selected_model}", fontweight="bold")
            ax.legend(); ax.spines[["top","right"]].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True); plt.close()

        with tab3:
            if results is not None:
                # Load learning curve data dari results.pkl (dihitung di notebook)
                lc = results[selected_model].get("learning_curve")
                if lc:
                    train_sz, train_sc, val_sc = lc["train_sizes"], lc["train_scores"], lc["val_scores"]
                else:
                    # Fallback: hitung ulang jika tidak ada di results.pkl
                    X = df.drop("Result",axis=1); y = df["Result"]
                    model_obj = models[selected_model]
                    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                    train_sz, train_sc, val_sc = learning_curve(
                        model_obj, X, y, cv=skf,
                        train_sizes=np.linspace(0.1,1.0,8), scoring="accuracy", n_jobs=-1
                    )
                fig, ax = plt.subplots(figsize=(8,4))
                fig.patch.set_facecolor("#f8f9fb"); ax.set_facecolor("#f8f9fb")
                t_mean = train_sc.mean(axis=1); t_std = train_sc.std(axis=1)
                v_mean = val_sc.mean(axis=1);   v_std = val_sc.std(axis=1)
                ax.plot(train_sz, t_mean*100, "o-", color="#E85D75", label="Train Score")
                ax.plot(train_sz, v_mean*100, "o-", color="#3DBE7A", label="Validation Score")
                ax.fill_between(train_sz, (t_mean-t_std)*100, (t_mean+t_std)*100, alpha=0.1, color="#E85D75")
                ax.fill_between(train_sz, (v_mean-v_std)*100, (v_mean+v_std)*100, alpha=0.1, color="#3DBE7A")
                ax.set_xlabel("Training Size"); ax.set_ylabel("Accuracy (%)")
                ax.set_title(f"Learning Curve — {selected_model}", fontweight="bold")
                ax.legend(); ax.set_ylim(75,105); ax.spines[["top","right"]].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True); plt.close()
                gap = abs(t_mean[-1] - v_mean[-1])*100
                if gap < 3:
                    st.success(f"✅ Gap Train-Val = {gap:.2f}% → Generalisasi baik, tidak overfitting")
                elif gap < 8:
                    st.warning(f"⚠️ Gap Train-Val = {gap:.2f}% → Sedikit overfitting")
                else:
                    st.error(f"❌ Gap Train-Val = {gap:.2f}% → Overfitting terdeteksi")

        with tab4:
            rpt = res["report"]
            rpt_df = pd.DataFrame(rpt).T.round(4)
            st.dataframe(rpt_df.style.format("{:.4f}", na_rep="—"), use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 5: PREDIKSI
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🔮 Prediksi":
    st.markdown("<div class='title-card'><h1>🔮 Prediksi Risiko Anemia</h1><p>Masukkan data pasien untuk mendapatkan prediksi dari ketiga model</p></div>", unsafe_allow_html=True)

    if models is not None:
        st.markdown("<div class='section-header'>🧪 Input Data Pasien</div>", unsafe_allow_html=True)

        with st.form("prediction_form"):
            c1, c2 = st.columns(2)
            with c1:
                gender    = st.selectbox("Gender", [0, 1], format_func=lambda x: "Perempuan (0)" if x==0 else "Laki-laki (1)")
                hemoglobin = st.number_input("Hemoglobin (g/dL)", min_value=4.0, max_value=20.0, value=12.5, step=0.1)
                mch       = st.number_input("MCH (pg)", min_value=10.0, max_value=45.0, value=27.0, step=0.1)
            with c2:
                mchc      = st.number_input("MCHC (g/dL)", min_value=20.0, max_value=40.0, value=32.0, step=0.1)
                mcv       = st.number_input("MCV (fL)", min_value=40.0, max_value=130.0, value=85.0, step=0.1)
                sel_model = st.selectbox("Model untuk Prediksi Utama", MODEL_NAMES)

            submitted = st.form_submit_button("🔍 Prediksi Sekarang", use_container_width=True)

        if submitted:
            # Input raw — NB Pipeline akan scale otomatis, RF & DT langsung predict
            input_data = pd.DataFrame([[gender, hemoglobin, mch, mchc, mcv]],
                                      columns=["Gender", "Hemoglobin", "MCH", "MCHC", "MCV"])

            st.markdown("<div class='section-header'>📋 Hasil Prediksi Semua Model</div>", unsafe_allow_html=True)
            cols = st.columns(3)
            for i, name in enumerate(MODEL_NAMES):
                pred  = models[name].predict(input_data)[0]
                prob  = models[name].predict_proba(input_data)[0]
                label = "Anemia" if pred == 1 else "Tidak Anemia"
                conf  = prob[pred] * 100
                css_class = "pred-anemia" if pred == 1 else "pred-normal"
                icon = "🔴" if pred == 1 else "🟢"
                with cols[i]:
                    st.markdown(f"""<div class='{css_class}'>
                        <div style='font-size:0.85rem;color:#6b7280;margin-bottom:0.3rem'>{name}</div>
                        <div style='font-size:1.6rem;font-weight:800'>{icon} {label}</div>
                        <div style='font-size:0.9rem;margin-top:0.4rem'>Keyakinan: <b>{conf:.1f}%</b></div>
                        <div style='font-size:0.8rem'>Anemia: {prob[1]*100:.1f}% | Normal: {prob[0]*100:.1f}%</div>
                    </div>""", unsafe_allow_html=True)

            # Probability comparison chart
            st.markdown("<div class='section-header'>📊 Probabilitas Anemia per Model</div>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(8,3))
            fig.patch.set_facecolor("#f8f9fb"); ax.set_facecolor("#f8f9fb")
            prob_vals = [models[n].predict_proba(input_data)[0][1]*100 for n in MODEL_NAMES]
            bars = ax.barh(MODEL_NAMES, prob_vals, color=[COLORS[n] for n in MODEL_NAMES],
                           alpha=0.85, edgecolor="white", height=0.45)
            ax.axvline(50, color="gray", linestyle="--", lw=1.5, label="Batas 50%")
            for bar, v in zip(bars, prob_vals):
                ax.text(v+0.5, bar.get_y()+bar.get_height()/2, f"{v:.1f}%", va="center", fontsize=10, fontweight="bold")
            ax.set_xlabel("Probabilitas Anemia (%)"); ax.set_xlim(0,110)
            ax.set_title("Probabilitas Prediksi Anemia", fontweight="bold")
            ax.legend(); ax.spines[["top","right"]].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True); plt.close()

            # Anjuran
            main_pred = models[sel_model].predict(input_data)[0]
            st.markdown("<div class='section-header'>💡 Rekomendasi</div>", unsafe_allow_html=True)
            if main_pred == 1:
                st.error("""
                **⚠️ Terindikasi Risiko Anemia**

                - Segera konsultasikan ke dokter atau tenaga kesehatan
                - Lakukan pemeriksaan darah lengkap (CBC)
                - Perhatikan asupan zat besi, asam folat, dan vitamin B12
                - Hindari kebiasaan yang memperparah kondisi (merokok, kurang tidur)
                """)
            else:
                st.success("""
                **✅ Tidak Terindikasi Anemia**

                - Pertahankan pola makan bergizi seimbang
                - Konsumsi makanan kaya zat besi (bayam, daging merah, kacang-kacangan)
                - Lakukan pemeriksaan rutin minimal 1x setahun
                """)

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 6: INFO DATASET
# ═══════════════════════════════════════════════════════════════════════════
elif page == "ℹ️ Info Dataset":
    st.markdown("<div class='title-card'><h1>ℹ️ Informasi Dataset & Model</h1><p>Detail teknis penelitian Kelompok 8</p></div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📁 Dataset Anemia")
        st.markdown("""
        | Atribut | Keterangan |
        |---------|-----------|
        | **Sumber** | Kaggle — Anemia Dataset |
        | **Jumlah Data** | 1.421 baris × 6 kolom |
        | **Target** | `Result` (0=Tidak Anemia, 1=Anemia) |
        | **Split** | 80% train, 20% test |
        | **Normalisasi** | StandardScaler |
        """)
        st.markdown("### 📌 Atribut Fitur")
        st.markdown("""
        - **Gender**: 0 = Perempuan, 1 = Laki-laki
        - **Hemoglobin**: Kadar protein pembawa oksigen (g/dL)
        - **MCH**: Mean Corpuscular Hemoglobin (pg)
        - **MCHC**: Mean Corpuscular Hemoglobin Concentration (g/dL)
        - **MCV**: Mean Corpuscular Volume (fL)
        """)

    with col2:
        st.markdown("### ⚙️ Konfigurasi Model")
        st.markdown("""
        **Naive Bayes (GaussianNB)**
        - Distribusi: Gaussian (cocok untuk fitur numerik kontinu)
        - Tidak ada hyperparameter khusus

        **Random Forest**
        - `n_estimators = 100`
        - `max_depth = None` (tidak dibatasi)
        - `random_state = 42`

        **Decision Tree**
        - `criterion = gini`
        - `max_depth = 3`
        - `min_samples_leaf = 10`
        - `ccp_alpha = 0.01` (pruning)
        """)
        st.markdown("### 📊 Metode Evaluasi")
        st.markdown("""
        - Accuracy, Precision, Recall, F1-Score, ROC-AUC
        - Cross-Validation: StratifiedKFold (k=10)
        - Learning Curve untuk deteksi overfitting
        - Confusion Matrix untuk analisis error
        """)

    st.divider()
    if df is not None:
        st.markdown("### 🗂️ Sample Data (10 baris pertama)")
        df_disp = df.head(10).copy()
        df_disp["Result"] = df_disp["Result"].map({0:"Tidak Anemia",1:"Anemia"})
        st.dataframe(df_disp, use_container_width=True, hide_index=True)
