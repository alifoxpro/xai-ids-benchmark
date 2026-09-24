"""
================================================================================
XAI-IDS BENCHMARK — Interactive Streamlit Dashboard
================================================================================
Displays results from all 10 stages + error analysis, ablation, comparison.

Usage:
    streamlit run dashboard.py
================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Optional, Dict, List

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

# ==============================================================================
# PATHS & CONSTANTS
# ==============================================================================
BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"
VIS_DIR = BASE_DIR / "visualizations"

STAGE_DIR_NAMES = {
    1: "stage1_artifacts",
    2: "stage2_feature_selection",
    3: "stage3_ml_models",
    4: "stage4_comprehensive",
    5: "stage5_xai",
    6: "stage6_performance",
    7: "stage7_cross_dataset",
    8: "stage8_robustness",
    9: "stage9_statistical",
    10: "stage10_final",
}

METRIC_COLS = ["Accuracy", "Balanced_Acc", "F1_Macro", "G_Mean", "FAR",
               "Precision", "Recall", "F1", "ROC-AUC", "MCC"]

# ==============================================================================
# DATA LOADERS  (each returns dict | DataFrame | None)
# ==============================================================================

def _rp(mode: str) -> Path:
    return RESULTS_DIR / mode

def _vp(mode: str) -> Path:
    return VIS_DIR / mode


@st.cache_data
def load_stage1(mode: str) -> Optional[Dict]:
    """Load Stage 1 metadata from cache or artifacts."""
    for p in [_rp(mode) / "stage1_cache" / "metadata.json",
              _rp(mode) / "stage1_artifacts" / "CIC_IoT_DIAD_2024_metadata.json"]:
        if p.exists():
            return json.loads(p.read_text())
    return None


@st.cache_data
def load_stage2(mode: str) -> Optional[Dict]:
    base = _rp(mode) / "stage2_feature_selection"
    if not base.exists():
        return None
    data = {}
    for csv in base.glob("*.csv"):
        data[csv.stem] = pd.read_csv(csv)
    for jf in base.glob("*.json"):
        data[jf.stem] = json.loads(jf.read_text())
    return data if data else None


@st.cache_data
def load_stage3(mode: str) -> Optional[Dict]:
    base = _rp(mode) / "stage3_ml_models"
    if not base.exists():
        return None
    data = {}
    comp = base / "model_comparison.csv"
    if comp.exists():
        data["comparison"] = pd.read_csv(comp)
    # Per-model JSONs
    models = {}
    for jf in base.glob("*_results.json"):
        mname = jf.stem.replace("_results", "")
        models[mname] = json.loads(jf.read_text())
    if models:
        data["models"] = models
    return data if data else None


@st.cache_data
def load_stage4(mode: str) -> Optional[Dict]:
    base = _rp(mode) / "stage4_comprehensive"
    if not base.exists():
        return None
    data = {}
    for csv in base.glob("*.csv"):
        data[csv.stem] = pd.read_csv(csv)
    for jf in base.glob("*.json"):
        data[jf.stem] = json.loads(jf.read_text())
    return data if data else None


@st.cache_data
def load_stage5(mode: str) -> Optional[Dict]:
    base = _rp(mode) / "stage5_xai"
    if not base.exists():
        return None
    data = {}
    for csv in base.glob("*.csv"):
        data[csv.stem] = pd.read_csv(csv)
    for jf in base.glob("*.json"):
        data[jf.stem] = json.loads(jf.read_text())
    return data if data else None


@st.cache_data
def load_stage6(mode: str) -> Optional[Dict]:
    base = _rp(mode) / "stage6_performance"
    if not base.exists():
        return None
    data = {}
    for csv in base.glob("*.csv"):
        data[csv.stem] = pd.read_csv(csv)
    for jf in base.glob("*.json"):
        data[jf.stem] = json.loads(jf.read_text())
    return data if data else None


@st.cache_data
def load_stage7(mode: str) -> Optional[Dict]:
    base = _rp(mode) / "stage7_cross_dataset"
    if not base.exists():
        return None
    data = {}
    for csv in base.glob("*.csv"):
        data[csv.stem] = pd.read_csv(csv)
    for jf in base.glob("*.json"):
        data[jf.stem] = json.loads(jf.read_text())
    return data if data else None


@st.cache_data
def load_stage8(mode: str) -> Optional[Dict]:
    base = _rp(mode) / "stage8_robustness"
    if not base.exists():
        return None
    data = {}
    for csv in base.glob("*.csv"):
        data[csv.stem] = pd.read_csv(csv)
    for jf in base.glob("*.json"):
        data[jf.stem] = json.loads(jf.read_text())
    return data if data else None


@st.cache_data
def load_stage9(mode: str) -> Optional[Dict]:
    base = _rp(mode) / "stage9_statistical"
    if not base.exists():
        return None
    data = {}
    for csv in base.glob("*.csv"):
        data[csv.stem] = pd.read_csv(csv)
    for jf in base.glob("*.json"):
        data[jf.stem] = json.loads(jf.read_text())
    return data if data else None


@st.cache_data
def load_stage10(mode: str) -> Optional[Dict]:
    base = _rp(mode) / "stage10_final"
    if not base.exists():
        return None
    data = {}
    for csv in base.glob("*.csv"):
        data[csv.stem] = pd.read_csv(csv)
    for jf in base.glob("*.json"):
        data[jf.stem] = json.loads(jf.read_text())
    return data if data else None


@st.cache_data
def load_error_analysis(mode: str) -> Optional[Dict]:
    base = _rp(mode) / "error_analysis"
    if not base.exists():
        return None
    data = {}
    for jf in base.glob("*.json"):
        data[jf.stem] = json.loads(jf.read_text())
    for csv in base.glob("*.csv"):
        data[csv.stem] = pd.read_csv(csv)
    return data if data else None


@st.cache_data
def load_ablation(mode: str) -> Optional[pd.DataFrame]:
    p = _rp(mode) / "ablation" / "ablation_results.csv"
    return pd.read_csv(p) if p.exists() else None


@st.cache_data
def load_comparison() -> Optional[pd.DataFrame]:
    p = RESULTS_DIR / "comparison" / "binary_vs_multiclass_summary.csv"
    return pd.read_csv(p) if p.exists() else None


# ==============================================================================
# HELPERS
# ==============================================================================

def stage_exists(mode: str, stage_num: int) -> bool:
    """Check if a stage's results directory exists and has files."""
    if stage_num == 1:
        return (load_stage1(mode) is not None)
    name = STAGE_DIR_NAMES.get(stage_num, "")
    d = _rp(mode) / name
    return d.exists() and any(d.iterdir()) if d.exists() else False


def show_pngs(mode: str, folder: str, title: str = "Saved Visualizations"):
    """Display all PNG images from a visualization folder."""
    vis_dir = _vp(mode) / folder
    if not vis_dir.exists():
        return
    pngs = sorted(vis_dir.glob("*.png"))
    if not pngs:
        return
    with st.expander(title, expanded=False):
        cols = st.columns(2)
        for i, png in enumerate(pngs):
            cols[i % 2].image(str(png), caption=png.stem.replace("_", " ").title(),
                              width="stretch")


def not_run_warning(stage: str, mode: str):
    st.warning(f"Stage {stage} has not been run yet for **{mode}** mode.")
    st.info(f"Run: `python main.py --stage {stage.split('.')[0] if '.' in stage else stage} --mode {mode}`")


def _label_map_to_names(meta: Dict) -> Dict[int, str]:
    """Convert label_mapping {name: id} → {id: name}."""
    lm = meta.get("label_mapping", {})
    return {int(v): k for k, v in lm.items()}


# ==============================================================================
# PAGE FUNCTIONS
# ==============================================================================

def page_overview(mode: str):
    st.title("XAI-Based IDS Benchmarking Study")
    st.markdown("**Comprehensive Comparison & Evaluation Framework** — CIC IoT-DIAD 2024 Dataset")
    st.markdown("---")

    meta = load_stage1(mode)

    # KPI cards
    c1, c2, c3, c4 = st.columns(4)
    if meta:
        c1.metric("Total Samples", f"{meta.get('n_samples', 0):,}")
        c2.metric("Features", meta.get("n_features", "—"))
        c3.metric("Classes", meta.get("n_classes", "—"))
    else:
        c1.metric("Total Samples", "—")
        c2.metric("Features", "—")
        c3.metric("Classes", "—")

    s3 = load_stage3(mode)
    if s3 and "comparison" in s3:
        best_acc = s3["comparison"]["Accuracy"].max() if "Accuracy" in s3["comparison"].columns else 0
        c4.metric("Best Accuracy", f"{best_acc:.4f}")
    else:
        c4.metric("Best Accuracy", "—")

    # Stage status
    st.markdown("### Stage Completion Status")
    status_data = []
    for i in range(1, 11):
        exists = stage_exists(mode, i)
        status_data.append({
            "Stage": f"Stage {i}",
            "Status": "Completed" if exists else "Not Run",
            "Icon": "✅" if exists else "⬜"
        })
    # Extra analyses
    for name, loader in [("Error Analysis", lambda: load_error_analysis(mode)),
                          ("Ablation Study", lambda: load_ablation(mode)),
                          ("Comparison", lambda: load_comparison())]:
        r = loader()
        exists = r is not None
        status_data.append({
            "Stage": name,
            "Status": "Completed" if exists else "Not Run",
            "Icon": "✅" if exists else "⬜"
        })

    status_df = pd.DataFrame(status_data)
    st.dataframe(status_df, width="stretch", hide_index=True)

    st.markdown("---")
    st.markdown("""
    **Pipeline Stages:**
    1. Dataset Preparation & Baseline
    2. Feature Selection Methods Benchmarking
    3. ML/DL Models Benchmarking (9 DL + VotingEnsemble)
    4. Comprehensive Results Benchmarking
    5. XAI Explanations Quality Benchmarking
    6. Performance Metrics Benchmarking
    7. Cross-Dataset Generalization Test
    8. Robustness & Adversarial Testing
    9. Statistical Significance Testing
    10. Final Benchmarking Summary & Report
    """)


def page_dataset(mode: str):
    st.title("Stage 1: Dataset & Preprocessing")
    meta = load_stage1(mode)
    if meta is None:
        not_run_warning("1", mode)
        return

    rev_map = _label_map_to_names(meta)
    counts_raw = meta.get("class_distribution", {}).get("counts", {})
    counts = {rev_map.get(int(k), f"Class_{k}"): int(v) for k, v in counts_raw.items()}

    # Summary
    c1, c2, c3 = st.columns(3)
    c1.metric("Samples", f"{meta['n_samples']:,}")
    c2.metric("Features", meta["n_features"])
    c3.metric("Classes", meta["n_classes"])

    st.markdown("---")

    # Class distribution bar
    if counts:
        df = pd.DataFrame({"Class": counts.keys(), "Samples": counts.values()})
        df = df.sort_values("Samples", ascending=False)

        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(df, x="Class", y="Samples", color="Class",
                         title="Class Distribution",
                         text_auto=True)
            fig.update_layout(showlegend=False, xaxis_tickangle=-45)
            st.plotly_chart(fig, width="stretch")

        with col2:
            fig2 = px.pie(df, names="Class", values="Samples",
                          title="Class Proportions", hole=0.35)
            st.plotly_chart(fig2, width="stretch")

        # Log scale toggle
        if st.checkbox("Show log scale"):
            fig_log = px.bar(df, x="Class", y="Samples", color="Class",
                             title="Class Distribution (Log Scale)",
                             text_auto=True, log_y=True)
            fig_log.update_layout(showlegend=False, xaxis_tickangle=-45)
            st.plotly_chart(fig_log, width="stretch")

    # Label mapping table
    st.markdown("### Label Mapping")
    lm = meta.get("label_mapping", {})
    st.dataframe(pd.DataFrame(lm.items(), columns=["Class Name", "Label ID"]),
                 hide_index=True)

    # Feature list
    feats = meta.get("feature_columns", [])
    with st.expander(f"Feature Columns ({len(feats)})"):
        st.write(feats)

    show_pngs(mode, "stage1")


def page_features(mode: str):
    st.title("Stage 2: Feature Selection")
    data = load_stage2(mode)
    if data is None:
        not_run_warning("2", mode)
        return

    # Display CSV tables
    for key in data:
        if isinstance(data[key], pd.DataFrame) and len(data[key]) > 0:
            df = data[key]
            st.markdown(f"### {key.replace('_', ' ').title()}")
            st.dataframe(df, width="stretch", hide_index=True)

            # Method comparison: Time and N_Features
            if "Method" in df.columns and "Time (s)" in df.columns:
                fig = px.bar(df, x="Method", y="Time (s)",
                             title="Feature Selection Time per Method",
                             color="Time (s)", color_continuous_scale="Blues",
                             text_auto=".1f")
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, width="stretch")

            # Try to plot feature importance if columns match
            if "feature" in df.columns and "importance" in df.columns:
                top = df.nlargest(20, "importance")
                fig = px.bar(top, x="importance", y="feature", orientation="h",
                             title=f"Top 20 Features — {key}",
                             color="importance", color_continuous_scale="Blues")
                fig.update_layout(yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig, width="stretch")

    # JSON summaries (feature lists)
    for key in data:
        if isinstance(data[key], (dict, list)):
            with st.expander(f"{key} (JSON)"):
                st.json(data[key])

    show_pngs(mode, "stage2")


def page_models(mode: str):
    st.title("Stage 3: Model Benchmarking")
    data = load_stage3(mode)
    if data is None:
        not_run_warning("3", mode)
        return

    comp = data.get("comparison")
    models = data.get("models", {})

    if comp is not None and len(comp) > 0:
        tab_perf, tab_eff, tab_cm, tab_train, tab_png = st.tabs(
            ["Performance", "Efficiency", "Confusion Matrices", "Training Curves", "Saved Plots"])

        with tab_perf:
            st.markdown("### Model Comparison")
            st.dataframe(comp, width="stretch", hide_index=True)

            # Metric selector
            avail = [c for c in METRIC_COLS if c in comp.columns]
            selected = st.multiselect("Select Metrics", avail,
                                       default=avail[:4] if len(avail) >= 4 else avail)
            if selected:
                model_col = "Model" if "Model" in comp.columns else comp.columns[0]
                melted = comp.melt(id_vars=[model_col], value_vars=selected,
                                   var_name="Metric", value_name="Score")
                fig = px.bar(melted, x=model_col, y="Score", color="Metric",
                             barmode="group", title="Performance Comparison",
                             text_auto=".4f")
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, width="stretch")

        with tab_eff:
            time_cols = [c for c in ["Train Time (s)", "Inference (ms)"] if c in comp.columns]
            model_col = "Model" if "Model" in comp.columns else comp.columns[0]

            if time_cols:
                for tc in time_cols:
                    fig = px.bar(comp.sort_values(tc), x=model_col, y=tc,
                                 title=tc, text_auto=".1f",
                                 color=tc, color_continuous_scale="Reds")
                    fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig, width="stretch")

            # Efficiency scatter
            train_time_col = "Train Time (s)" if "Train Time (s)" in comp.columns else None
            if train_time_col and "Accuracy" in comp.columns:
                fig = px.scatter(comp, x=train_time_col, y="Accuracy",
                                 text=model_col, size_max=15,
                                 title="Efficiency: Training Time vs Accuracy")
                fig.update_traces(textposition="top center", marker=dict(size=12))
                st.plotly_chart(fig, width="stretch")

        with tab_cm:
            if models:
                model_names = [m for m in models if "error" not in models[m]]
                sel_model = st.selectbox("Select Model", model_names)
                if sel_model and sel_model in models:
                    cm = models[sel_model].get("test", {}).get("confusion_matrix")
                    if cm is not None:
                        cm_arr = np.array(cm)
                        meta = load_stage1(mode)
                        class_names = list(meta.get("label_mapping", {}).keys()) if meta else None
                        fig = px.imshow(cm_arr, text_auto=True,
                                        color_continuous_scale="Blues",
                                        x=class_names, y=class_names,
                                        labels=dict(x="Predicted", y="True"),
                                        title=f"Confusion Matrix — {sel_model}")
                        st.plotly_chart(fig, width="stretch")
                    else:
                        st.info("No confusion matrix saved for this model.")
            else:
                st.info("No per-model results found.")

        with tab_train:
            if models:
                model_names = [m for m in models
                               if "error" not in models[m]
                               and models[m].get("training_history")]
                if model_names:
                    sel = st.selectbox("Select Model", model_names, key="train_sel")
                    hist = models[sel].get("training_history", {})
                    if hist:
                        epochs = list(range(1, len(hist.get("train_loss", hist.get("loss", []))) + 1))
                        traces = []
                        for key, label in [("train_loss", "Train Loss"), ("val_loss", "Val Loss"),
                                            ("loss", "Train Loss")]:
                            if key in hist:
                                traces.append(go.Scatter(x=epochs, y=hist[key],
                                                          mode="lines+markers", name=label))
                        if traces:
                            fig = go.Figure(data=traces)
                            fig.update_layout(title=f"Training Curves — {sel}",
                                              xaxis_title="Epoch", yaxis_title="Loss")
                            st.plotly_chart(fig, width="stretch")

                        # Accuracy curves
                        acc_traces = []
                        for key, label in [("train_acc", "Train Acc"), ("val_acc", "Val Acc")]:
                            if key in hist:
                                acc_traces.append(go.Scatter(x=epochs, y=hist[key],
                                                              mode="lines+markers", name=label))
                        if acc_traces:
                            fig2 = go.Figure(data=acc_traces)
                            fig2.update_layout(title=f"Accuracy Curves — {sel}",
                                               xaxis_title="Epoch", yaxis_title="Accuracy")
                            st.plotly_chart(fig2, width="stretch")
                else:
                    st.info("No training history found.")
            else:
                st.info("No per-model results found.")

        with tab_png:
            show_pngs(mode, "stage3", "Stage 3 Saved Visualizations")

    else:
        st.info("No model comparison data found.")
        show_pngs(mode, "stage3")


def page_comprehensive(mode: str):
    st.title("Stage 4: Comprehensive Results")
    data = load_stage4(mode)
    if data is None:
        not_run_warning("4", mode)
        return

    for key in data:
        if isinstance(data[key], pd.DataFrame) and len(data[key]) > 0:
            df = data[key]
            st.markdown(f"### {key.replace('_', ' ').title()}")

            # Feature x Model heatmap
            if "Feature_Method" in df.columns and "Model" in df.columns and "Accuracy" in df.columns:
                pivot = df.pivot_table(index="Feature_Method", columns="Model",
                                        values="Accuracy", aggfunc="mean")
                fig = px.imshow(pivot, text_auto=".3f",
                                color_continuous_scale="YlOrRd",
                                title="Feature Method x Model Accuracy Matrix")
                st.plotly_chart(fig, width="stretch")
            else:
                st.dataframe(df, width="stretch", hide_index=True)

    for key in data:
        if isinstance(data[key], dict):
            with st.expander(f"{key} (JSON)"):
                st.json(data[key])

    show_pngs(mode, "stage4")


def page_xai(mode: str):
    st.title("Stage 5: XAI Quality Benchmarking")
    data = load_stage5(mode)
    if data is None:
        not_run_warning("5", mode)
        return

    for key in data:
        if isinstance(data[key], pd.DataFrame) and len(data[key]) > 0:
            df = data[key]
            st.markdown(f"### {key.replace('_', ' ').title()}")
            st.dataframe(df, width="stretch", hide_index=True)

            # Try plotting fidelity/stability/time
            method_col = None
            for c in ["Method", "XAI_Method", "method"]:
                if c in df.columns:
                    method_col = c
                    break

            if method_col:
                for metric in ["fidelity", "Fidelity", "stability", "Stability",
                                "explanation_time", "Explanation_Time_ms"]:
                    if metric in df.columns:
                        fig = px.bar(df, x=method_col, y=metric,
                                     title=f"{metric.replace('_', ' ').title()} per XAI Method",
                                     color=method_col, text_auto=".4f")
                        st.plotly_chart(fig, width="stretch")

    for key in data:
        if isinstance(data[key], dict):
            with st.expander(f"{key} (JSON)"):
                st.json(data[key])

    show_pngs(mode, "stage5")


def page_performance(mode: str):
    st.title("Stage 6: Performance Metrics")
    data = load_stage6(mode)
    if data is None:
        not_run_warning("6", mode)
        return

    # Per-class performance
    for key in data:
        if isinstance(data[key], pd.DataFrame) and "Class" in data[key].columns:
            df = data[key]
            st.markdown("### Per-Class Performance")
            st.dataframe(df, width="stretch", hide_index=True)

            # Grouped bar chart
            metric_cols = [c for c in ["Precision", "Recall", "F1-Score", "F1_Score",
                                        "Detection_Rate"] if c in df.columns]
            if metric_cols:
                melted = df.melt(id_vars=["Class"], value_vars=metric_cols,
                                  var_name="Metric", value_name="Score")
                fig = px.bar(melted, x="Class", y="Score", color="Metric",
                             barmode="group", title="Per-Class Metrics",
                             text_auto=".3f")
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, width="stretch")

    # Radar chart from effectiveness
    for key in data:
        if isinstance(data[key], dict) and "accuracy" in data[key]:
            eff = data[key]
            st.markdown("### Effectiveness Summary")
            radar_metrics = ["accuracy", "precision_weighted", "recall_weighted",
                              "f1_weighted", "mcc"]
            names = ["Accuracy", "Precision", "Recall", "F1", "MCC"]
            values = [eff.get(m, 0) for m in radar_metrics]
            values_plot = values + [values[0]]  # close the polygon
            names_plot = names + [names[0]]

            fig = go.Figure(data=go.Scatterpolar(
                r=values_plot, theta=names_plot,
                fill="toself", name="Model"
            ))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                              title="Performance Radar Chart")
            st.plotly_chart(fig, width="stretch")

    for key in data:
        if isinstance(data[key], dict):
            with st.expander(f"{key} (JSON)"):
                st.json(data[key])

    show_pngs(mode, "stage6")


def page_generalization(mode: str):
    st.title("Stage 7: Cross-Dataset Generalization")
    data = load_stage7(mode)
    if data is None:
        not_run_warning("7", mode)
        return

    # Transfer results
    for key in data:
        if isinstance(data[key], pd.DataFrame) and len(data[key]) > 0:
            df = data[key]
            st.markdown(f"### {key.replace('_', ' ').title()}")
            st.dataframe(df, width="stretch", hide_index=True)

            model_col = None
            for c in ["Model", "model"]:
                if c in df.columns:
                    model_col = c
                    break

            # Generalization gap
            if model_col and "generalization_gap" in [c.lower().replace(" ", "_") for c in df.columns]:
                gap_col = [c for c in df.columns if "gap" in c.lower()][0]
                fig = px.bar(df, x=model_col, y=gap_col,
                             title="Generalization Gap per Model",
                             color=gap_col, color_continuous_scale="RdYlGn_r",
                             text_auto=".3f")
                st.plotly_chart(fig, width="stretch")

            # Source vs target accuracy
            acc_cols = [c for c in df.columns if "accuracy" in c.lower() or "acc" in c.lower()]
            if model_col and len(acc_cols) >= 2:
                melted = df.melt(id_vars=[model_col], value_vars=acc_cols,
                                  var_name="Dataset", value_name="Accuracy")
                fig = px.bar(melted, x=model_col, y="Accuracy", color="Dataset",
                             barmode="group", title="Source vs Target Accuracy",
                             text_auto=".3f")
                st.plotly_chart(fig, width="stretch")

    show_pngs(mode, "stage7")


def page_robustness(mode: str):
    st.title("Stage 8: Robustness & Adversarial Testing")
    data = load_stage8(mode)
    if data is None:
        not_run_warning("8", mode)
        return

    tabs = st.tabs(["FGSM", "PGD", "Noise", "Overall", "Saved Plots"])

    attack_keys = {"fgsm": "FGSM", "pgd": "PGD", "noise": "Gaussian Noise"}

    for i, (akey, alabel) in enumerate(attack_keys.items()):
        with tabs[i]:
            # Find matching DataFrame
            attack_df = None
            for key in data:
                if akey in key.lower() and isinstance(data[key], pd.DataFrame):
                    attack_df = data[key]
                    break
            if attack_df is not None and len(attack_df) > 0:
                st.dataframe(attack_df, width="stretch", hide_index=True)
                # Find epsilon/sigma column and accuracy column
                x_col = None
                y_col = None
                for c in attack_df.columns:
                    cl = c.lower()
                    if "epsilon" in cl or "sigma" in cl or "level" in cl or "strength" in cl:
                        x_col = c
                    if "accuracy" in cl or "acc" in cl:
                        y_col = c
                if x_col and y_col:
                    fig = px.line(attack_df, x=x_col, y=y_col, markers=True,
                                  title=f"Accuracy Under {alabel} Attack")
                    fig.update_layout(yaxis_range=[0, 1.05])
                    st.plotly_chart(fig, width="stretch")
            else:
                st.info(f"No {alabel} results found.")

    with tabs[3]:
        # Robustness scores
        for key in data:
            if isinstance(data[key], dict) and any("robust" in k.lower() for k in data[key]):
                st.markdown("### Overall Robustness Scores")
                scores = data[key]
                score_df = pd.DataFrame([
                    {"Attack": k.replace("_", " ").title(), "Score": v}
                    for k, v in scores.items() if isinstance(v, (int, float))
                ])
                if len(score_df) > 0:
                    fig = px.bar(score_df, x="Attack", y="Score",
                                 title="Robustness Scores",
                                 color="Score", color_continuous_scale="RdYlGn",
                                 text_auto=".3f")
                    fig.update_layout(yaxis_range=[0, 1.05])
                    st.plotly_chart(fig, width="stretch")
                st.json(scores)
                break
        else:
            st.info("No robustness scores found.")

    with tabs[4]:
        show_pngs(mode, "stage8", "Stage 8 Saved Visualizations")


def page_statistical(mode: str):
    st.title("Stage 9: Statistical Significance Testing")
    data = load_stage9(mode)
    if data is None:
        not_run_warning("9", mode)
        return

    # P-value heatmap from pairwise tests
    for key in data:
        if isinstance(data[key], pd.DataFrame):
            df = data[key]
            st.markdown(f"### {key.replace('_', ' ').title()}")
            st.dataframe(df, width="stretch", hide_index=True)

            # Try to create p-value heatmap
            m1 = m2 = pv = None
            for c in df.columns:
                cl = c.lower()
                if "model_1" in cl or "model_a" in cl:
                    m1 = c
                elif "model_2" in cl or "model_b" in cl:
                    m2 = c
                elif "p_value" in cl or "pvalue" in cl:
                    pv = c
            if m1 and m2 and pv:
                models = sorted(set(df[m1].tolist() + df[m2].tolist()))
                matrix = pd.DataFrame(1.0, index=models, columns=models)
                for _, row in df.iterrows():
                    matrix.loc[row[m1], row[m2]] = row[pv]
                    matrix.loc[row[m2], row[m1]] = row[pv]
                np.fill_diagonal(matrix.values, 0.0)

                fig = px.imshow(matrix, text_auto=".4f",
                                color_continuous_scale="RdYlGn_r",
                                zmin=0, zmax=0.1,
                                title="P-Value Matrix (Pairwise Tests)")
                st.plotly_chart(fig, width="stretch")

    # JSON results
    for key in data:
        if isinstance(data[key], dict):
            with st.expander(f"{key} (JSON)"):
                st.json(data[key])

    show_pngs(mode, "stage9")


def page_final_report(mode: str):
    st.title("Stage 10: Final Report & Recommendations")
    data = load_stage10(mode)
    if data is None:
        not_run_warning("10", mode)
        return

    # Rankings
    for key in data:
        if isinstance(data[key], dict) and "rankings" in key.lower():
            st.markdown("### Final Rankings")
            rankings = data[key]
            if isinstance(rankings, dict):
                for category, items in rankings.items():
                    st.markdown(f"**{category.replace('_', ' ').title()}**")
                    if isinstance(items, list):
                        st.dataframe(pd.DataFrame(items), hide_index=True)
                    elif isinstance(items, dict):
                        st.json(items)

    # Recommendations
    for key in data:
        if isinstance(data[key], dict) and "recommend" in key.lower():
            st.markdown("### Recommendations")
            recs = data[key]
            for cat, info in recs.items():
                if isinstance(info, dict):
                    model = info.get("model", info.get("method", "—"))
                    score = info.get("score", info.get("accuracy", "—"))
                    reason = info.get("reason", info.get("rationale", ""))
                    st.markdown(f"""
                    > **{cat.replace('_', ' ').title()}**: `{model}` (Score: {score})
                    > {reason}
                    """)
                else:
                    st.markdown(f"**{cat}**: {info}")

    # CSV tables
    for key in data:
        if isinstance(data[key], pd.DataFrame) and len(data[key]) > 0:
            st.markdown(f"### {key.replace('_', ' ').title()}")
            df = data[key]
            st.dataframe(df, width="stretch", hide_index=True)

            model_col = None
            for c in ["Model", "model"]:
                if c in df.columns:
                    model_col = c
                    break
            score_col = None
            for c in df.columns:
                if "score" in c.lower() or "accuracy" in c.lower() or "rank" in c.lower():
                    score_col = c
                    break
            if model_col and score_col:
                fig = px.bar(df, x=model_col, y=score_col,
                             title="Final Model Rankings", text_auto=".4f",
                             color=score_col, color_continuous_scale="Greens")
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, width="stretch")

    show_pngs(mode, "stage10")


def page_error_analysis(mode: str):
    st.title("Error Analysis")
    data = load_error_analysis(mode)
    if data is None:
        not_run_warning("error_analysis", mode)
        return

    # Find error analysis JSONs
    ea_dicts = {k: v for k, v in data.items() if isinstance(v, dict)}
    ea_dfs = {k: v for k, v in data.items() if isinstance(v, pd.DataFrame)}

    for name, ea in ea_dicts.items():
        st.markdown(f"### {name.replace('_', ' ').title()}")

        # Summary metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Samples", f"{ea.get('total_samples', 0):,}")
        c2.metric("Total Errors", f"{ea.get('total_errors', 0):,}")
        c3.metric("Error Rate", f"{ea.get('error_rate', 0) * 100:.2f}%")

        # Confusion pairs
        pairs = ea.get("confusion_pairs", [])
        if pairs:
            pairs_df = pd.DataFrame(pairs)
            if "True_Class" in pairs_df.columns and "Predicted_Class" in pairs_df.columns:
                pairs_df["Pair"] = pairs_df["True_Class"] + " → " + pairs_df["Predicted_Class"]
                top = pairs_df.head(10)
                fig = px.bar(top, x="Count", y="Pair", orientation="h",
                             title="Top 10 Confusion Pairs",
                             color="Count", color_continuous_scale="Reds",
                             text_auto=True)
                fig.update_layout(yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig, width="stretch")

        # Per-class error rate
        pce = ea.get("per_class_error", [])
        if pce:
            pce_df = pd.DataFrame(pce)
            if "Class" in pce_df.columns and "Error_Rate" in pce_df.columns:
                pce_df = pce_df.sort_values("Error_Rate", ascending=False)
                fig = px.bar(pce_df, x="Class", y="Error_Rate",
                             title="Error Rate per Class",
                             color="Error_Rate", color_continuous_scale="OrRd",
                             text_auto=".3f")
                fig.add_hline(y=0.05, line_dash="dash", line_color="green",
                              annotation_text="5% threshold")
                st.plotly_chart(fig, width="stretch")

    for name, df in ea_dfs.items():
        st.markdown(f"### {name.replace('_', ' ').title()}")
        st.dataframe(df, width="stretch", hide_index=True)

    show_pngs(mode, "error_analysis")


def page_ablation(mode: str):
    st.title("Ablation Study")
    df = load_ablation(mode)
    if df is None:
        not_run_warning("ablation", mode)
        return

    st.dataframe(df, width="stretch", hide_index=True)

    # Multi-metric line chart
    metric_cols = [c for c in ["accuracy", "f1", "balanced_accuracy", "Accuracy",
                                "F1", "Balanced_Accuracy"] if c in df.columns]
    feat_col = None
    for c in df.columns:
        if "feature" in c.lower() or "n_feat" in c.lower():
            feat_col = c
            break

    if feat_col and metric_cols:
        melted = df.melt(id_vars=[feat_col], value_vars=metric_cols,
                          var_name="Metric", value_name="Score")
        fig = px.line(melted, x=feat_col, y="Score", color="Metric",
                       markers=True, title="Ablation: Performance vs Feature Count")
        fig.update_layout(xaxis_title="Number of Features", yaxis_title="Score")
        st.plotly_chart(fig, width="stretch")

    # Training time
    time_col = None
    for c in df.columns:
        if "time" in c.lower():
            time_col = c
            break
    if feat_col and time_col:
        fig = px.bar(df, x=feat_col, y=time_col,
                      title="Training Time vs Feature Count",
                      text_auto=".1f", color=time_col,
                      color_continuous_scale="Blues")
        st.plotly_chart(fig, width="stretch")

    show_pngs(mode, "ablation")


def page_comparison():
    st.title("Binary vs Multiclass Comparison")
    comp = load_comparison()

    if comp is None:
        # Try loading stage 3 from both modes
        s3_bin = load_stage3("binary")
        s3_multi = load_stage3("multiclass")

        if s3_bin is None or s3_multi is None:
            st.warning("Run both binary and multiclass modes first to see comparison.")
            st.info("Run: `python main.py --mode binary multiclass`")
            return

        # Build comparison on-the-fly
        bin_comp = s3_bin.get("comparison")
        multi_comp = s3_multi.get("comparison")
        if bin_comp is None or multi_comp is None:
            st.warning("No Stage 3 comparison data for both modes.")
            return

        comp = None  # Fall through to dynamic comparison

        model_col = "Model" if "Model" in bin_comp.columns else bin_comp.columns[0]
        for metric in ["Accuracy", "F1_Macro", "Balanced_Acc", "G_Mean"]:
            if metric in bin_comp.columns and metric in multi_comp.columns:
                b = bin_comp[[model_col, metric]].copy()
                b["Mode"] = "Binary"
                m = multi_comp[[model_col, metric]].copy()
                m["Mode"] = "Multiclass"
                combined = pd.concat([b, m], ignore_index=True)
                fig = px.bar(combined, x=model_col, y=metric, color="Mode",
                             barmode="group", title=f"{metric}: Binary vs Multiclass",
                             text_auto=".4f")
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, width="stretch")
        return

    st.dataframe(comp, width="stretch", hide_index=True)

    # Visualization
    if "Metric" in comp.columns:
        for _, row in comp.iterrows():
            metric = row["Metric"]
            vals = {k: float(v) for k, v in row.items() if k != "Metric" and v != "—"}
            if vals:
                fig = px.bar(x=list(vals.keys()), y=list(vals.values()),
                             title=f"{metric} Comparison", text_auto=".4f")
                st.plotly_chart(fig, width="stretch")

    # Show comparison PNGs
    comp_dir = RESULTS_DIR / "comparison"
    if comp_dir.exists():
        pngs = sorted(comp_dir.glob("*.png"))
        if pngs:
            with st.expander("Saved Comparison Plots"):
                cols = st.columns(2)
                for i, png in enumerate(pngs):
                    cols[i % 2].image(str(png), caption=png.stem.replace("_", " ").title(),
                                      width="stretch")


# ==============================================================================
# MAIN APP
# ==============================================================================

def main():
    st.set_page_config(
        page_title="XAI-IDS Benchmark Dashboard",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    pio.templates.default = "plotly_white"

    # Custom CSS
    st.markdown("""
    <style>
        .main .block-container { max-width: 1400px; padding-top: 1.5rem; }
        [data-testid="stMetric"] {
            background: rgba(28, 131, 225, 0.08);
            border: 1px solid rgba(28, 131, 225, 0.2);
            border-radius: 8px;
            padding: 12px 16px;
        }
        h1 { color: #1f77b4; }
        h3 { border-bottom: 1px solid rgba(128,128,128,0.3); padding-bottom: 0.3rem; }
    </style>
    """, unsafe_allow_html=True)

    # ---- Sidebar ----
    st.sidebar.title("🛡️ XAI-IDS Benchmark")
    st.sidebar.markdown("---")
    mode = st.sidebar.radio("Classification Mode", ["multiclass", "binary"])

    pages = {
        "📊 Overview": "overview",
        "📁 1. Dataset & Preprocessing": "dataset",
        "🔍 2. Feature Selection": "features",
        "🤖 3. Model Benchmarking": "models",
        "📈 4. Comprehensive Results": "comprehensive",
        "🧠 5. XAI Quality": "xai",
        "📏 6. Performance Metrics": "performance",
        "🌐 7. Cross-Dataset Generalization": "generalization",
        "🛡️ 8. Robustness Testing": "robustness",
        "📐 9. Statistical Tests": "statistical",
        "📋 10. Final Report": "final_report",
        "❌ Error Analysis": "error_analysis",
        "🔬 Ablation Study": "ablation",
        "⚖️ Binary vs Multiclass": "comparison",
    }

    selected = st.sidebar.selectbox("Navigate to", list(pages.keys()))

    # Stage status
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Stage Status**")
    for i in range(1, 11):
        icon = "✅" if stage_exists(mode, i) else "⬜"
        st.sidebar.text(f"  {icon} Stage {i}")

    # Route to page
    page_id = pages[selected]
    page_map = {
        "overview": lambda: page_overview(mode),
        "dataset": lambda: page_dataset(mode),
        "features": lambda: page_features(mode),
        "models": lambda: page_models(mode),
        "comprehensive": lambda: page_comprehensive(mode),
        "xai": lambda: page_xai(mode),
        "performance": lambda: page_performance(mode),
        "generalization": lambda: page_generalization(mode),
        "robustness": lambda: page_robustness(mode),
        "statistical": lambda: page_statistical(mode),
        "final_report": lambda: page_final_report(mode),
        "error_analysis": lambda: page_error_analysis(mode),
        "ablation": lambda: page_ablation(mode),
        "comparison": lambda: page_comparison(),
    }

    page_map[page_id]()


if __name__ == "__main__":
    main()
