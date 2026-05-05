"""
CIRRUS — Chemical Intelligence & Residual Heat Recovery for Unified Sustainability
===================================================================================
Integrated ML Platform for Semiconductor Fab Resource Recovery
Team 28 | SEMICON Southeast Asia 2026 | TECH Zoomers Challenge

MODEL ARCHITECTURE OVERVIEW
-----------------------------
Subsystem A: Wet Bath Lifecycle Prediction
    Input  : Time-series sensor readings (pH, ORP, ion concentration, turbidity,
             temperature, cumulative lots processed)
    Model  : Gradient Boosting Regressor (ensemble of shallow decision trees,
             trained with staged additive optimization — suitable for tabular
             sensor data with non-linear degradation kinetics)
    Output : Remaining bath life [0–100%] + discrete action trigger
             (CONTINUE / TOP-OFF / DUMP)

Subsystem B: Waste Heat Cascade Routing
    Input  : Exhaust stream telemetry (temperature, flow rate) + fab utility
             demand signals (DI water heater load, HVAC load, ORC capacity)
    Model  : Random Forest Classifier (multi-class, majority-vote ensemble —
             robust to noisy real-time sensor readings, interpretable feature
             importance for fab engineers)
    Output : Optimal routing decision per stream
             (ORC_GENERATION / DI_PREHEAT / HVAC_SUPPLY / BLEND)

Integration Layer:
    - Shared StandardScaler pipeline per subsystem
    - Unified CIRRUS Dashboard: consolidated savings metrics
    - Both models run on the same edge compute inference loop
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error, r2_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import argparse
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = ROOT_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = ROOT_DIR / "data"

BATH_REQUIRED_COLUMNS = [
    "ph", "orp_mv", "ion_ppm", "turbidity", "temp_c",
    "lots_run", "bath_age_hr", "remaining_life_pct"
]

HEAT_REQUIRED_COLUMNS = [
    "exhaust_temp_c", "flow_rate_m3h", "di_demand_kw",
    "hvac_demand_kw", "orc_capacity_pct", "time_of_day_hr", "routing"
]


def parse_args():
    """Parse optional CSV paths for real-data mode."""
    parser = argparse.ArgumentParser(
        description="Run CIRRUS with synthetic data or user-provided CSV files."
    )
    parser.add_argument(
        "--bath-data",
        type=Path,
        default=DATA_DIR / "bath_data.csv",
        help="Path to bath subsystem CSV. Default: data/bath_data.csv",
    )
    parser.add_argument(
        "--heat-data",
        type=Path,
        default=DATA_DIR / "heat_data.csv",
        help="Path to heat subsystem CSV. Default: data/heat_data.csv",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "synthetic", "real"],
        default="auto",
        help=(
            "Data mode: auto (use CSVs if both exist, else synthetic), "
            "synthetic (force generated data), real (require CSVs)."
        ),
    )
    return parser.parse_args()


def _validate_columns(df: pd.DataFrame, required_cols: list[str], dataset_name: str):
    """Raise a readable error when required columns are missing."""
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"{dataset_name} CSV missing required columns: {missing}. "
            f"Expected columns: {required_cols}"
        )


def load_real_data(bath_path: Path, heat_path: Path):
    """Load and validate user-provided bath and heat datasets."""
    if not bath_path.exists() or not heat_path.exists():
        raise FileNotFoundError(
            "Real mode requires both files to exist: "
            f"bath={bath_path}, heat={heat_path}"
        )

    bath_df = pd.read_csv(bath_path)
    heat_df = pd.read_csv(heat_path)

    _validate_columns(bath_df, BATH_REQUIRED_COLUMNS, "Bath")
    _validate_columns(heat_df, HEAT_REQUIRED_COLUMNS, "Heat")

    # Standardize optional action field for consistency in reporting.
    if "action" not in bath_df.columns:
        bath_df["action"] = pd.cut(
            bath_df["remaining_life_pct"],
            bins=[-1, 20, 50, 101],
            labels=["DUMP", "TOP-OFF", "CONTINUE"]
        )

    bath_df = bath_df.copy()
    heat_df = heat_df.copy()
    return bath_df, heat_df


def load_data(mode: str, bath_path: Path, heat_path: Path):
    """Resolve data source based on mode and file availability."""
    if mode == "synthetic":
        print("Generating synthetic fab sensor data...")
        return generate_bath_data(n_samples=1200), generate_heat_data(n_samples=1200), "synthetic"

    if mode == "real":
        print(f"Loading real CSV data from {bath_path} and {heat_path}...")
        bath_df, heat_df = load_real_data(bath_path, heat_path)
        return bath_df, heat_df, "real"

    # auto mode
    if bath_path.exists() and heat_path.exists():
        print(f"Auto mode: detected CSV files at {bath_path} and {heat_path}.")
        bath_df, heat_df = load_real_data(bath_path, heat_path)
        return bath_df, heat_df, "real"

    print("Auto mode: CSV files not found for both subsystems. Falling back to synthetic data.")
    return generate_bath_data(n_samples=1200), generate_heat_data(n_samples=1200), "synthetic"

# =============================================================================
# SECTION 1 — SYNTHETIC DATA GENERATION
# =============================================================================

def generate_bath_data(n_samples: int = 1200) -> pd.DataFrame:
    """
    Synthetic wet bench bath condition dataset.

    Features (inline sensor readings per bath sample):
        ph          : Bath acidity/alkalinity [0–14]
        orp_mv      : Oxidation-Reduction Potential [mV] — proxy for chemical activity
        ion_ppm     : Dissolved ion concentration [ppm] — contaminant buildup indicator
        turbidity   : Optical turbidity [NTU] — particulate contamination
        temp_c      : Bath temperature [°C]
        lots_run    : Cumulative wafer lots processed since last refresh
        bath_age_hr : Hours elapsed since last bath refresh

    Target:
        remaining_life_pct : Remaining usable bath life [0–100%]
        action             : Discrete trigger derived from remaining_life_pct
    """
    lots_run    = np.random.randint(0, 200, n_samples).astype(float)
    bath_age_hr = lots_run * 0.5 + np.random.normal(0, 2, n_samples)
    bath_age_hr = np.clip(bath_age_hr, 0, None)

    # Sensor values degrade non-linearly with bath age and lots processed
    decay = (lots_run / 200) ** 1.3 + (bath_age_hr / 100) ** 1.1

    ph          = 7.0 - 2.5 * decay + np.random.normal(0, 0.15, n_samples)
    orp_mv      = 650 - 300 * decay + np.random.normal(0, 15,  n_samples)
    ion_ppm     = 5   + 120  * decay + np.random.normal(0, 8,   n_samples)
    turbidity   = 0.5 + 40   * decay + np.random.normal(0, 2,   n_samples)
    temp_c      = 70  + np.random.normal(0, 1.5, n_samples)

    ph          = np.clip(ph,        0,    14)
    orp_mv      = np.clip(orp_mv,    100,  800)
    ion_ppm     = np.clip(ion_ppm,   0,    None)
    turbidity   = np.clip(turbidity, 0,    None)

    # Remaining life: 100% at start, degrades toward 0
    remaining_life_pct = np.clip(100 - 50 * decay + np.random.normal(0, 3, n_samples), 0, 100)

    # Derive action trigger from remaining life thresholds
    action = pd.cut(
        remaining_life_pct,
        bins=[-1, 20, 50, 101],
        labels=["DUMP", "TOP-OFF", "CONTINUE"]
    )

    return pd.DataFrame({
        "ph": ph, "orp_mv": orp_mv, "ion_ppm": ion_ppm,
        "turbidity": turbidity, "temp_c": temp_c,
        "lots_run": lots_run, "bath_age_hr": bath_age_hr,
        "remaining_life_pct": remaining_life_pct,
        "action": action
    })


def generate_heat_data(n_samples: int = 1200) -> pd.DataFrame:
    """
    Synthetic thermal exhaust telemetry dataset.

    Features:
        exhaust_temp_c    : Exhaust stream temperature [°C]
        flow_rate_m3h     : Exhaust volumetric flow rate [m³/hr]
        di_demand_kw      : Current DI water heater utility load [kW]
        hvac_demand_kw    : Current HVAC supply load [kW]
        orc_capacity_pct  : Available ORC generator headroom [%]
        time_of_day_hr    : Hour of day [0–23] — fab demand follows shift patterns

    Target:
        routing : Optimal heat routing decision (4-class)
    """
    exhaust_temp_c   = np.random.uniform(100, 1200, n_samples)
    flow_rate_m3h    = np.random.uniform(50,  500,  n_samples)
    di_demand_kw     = np.random.uniform(10,  200,  n_samples)
    hvac_demand_kw   = np.random.uniform(50,  400,  n_samples)
    orc_capacity_pct = np.random.uniform(10,  100,  n_samples)
    time_of_day_hr   = np.random.uniform(0,   23,   n_samples)

    # Routing logic: physics-informed labels
    # High temp + available ORC capacity → ORC generation
    # Mid temp + high DI demand           → DI pre-heat
    # Low temp + high HVAC demand         → HVAC supply
    # Mixed conditions                    → BLEND (split routing)
    routing = []
    for i in range(n_samples):
        t   = exhaust_temp_c[i]
        orc = orc_capacity_pct[i]
        di  = di_demand_kw[i]
        hv  = hvac_demand_kw[i]
        noise = np.random.rand()

        if t > 700 and orc > 50 and noise > 0.1:
            routing.append("ORC_GENERATION")
        elif 300 < t <= 700 and di > 100 and noise > 0.1:
            routing.append("DI_PREHEAT")
        elif t <= 300 and hv > 200 and noise > 0.1:
            routing.append("HVAC_SUPPLY")
        else:
            routing.append("BLEND")

    return pd.DataFrame({
        "exhaust_temp_c": exhaust_temp_c,
        "flow_rate_m3h": flow_rate_m3h,
        "di_demand_kw": di_demand_kw,
        "hvac_demand_kw": hvac_demand_kw,
        "orc_capacity_pct": orc_capacity_pct,
        "time_of_day_hr": time_of_day_hr,
        "routing": routing
    })


# =============================================================================
# SECTION 2 — SUBSYSTEM A: WET BATH LIFECYCLE MODEL
# Architecture: StandardScaler → GradientBoostingRegressor
#   - n_estimators : 200 boosting rounds (bias-variance tradeoff)
#   - max_depth    : 4 (shallow trees prevent overfitting on sensor noise)
#   - learning_rate: 0.05 (conservative step size for stable convergence)
#   - loss          : 'squared_error' (minimizes MSE on continuous life %)
# =============================================================================

def build_bath_model(df: pd.DataFrame):
    """Train and evaluate the bath lifecycle regression model."""
    FEATURES = ["ph", "orp_mv", "ion_ppm", "turbidity", "temp_c",
                "lots_run", "bath_age_hr"]
    TARGET   = "remaining_life_pct"

    X = df[FEATURES].values
    y = df[TARGET].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = Pipeline([
        ("scaler", StandardScaler()),          # Z-score normalisation — equalises
                                               # sensor scale differences (pH 0–14
                                               # vs ORP 100–800 mV)
        ("gbr", GradientBoostingRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,                     # Stochastic gradient boosting:
                                               # 80% row sampling per tree
                                               # reduces variance
            min_samples_leaf=10,               # Regularisation: prevents single
                                               # noisy samples driving splits
            random_state=42
        ))
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("=" * 60)
    print("SUBSYSTEM A — WET BATH LIFECYCLE PREDICTION")
    print("  Model   : Gradient Boosting Regressor (Pipeline)")
    print(f"  Features: {FEATURES}")
    print(f"  Train N : {len(X_train)} | Test N: {len(X_test)}")
    print("-" * 60)
    print(f"  MAE     : {mean_absolute_error(y_test, y_pred):.2f}%  remaining life")
    print(f"  R²      : {r2_score(y_test, y_pred):.4f}")
    print("=" * 60)

    # Feature importance (intrinsic to GBR — no permutation needed)
    gbr      = model.named_steps["gbr"]
    feat_imp = pd.Series(gbr.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print("\n  Feature Importances (Gini-based):")
    for feat, imp in feat_imp.items():
        bar = "█" * int(imp * 40)
        print(f"    {feat:<15} {bar}  {imp:.4f}")

    return model, X_test, y_test, y_pred, feat_imp, FEATURES


def infer_action(remaining_life_pct: float) -> str:
    """Convert continuous life prediction to discrete fab action trigger."""
    if remaining_life_pct <= 20:
        return "DUMP"
    elif remaining_life_pct <= 50:
        return "TOP-OFF"
    else:
        return "CONTINUE"


# =============================================================================
# SECTION 3 — SUBSYSTEM B: HEAT CASCADE ROUTING MODEL
# Architecture: StandardScaler → RandomForestClassifier
#   - n_estimators : 200 trees (ensemble diversity for robust multi-class)
#   - max_depth    : 10 (allows complex routing boundaries without overfitting)
#   - class_weight : 'balanced' (handles imbalanced BLEND class distribution)
#   - criterion    : 'gini' (standard impurity measure for classification)
# =============================================================================

def build_heat_model(df: pd.DataFrame):
    """Train and evaluate the heat routing multi-class classifier."""
    FEATURES = ["exhaust_temp_c", "flow_rate_m3h", "di_demand_kw",
                "hvac_demand_kw", "orc_capacity_pct", "time_of_day_hr"]
    TARGET   = "routing"

    le   = LabelEncoder()
    X    = df[FEATURES].values
    y    = le.fit_transform(df[TARGET].values)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("rfc", RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=5,
            class_weight="balanced",           # Compensates for uneven routing
                                               # class distribution in real fabs
            random_state=42,
            n_jobs=-1                          # Parallel tree building
        ))
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\n" + "=" * 60)
    print("SUBSYSTEM B — HEAT CASCADE ROUTING CLASSIFICATION")
    print("  Model   : Random Forest Classifier (Pipeline)")
    print(f"  Features: {FEATURES}")
    print(f"  Classes : {list(le.classes_)}")
    print(f"  Train N : {len(X_train)} | Test N: {len(X_test)}")
    print("-" * 60)
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    rfc      = model.named_steps["rfc"]
    feat_imp = pd.Series(rfc.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print("  Feature Importances (Mean Decrease Impurity):")
    for feat, imp in feat_imp.items():
        bar = "█" * int(imp * 40)
        print(f"    {feat:<22} {bar}  {imp:.4f}")

    return model, le, X_test, y_test, y_pred, feat_imp, FEATURES


# =============================================================================
# SECTION 4 — INTEGRATION LAYER: CIRRUS INFERENCE LOOP
# Simulates real-time inference on a single fab snapshot — both models
# running on the same edge compute node.
# =============================================================================

def run_cirrus_inference(bath_model, heat_model, le):
    """
    Single-cycle inference demo.
    Mimics the edge compute loop that runs on every sensor poll (e.g. every 60s).
    """
    # Live sensor snapshot — Wet Bench #3 (SPM bath)
    bath_snapshot = {
        "ph": 6.1, "orp_mv": 420, "ion_ppm": 65,
        "turbidity": 18, "temp_c": 71,
        "lots_run": 95, "bath_age_hr": 47
    }

    # Live sensor snapshot — Furnace Bay A exhaust
    heat_snapshot = {
        "exhaust_temp_c": 850, "flow_rate_m3h": 220,
        "di_demand_kw": 145, "hvac_demand_kw": 280,
        "orc_capacity_pct": 72, "time_of_day_hr": 14.5
    }

    bath_X     = np.array([[bath_snapshot[f] for f in
                  ["ph","orp_mv","ion_ppm","turbidity","temp_c","lots_run","bath_age_hr"]]])
    heat_X     = np.array([[heat_snapshot[f] for f in
                  ["exhaust_temp_c","flow_rate_m3h","di_demand_kw",
                   "hvac_demand_kw","orc_capacity_pct","time_of_day_hr"]]])

    life_pct       = bath_model.predict(bath_X)[0]
    action         = infer_action(life_pct)
    routing_enc    = heat_model.predict(heat_X)[0]
    routing_label  = le.inverse_transform([routing_enc])[0]

    # Routing probabilities for uncertainty-awareness display
    routing_proba  = heat_model.predict_proba(heat_X)[0]
    routing_proba_df = pd.Series(routing_proba, index=le.classes_).sort_values(ascending=False)

    print("\n" + "=" * 60)
    print("CIRRUS — REAL-TIME INFERENCE SNAPSHOT")
    print("=" * 60)
    print("\n[Subsystem A] Wet Bench #3 — SPM Bath")
    for k, v in bath_snapshot.items():
        print(f"  {k:<15}: {v}")
    print(f"\n  → Predicted Remaining Life : {life_pct:.1f}%")
    print(f"  → Action Trigger           : *** {action} ***")

    print("\n[Subsystem B] Furnace Bay A — Exhaust Stream")
    for k, v in heat_snapshot.items():
        print(f"  {k:<22}: {v}")
    print(f"\n  → Routing Decision         : *** {routing_label} ***")
    print(f"  → Routing Confidence       :")
    for cls, prob in routing_proba_df.items():
        bar = "█" * int(prob * 30)
        print(f"     {cls:<20} {bar}  {prob:.3f}")

    return life_pct, action, routing_label


# =============================================================================
# SECTION 5 — VISUALISATIONS (Poster / Presentation Quality)
# =============================================================================

def plot_cirrus_dashboard(
    bath_model, bath_X_test, bath_y_test, bath_y_pred, bath_feat_imp,
    heat_model, heat_le, heat_X_test, heat_y_test, heat_y_pred, heat_feat_imp,
    bath_df, life_pct, action, routing_label
):
    fig = plt.figure(figsize=(20, 14), facecolor="#0d1117")
    fig.suptitle(
        "CIRRUS — Integrated ML Resource Recovery Platform\n"
        "Chemical Intelligence & Residual Heat Recovery for Unified Sustainability",
        fontsize=16, fontweight="bold", color="white", y=0.98
    )

    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.4)

    ACCENT_A = "#00d4ff"   # cyan — Subsystem A
    ACCENT_B = "#ff6b35"   # orange — Subsystem B
    BG       = "#161b22"
    TEXT     = "white"

    # ── A1: Bath lifecycle scatter (predicted vs actual) ──────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.set_facecolor(BG)
    ax1.scatter(bath_y_test, bath_y_pred, alpha=0.4, s=12, color=ACCENT_A)
    lims = [0, 100]
    ax1.plot(lims, lims, "w--", lw=1, label="Perfect prediction")
    ax1.set_xlabel("Actual Remaining Life (%)", color=TEXT, fontsize=9)
    ax1.set_ylabel("Predicted Remaining Life (%)", color=TEXT, fontsize=9)
    ax1.set_title("A — Bath Lifecycle: Predicted vs Actual", color=ACCENT_A, fontweight="bold")
    ax1.tick_params(colors=TEXT)
    ax1.legend(fontsize=8, labelcolor=TEXT, facecolor=BG)
    for spine in ax1.spines.values():
        spine.set_edgecolor("#30363d")

    # ── A2: Feature importances (Subsystem A) ─────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_facecolor(BG)
    colors_a = [ACCENT_A if i == 0 else "#4a9eca" for i in range(len(bath_feat_imp))]
    ax2.barh(bath_feat_imp.index[::-1], bath_feat_imp.values[::-1], color=colors_a[::-1])
    ax2.set_title("A — Feature Importance\n(GBR Gini)", color=ACCENT_A, fontweight="bold")
    ax2.tick_params(colors=TEXT, labelsize=8)
    ax2.set_xlabel("Importance", color=TEXT, fontsize=9)
    for spine in ax2.spines.values():
        spine.set_edgecolor("#30363d")

    # ── A3: Bath degradation curve (sample trace) ──────────────────────────────
    ax3 = fig.add_subplot(gs[0, 3])
    ax3.set_facecolor(BG)
    sample = bath_df.sort_values("lots_run").iloc[::12]
    ax3.plot(sample["lots_run"], sample["remaining_life_pct"],
             color=ACCENT_A, lw=1.5, alpha=0.8)
    ax3.axhline(50, color="yellow", lw=1, linestyle="--", label="TOP-OFF threshold")
    ax3.axhline(20, color="red",    lw=1, linestyle="--", label="DUMP threshold")
    ax3.set_xlabel("Lots Processed", color=TEXT, fontsize=9)
    ax3.set_ylabel("Remaining Life (%)", color=TEXT, fontsize=9)
    ax3.set_title("A — Bath Degradation Curve", color=ACCENT_A, fontweight="bold")
    ax3.legend(fontsize=7, labelcolor=TEXT, facecolor=BG)
    ax3.tick_params(colors=TEXT)
    for spine in ax3.spines.values():
        spine.set_edgecolor("#30363d")

    # ── B1: Routing confusion matrix ───────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, :2])
    ax4.set_facecolor(BG)
    cm = confusion_matrix(heat_y_test, heat_y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=heat_le.classes_)
    disp.plot(ax=ax4, colorbar=False, cmap="Blues")
    ax4.set_title("B — Heat Routing: Confusion Matrix", color=ACCENT_B, fontweight="bold")
    ax4.tick_params(colors=TEXT, labelsize=8)
    ax4.xaxis.label.set_color(TEXT)
    ax4.yaxis.label.set_color(TEXT)

    # ── B2: Feature importances (Subsystem B) ─────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.set_facecolor(BG)
    colors_b = [ACCENT_B if i == 0 else "#cc8866" for i in range(len(heat_feat_imp))]
    ax5.barh(heat_feat_imp.index[::-1], heat_feat_imp.values[::-1], color=colors_b[::-1])
    ax5.set_title("B — Feature Importance\n(RF Mean Decrease Impurity)", color=ACCENT_B, fontweight="bold")
    ax5.tick_params(colors=TEXT, labelsize=8)
    ax5.set_xlabel("Importance", color=TEXT, fontsize=9)
    for spine in ax5.spines.values():
        spine.set_edgecolor("#30363d")

    # ── B3: Routing class distribution ────────────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 3])
    ax6.set_facecolor(BG)
    routing_counts = pd.Series(heat_le.inverse_transform(heat_y_test)).value_counts()
    bars = ax6.bar(routing_counts.index, routing_counts.values,
                   color=[ACCENT_B, "#cc8866", "#ff9966", "#ffbb99"])
    ax6.set_title("B — Routing Class Distribution\n(Test Set)", color=ACCENT_B, fontweight="bold")
    ax6.tick_params(colors=TEXT, labelsize=8)
    ax6.set_ylabel("Count", color=TEXT, fontsize=9)
    for spine in ax6.spines.values():
        spine.set_edgecolor("#30363d")
    for bar in bars:
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                 str(int(bar.get_height())), ha="center", color=TEXT, fontsize=8)

    # ── Integration: CIRRUS Live Dashboard ────────────────────────────────────
    ax7 = fig.add_subplot(gs[2, :])
    ax7.set_facecolor(BG)
    ax7.axis("off")

    action_color  = {"CONTINUE": "#00ff88", "TOP-OFF": "yellow", "DUMP": "red"}
    route_colors  = {"ORC_GENERATION": "#ff6b35", "DI_PREHEAT": ACCENT_A,
                     "HVAC_SUPPLY": "#88cc44", "BLEND": "#cc88ff"}

    ax7.text(0.5, 0.92, "CIRRUS UNIFIED SUSTAINABILITY DASHBOARD — Live Inference",
             ha="center", va="top", color="white", fontsize=13, fontweight="bold",
             transform=ax7.transAxes)

    # Subsystem A card
    ax7.add_patch(plt.Rectangle((0.02, 0.05), 0.44, 0.78,
                                transform=ax7.transAxes,
                                facecolor="#1c2128", edgecolor=ACCENT_A, lw=2))
    ax7.text(0.24, 0.75, "SUBSYSTEM A — Wet Bath Lifecycle",
             ha="center", color=ACCENT_A, fontsize=10, fontweight="bold",
             transform=ax7.transAxes)
    ax7.text(0.24, 0.62, f"Remaining Life: {life_pct:.1f}%",
             ha="center", color="white", fontsize=22, fontweight="bold",
             transform=ax7.transAxes)

    # Life bar
    bar_x, bar_y, bar_w, bar_h = 0.05, 0.38, 0.38, 0.08
    ax7.add_patch(plt.Rectangle((bar_x, bar_y), bar_w, bar_h,
                                transform=ax7.transAxes,
                                facecolor="#30363d", edgecolor="none"))
    fill_color = action_color.get(action, "white")
    ax7.add_patch(plt.Rectangle((bar_x, bar_y), bar_w * (life_pct / 100), bar_h,
                                transform=ax7.transAxes,
                                facecolor=fill_color, edgecolor="none", alpha=0.9))
    ax7.text(0.24, 0.28, f"Action: {action}",
             ha="center", color=fill_color, fontsize=14, fontweight="bold",
             transform=ax7.transAxes)
    ax7.text(0.24, 0.16, "Wet Bench #3 — SPM Bath",
             ha="center", color="#8b949e", fontsize=9, transform=ax7.transAxes)

    # Subsystem B card
    ax7.add_patch(plt.Rectangle((0.54, 0.05), 0.44, 0.78,
                                transform=ax7.transAxes,
                                facecolor="#1c2128", edgecolor=ACCENT_B, lw=2))
    ax7.text(0.76, 0.75, "SUBSYSTEM B — Heat Cascade Routing",
             ha="center", color=ACCENT_B, fontsize=10, fontweight="bold",
             transform=ax7.transAxes)
    ax7.text(0.76, 0.62, routing_label.replace("_", " "),
             ha="center", color=route_colors.get(routing_label, "white"),
             fontsize=20, fontweight="bold", transform=ax7.transAxes)
    ax7.text(0.76, 0.45, "Exhaust Temp: 850°C  |  Flow: 220 m³/hr",
             ha="center", color="white", fontsize=10, transform=ax7.transAxes)
    ax7.text(0.76, 0.34, "ORC Headroom: 72%  |  DI Demand: 145 kW",
             ha="center", color="white", fontsize=10, transform=ax7.transAxes)
    ax7.text(0.76, 0.16, "Furnace Bay A — Exhaust Stream",
             ha="center", color="#8b949e", fontsize=9, transform=ax7.transAxes)

    # Divider arrow
    ax7.annotate("", xy=(0.54, 0.44), xytext=(0.46, 0.44),
                 xycoords="axes fraction", textcoords="axes fraction",
                 arrowprops=dict(arrowstyle="->", color="white", lw=2))
    ax7.text(0.5, 0.5, "Edge\nCompute", ha="center", va="center",
             color="white", fontsize=8, transform=ax7.transAxes,
             bbox=dict(facecolor="#30363d", edgecolor="white", boxstyle="round,pad=0.3"))

    output_path = OUTPUTS_DIR / "cirrus_dashboard.png"
    plt.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="#0d1117")
    print(f"\n  Dashboard saved -> {output_path.relative_to(ROOT_DIR)}")
    plt.show()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    args = parse_args()

    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║  CIRRUS — Integrated ML Platform  |  Team 28             ║")
    print("║  SEMICON SEA 2026 TECH Zoomers Challenge                 ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    # Load real data or generate synthetic data
    bath_df, heat_df, data_mode = load_data(args.mode, args.bath_data, args.heat_data)
    print(f"Data mode: {data_mode}")
    print(f"  Bath dataset : {bath_df.shape}  |  Action dist: {bath_df['action'].value_counts().to_dict()}")
    print(f"  Heat dataset : {heat_df.shape}  |  Route dist : {heat_df['routing'].value_counts().to_dict()}\n")

    # Train models
    bath_model, bath_X_test, bath_y_test, bath_y_pred, bath_feat_imp, bath_feats = build_bath_model(bath_df)
    heat_model, heat_le, heat_X_test, heat_y_test, heat_y_pred, heat_feat_imp, heat_feats = build_heat_model(heat_df)

    # Run integrated inference loop
    life_pct, action, routing_label = run_cirrus_inference(bath_model, heat_model, heat_le)

    # Plot unified dashboard
    print("\nRendering CIRRUS dashboard...")
    plot_cirrus_dashboard(
        bath_model, bath_X_test, bath_y_test, bath_y_pred, bath_feat_imp,
        heat_model, heat_le, heat_X_test, heat_y_test, heat_y_pred, heat_feat_imp,
        bath_df, life_pct, action, routing_label
    )

    print("\n✓ CIRRUS model run complete.")
