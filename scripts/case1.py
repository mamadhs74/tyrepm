import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.inspection import permutation_importance
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import re
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from collections import defaultdict
from pathlib import Path                   
from torch.nn.utils import clip_grad_norm_  
import torch.nn.functional as F  # for SmoothL1 (Huber) loss                    

try:
    from filterpy.kalman import ExtendedKalmanFilter
    HAS_FILTERPY = True
except Exception:
    HAS_FILTERPY = False

try:
    import pyproj
    HAS_PYPROJ = True
except Exception:
    HAS_PYPROJ = False

try:
    import contextily as ctx
    HAS_CTX = True
except Exception:
    HAS_CTX = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:
    HAS_LGB = False

try:
    import optuna
    HAS_OPTUNA = True
except Exception:
    HAS_OPTUNA = False

# Optional switch for basemaps (needs internet & contextily)
USE_BASEMAPS = False

# ====== NaN/Inf tripwires (add once, near imports) ======
from contextlib import contextmanager

def register_nan_hooks(module: torch.nn.Module, name="model"):
    def _fwd_hook(mod, inp, out):
        def _check(t):
            return isinstance(t, torch.Tensor) and (not torch.isfinite(t).all())
        bad_in = any(_check(t) for t in (inp if isinstance(inp, (tuple, list)) else [inp]))
        bad_out = any(_check(t) for t in (out if isinstance(out, (tuple, list)) else [out]))
        if bad_in or bad_out:
            raise RuntimeError(f"[NaNGuard] Non-finite in {name}:{mod.__class__.__name__} (forward)")
    def _bwd_hook(mod, grad_input, grad_output):
        def _check(t):
            return t is not None and isinstance(t, torch.Tensor) and (not torch.isfinite(t).all())
        bad = any(_check(t) for t in grad_input) or any(_check(t) for t in grad_output)
        if bad:
            raise RuntimeError(f"[NaNGuard] Non-finite gradients in {name}:{mod.__class__.__name__} (backward)")
    for m in module.modules():
        m.register_forward_hook(_fwd_hook)
         # m.register_full_backward_hook(_bwd_hook)  # noisy + slow; rely on anomaly_guard()


@contextmanager
def anomaly_guard():
    old = torch.is_anomaly_enabled()
    torch.autograd.set_detect_anomaly(True)
    try:
        yield
    finally:
        torch.autograd.set_detect_anomaly(old)
# ========================================================
# ====================== SANITY & DIAGNOSTICS UTILITIES ======================

def _align_on_index(y, x):
    y = y.copy().reset_index(drop=True)
    x = x.copy().reset_index(drop=True)
    return y, x

def compute_persistence_and_rolling(y_test: pd.Series, test_idx: np.ndarray, window:int=5):
    y = y_test.iloc[test_idx].copy().reset_index(drop=True)
    y_pred_persist = y.shift(1)
    y_pred_roll = y.rolling(window=window, min_periods=window).mean()
    df = pd.DataFrame({"y": y, "persist": y_pred_persist, "roll": y_pred_roll}).dropna()
    if len(df) == 0:
        return np.nan, np.nan, 0
    mae_persist = np.abs(df["y"] - df["persist"]).mean()
    mae_roll = np.abs(df["y"] - df["roll"]).mean()
    return float(mae_persist), float(mae_roll), int(len(df))

def run_permutation_test(X: pd.DataFrame, y: pd.Series, tr_idx: np.ndarray, te_idx: np.ndarray, seed: int = 13):
    Xr = X.reset_index(drop=True)
    yr = y.reset_index(drop=True)
    rs = np.random.RandomState(seed)
    y_perm = yr.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    def _fit_predict(X_train, y_train, X_test):
        gb = lgb.LGBMRegressor(n_estimators=300, random_state=seed)
        gb.fit(X_train, y_train)
        return gb.predict(X_test)

    y_pred_real = _fit_predict(Xr.iloc[tr_idx], yr.iloc[tr_idx], Xr.iloc[te_idx])
    y_pred_perm = _fit_predict(Xr.iloc[tr_idx], y_perm.iloc[tr_idx], Xr.iloc[te_idx])

    y_true = yr.iloc[te_idx].to_numpy()
    mae_real = np.mean(np.abs(y_true - y_pred_real))
    mae_perm = np.mean(np.abs(y_true - y_pred_perm))
    return float(mae_real), float(mae_perm)

def lookahead_smoke_checks(X: pd.DataFrame, y: pd.Series, tr_idx: np.ndarray, te_idx: np.ndarray, K:int=5, seed:int=42):
    Xr = X.reset_index(drop=True).copy()
    yr = y.reset_index(drop=True).copy()

    # (A) BAD: shift features forward +K (should degrade)
    X_bad = Xr.shift(+K)
    valid_mask = X_bad.notna().all(axis=1)
    tr = np.array([i for i in tr_idx if valid_mask.iloc[i]])
    te = np.array([i for i in te_idx if valid_mask.iloc[i]])
    gb_bad = lgb.LGBMRegressor(n_estimators=600, random_state=seed)
    gb_bad.fit(X_bad.iloc[tr], yr.iloc[tr])
    yhat_bad = gb_bad.predict(X_bad.iloc[te])
    mae_bad = np.mean(np.abs(yr.iloc[te].to_numpy() - yhat_bad))

    # (B) CHEAT: leak future target (should improve a lot)
    X_cheat = Xr.copy()
    X_cheat["target_t+K"] = yr.shift(-K)
    valid_mask2 = X_cheat["target_t+K"].notna()
    tr2 = np.array([i for i in tr_idx if valid_mask2.iloc[i]])
    te2 = np.array([i for i in te_idx if valid_mask2.iloc[i]])
    gb_cheat = lgb.LGBMRegressor(n_estimators=600, random_state=seed)
    gb_cheat.fit(X_cheat.iloc[tr2], yr.iloc[tr2])
    yhat_cheat = gb_cheat.predict(X_cheat.iloc[te2])
    mae_cheat = np.mean(np.abs(yr.iloc[te2].to_numpy() - yhat_cheat))

    return float(mae_bad), float(mae_cheat), int(len(te)), int(len(te2))

def lofo_mae(model, X: pd.DataFrame, y: pd.Series, te_idx: np.ndarray, feature_names: list):
    Xr = X.reset_index(drop=True)
    yr = y.reset_index(drop=True)
    y_true = yr.iloc[te_idx].to_numpy()
    yhat_base = model.predict(Xr.iloc[te_idx])
    mae_base = np.mean(np.abs(y_true - yhat_base))
    results = []
    for f in feature_names:
        X_abl = Xr.copy()
        X_abl[f] = X_abl[f].mean()  # hard ablation (mean-impute)
        yhat = model.predict(X_abl.iloc[te_idx])
        mae = np.mean(np.abs(y_true - yhat))
        results.append((f, float(mae - mae_base)))
    return sorted(results, key=lambda t: t[1], reverse=True), float(mae_base)
# ============================================================================ 


def safe_cols(cols):
    """
    Replace every character LightGBM dislikes by '_'.
    Keep only letters, digits and underscores.
    """
    return [re.sub(r'[^A-Za-z0-9_]', '_', c) for c in cols]

def rmse(y_true, y_pred):
    """Root-mean-square error with no ‘squared=’ kwarg required."""
    return np.sqrt(mean_squared_error(y_true, y_pred))




# ----------------------------------------------------------------------
#  MONOTONE-CONSTRAINT SAFE HELPERS  (drop-in replacement)
# ----------------------------------------------------------------------
def build_monotone_list(feature_names):
    """
    Return +1/-1/0 per column, matching the ORIGINAL CSV column names.
    """
    mono_map = {
        "KO1_[km\\h]"           : +1,  # Speed ↑ → PM ↑
        "Pitch_Angle_[Grad]"    : +1,
        "a_long"                : 0,
        "BR2_Querbeschl_[m/s²]" : +1,
    }
    return [mono_map.get(c, 0) for c in feature_names]



BEST_LGB_PARAMS_BY_TGT = {}  # target -> best param dict


    # chronological split
def make_lgb_fast(feature_names, device="cpu", params_override=None):
    base = dict(
        device_type="cpu",
        objective="regression",
        n_estimators=20_000,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=0.2,
        monotone_constraints=build_monotone_list(feature_names),
        random_state=42,
        n_jobs=-1,
        max_depth=-1,
        verbosity=-1,
    )
    if params_override:
        base.update(params_override)
        base["objective"] = "regression"
        base["device_type"] = device
        base["random_state"] = 42
        base["n_jobs"] = -1
        base["monotone_constraints"] = build_monotone_list(feature_names)
    return lgb.LGBMRegressor(**base)

if not HAS_LGB:
    # Safe stubs so later calls fail with a clear message instead of NameError
    class _LGBMissing(RuntimeError): pass
    def make_lgb_fast(*args, **kwargs):
        raise _LGBMissing("lightgbm is required for the GBM parts of the pipeline.")
    def lgb_fit(*args, **kwargs):
        raise _LGBMissing("lightgbm is required for the GBM parts of the pipeline.")
else:
    def lgb_fit(X, y, device="cpu", tgt=None):
        # mask non-finite targets
        m = np.isfinite(y.to_numpy(dtype=float))
        X = X.loc[m].copy()
        y = y.loc[m].copy()
        if len(y) < 20:
            raise ValueError("Not enough labeled rows for LightGBM fit.")

        # chronological split
        val_split = int(0.8 * len(X))
        X_tr_orig = X.iloc[:val_split].copy()   # ORIGINAL NAMES
        X_val_orig= X.iloc[val_split:].copy()
        y_tr      = y.iloc[:val_split].copy()
        y_val     = y.iloc[val_split:].copy()

        # drop constant/all-NaN columns on THIS split (still original names)
        non_const = X_tr_orig.columns[X_tr_orig.nunique(dropna=False) > 1]
        X_tr_orig = X_tr_orig[non_const]
        X_val_orig= X_val_orig[non_const].copy()
        if X_tr_orig.shape[1] == 0:
            raise ValueError("All features are constant in this split. Check preprocessing.")
        if pd.Series(y_tr).nunique() <= 1:
            raise ValueError("Training target is constant in this split; cannot fit LightGBM.")

        # build monotone constraints BEFORE sanitizing
        mono = build_monotone_list(list(X_tr_orig.columns))

        # sanitize AFTER pruning (mirror columns on val)
        X_tr = X_tr_orig.copy(); X_tr.columns = safe_cols(X_tr.columns)
        X_val= X_val_orig.copy(); X_val.columns = X_tr.columns

        model = make_lgb_fast(
            X_tr.columns, device=device,
            params_override=BEST_LGB_PARAMS_BY_TGT.get(tgt)
        )
        model.set_params(monotone_constraints=mono)

        early_stop = lgb.early_stopping(1000, first_metric_only=True, verbose=False)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric="l2",
            callbacks=[early_stop]
        )

        # return a sanitized view aligned to what the model saw (for permutation_importance)
        X_safe = X[non_const].copy()
        X_safe.columns = safe_cols(X_safe.columns)
        return model, X_safe



class Chomp1d(nn.Module):
    """Remove padding introduced by causal convolution."""
    def __init__(self, chomp):
        super().__init__()
        self.chomp = chomp
    def forward(self, x):
        return x[:, :, :-self.chomp].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_in, n_out, k, dil, p):
        super().__init__()
        self.conv1 = nn.Conv1d(n_in, n_out, k, padding=(k-1)*dil, dilation=dil)
        self.chomp1 = Chomp1d((k-1)*dil)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(p)

        self.conv2 = nn.Conv1d(n_out, n_out, k, padding=(k-1)*dil, dilation=dil)
        self.chomp2 = Chomp1d((k-1)*dil)
        self.act2 = nn.ReLU()
        self.drop2 = nn.Dropout(p)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.act1, self.drop1,
                                 self.conv2, self.chomp2, self.act2, self.drop2)
        self.down = nn.Conv1d(n_in, n_out, 1) if n_in != n_out else nn.Identity()
        self.act  = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        return self.act(out + self.down(x))

class TemporalConvNet(nn.Module):
    def __init__(self, n_inputs, n_channels, k=3, p=0.1):
        super().__init__()
        layers = []
        for i, n_out in enumerate(n_channels):
            dil = 2 ** i
            n_in = n_inputs if i == 0 else n_channels[i-1]
            layers.append(TemporalBlock(n_in, n_out, k, dil, p))
        self.tcn = nn.Sequential(*layers)

    def forward(self, x):
        return self.tcn(x)

class TCN(nn.Module):
    """Drop-in replacement for `from tcn import TCN` (PyTorch version)."""
    def __init__(self, input_size, output_size, num_channels,
                 kernel_size=3, dropout=0.1):
        super().__init__()
        self.tcn = TemporalConvNet(input_size, num_channels,
                                   k=kernel_size, p=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        # x: (B, T, F)  →  TCN expects (B, F, T)
        y = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        return self.linear(y[:, -1, :])          # last time step
    





# === INSERT HERE (right after class TCN) =========================
class TCNEncoder(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size=5, dropout=0.15):
        super().__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, k=kernel_size, p=dropout)
        self.outdim = num_channels[-1]
    def forward(self, x):                 # x: (B,T,F)
        h = self.tcn(x.transpose(1,2)).transpose(1,2)  # (B,T,C)
        return h[:, -1, :]                # last step embedding (B,C)

class SSLHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
    def forward(self, z):
        return self.linear(z)             # predicts next-step feature vector

class FeatureGate(nn.Module):
    def __init__(self, n_feat, init=1.0):
        super().__init__()
        self.log_g = nn.Parameter(torch.full((n_feat,), np.log(init), dtype=torch.float32))
    def forward(self, x):                 # x: (B,T,F)
        g = torch.nn.functional.softplus(self.log_g)   # positive weights
        return x * g[None, None, :], g

class PMHead(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)
    def forward(self, z):                 # z: (B,C)
        return self.fc(z).squeeze(-1)

class PMModel(nn.Module):
    """Gated encoder + PM head for supervised fine-tuning (+ SSL aux)."""
    def __init__(self, n_feat, channels, kernel_size=5, dropout=0.15):
        super().__init__()
        self.gate    = FeatureGate(n_feat)
        self.encoder = TCNEncoder(n_feat, channels, kernel_size, dropout)
        self.head    = PMHead(self.encoder.outdim)          # PM regressor
        self.ssl_aux = SSLHead(self.encoder.outdim, n_feat) # reconstruct features (SSL)

    def forward(self, x, return_ssl=False):
        xg, g = self.gate(x)
        z     = self.encoder(xg)
        yhat  = self.head(z)
        if return_ssl:
            recon = self.ssl_aux(z)
            return yhat, g, recon
        return yhat, g

# ================================================================

# ────────────────────────────────────────────────────────────────────────────



WIN          = 256          # sequence length (was 1024 → halves GPU load)
BATCH_SIZE   = 12          # DataLoader batches (was 32)
CHANNELS     = [64]*6      # TCN depth (was 8 layers)

EPOCHS = 50  
# ---- Diagnostics controls ----
ENABLE_GRAD_PLOTS = True       # set False to skip plotting entirely
# ---- One-switch preset for extra stability during bring-up ----
STABLE_MODE = False   # ← flip to False to go back to your normal settings
# === [NEW] SSL experiment config ===
EXPERIMENT_SSL = True   # False = disable SSL (baseline). True = enable SSL.
SEED = 42               # run the script with SEED=41,42,43 and compare
MASK_RATE = 0.20        # % of feature dims to mask in SSL losses
# ---- Augmentation knobs (z-scored feature space) ----
AUG_NOISE_STD  = 0.02   # small Gaussian noise on a few features
KO1_JITTER_STD = 0.05   # extra jitter ONLY on KO1_[km\h]
JITTER_KO1     = True   # quick on/off switch


# --- augmentation schedule helper (prevents Pylance "aug_scale" warning) ---
def aug_scale(epoch: int, total: int = EPOCHS) -> float:
    """Return multiplicative factor for augmentation intensity.
    Keep 1.0 for now; replace with a schedule later if you like."""
    return 1.0
# gentler when stabilizing
if STABLE_MODE:
    AUG_NOISE_STD  = 0.00
    KO1_JITTER_STD = 0.02

# features eligible for the generic noise
AUG_COLS = ["KO1_[km\\h]","Pitch_Angle_[Grad]","BR5_Bremsdruck_[bar]",
            "Heading_rate_[deg_s]","Curvature_[1_m]"]


if STABLE_MODE:
    # Safer sequence/windowing during stabilization
    WIN = 128                              # shorter windows ease optimization
    BATCH_SIZE = max(8, BATCH_SIZE)        # keep at least 8
    # Turn off data augmentation (guard used later in the training loop)
    AUG_DISABLED = True
else:
    AUG_DISABLED = False

MAX_GRAD_SERIES   = 25         # plot at most this many parameter series
GRAD_LOG_EVERY    = 10         # record one grad point every N steps
TRACKED_PARAM_PATTERNS = [
    r"__global__",
    r"\bgate\.log_g\b",
    r"\bencoder\.tcn\.tcn\.0\.conv1\.weight\b",
]

# (optional but recommended) reproducibility
import random
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("🔌  using", DEVICE.upper())

# ——————————————————————————————————————— 1. load data
df = pd.read_csv("1.csv")
TICK_RATE            = 0.1     # logger tick [s]
# Ensure a time axis exists for downstream slicing/plots
if "X_Achse_[s]" not in df.columns:
    df["X_Achse_[s]"] = np.arange(len(df), dtype=float) * TICK_RATE

DRY_ASPHALT_BASE_MU  = 1.2
USE_DYNAMIC_MU       = False   # still honoured for temp-μ, slip-μ always applied




# ——————————————————————————————————————— constants & folders
for sub in ["feature_importance",
            "friction_vs_emission",
            "prediction",
            "diagnostics"
            ]:        
    os.makedirs(f"ml_charts/{sub}", exist_ok=True)


print("μ mode →", "dynamic" if USE_DYNAMIC_MU else "constant 1.2")
if "DSN" not in df.columns:
    df["DSN"] = 0          # fills with zero so later code never crashes
# lateral g  → m/s²
if "BR2_Querbeschl_[g]" in df.columns and "BR2_Querbeschl_[m/s²]" not in df.columns:
    df["BR2_Querbeschl_[m/s²]"] = df["BR2_Querbeschl_[g]"] * 9.80665

# instantaneous fuel use → L/s and L/km
if "MO5_Verbrauch_[ul]" in df.columns:
    df["Fuel_Rate_Lps"] = df["MO5_Verbrauch_[ul]"] * 1e-5
    if "KO1_[km\\h]" in df.columns:
        dist_km_tick = df["KO1_[km\\h]"] * TICK_RATE / 3600
        df["Fuel_per_km"] = np.where(dist_km_tick > 0,
                                     (df["MO5_Verbrauch_[ul]"] * 1e-6) / dist_km_tick,
                                     0.0)
else:
    df["Fuel_Rate_Lps"] = 0.0
    df["Fuel_per_km"]   = 0.0

FEATURE_UNITS = {
    "a_long": "m/s²",
    "KO1_[km\\h]": "km/h",
    "BR2_Querbeschl_[m/s²]": "m/s²",
    "BR5_Bremsdruck_[bar]": "bar",
    "Pitch_Angle_[Grad]": "°",
    "Slip_Angle_[Grad]": "°",
    "BR5_Giergeschw_[Grad\\s]": "°/s",
    "DSN": "",
    "Friction_Force_Dynamic_[N]": "N",
    "Friction_Utilization": "",
    "Slip_Rate": "°/s",
    "Mech_Power": "kW",
    "Probe_Height_[m]": "m",
    "slip_x_speed": "°·km/h",
    "slip_energy": "kJ",
    "Slip_Ratio_Avg": "",
    "mu_final": "",
    "Fuel_Rate_Lps": "L/s",
    "Fuel_per_km": "L/km",
    "Torque_Utilization": "",
    "Power_Estimate": "kW",
    "Steering_Power": "",
    "BR8_Laengsbeschl_[m\\s2]": "m/s²"
}
# ——————————————————————————————————————— 3. model inputs
FEATURES = [
    "a_long","KO1_[km\\h]","BR2_Querbeschl_[m/s²]","BR5_Bremsdruck_[bar]",
    "Pitch_Angle_[Grad]","Slip_Angle_[Grad]","BR5_Giergeschw_[Grad\\s]",
    "DSN","Friction_Force_Dynamic_[N]","Friction_Utilization","Slip_Rate",
    "Mech_Power","Probe_Height_[m]","slip_x_speed","slip_energy",
    "Slip_Ratio_Avg","mu_final","Fuel_Rate_Lps","Fuel_per_km",
    "Torque_Utilization","Power_Estimate","Steering_Power"
]

SELECTED = [
    "KO1_[km\\h]",
    "Pitch_Angle_[Grad]",
    "a_long",
    "BR2_Querbeschl_[m/s²]",
    "Slip_Angle_[Grad]",
    "slip_x_speed",
    "BR5_Giergeschw_[Grad\\s]",
    "BR5_Bremsdruck_[bar]",
    "Slip_Rate",
    
]

# ④ Register new features
EXTRA_FEATS = ["Heading_rate_[deg_s]", "Curvature_[1_m]"]
FEATURES  += EXTRA_FEATS
SELECTED += EXTRA_FEATS
FEATURE_UNITS.update({"Heading_rate_[deg_s]": "°/s", "Curvature_[1_m]": "1/m"})


# === Define model feature list once (after SELECTED + EXTRA_FEATS are final)
feat_cols_for_model = list(SELECTED)




TARGETS = ["PM1_shifted","PM2_5_shifted","PM10_shifted"]
PRETTY  = {"PM1_shifted":"PM₁","PM2_5_shifted":"PM₂.₅","PM10_shifted":"PM₁₀"}
UNIT    = "µg/m³"

# --- ENGINE TORQUE UTILIZATION & POWER ESTIMATION ---

df["Torque_Utilization"]   = df.get("MO6_Ist_Moment_[Prozent]", 0.0) / 100.0
df["Actual_Torque_[Nm]"]   = df["Torque_Utilization"] * df.get("MO5_max_Moment_[Nm]", 0.0)
df["Engine_Omega_[rad_s]"] = df.get("MO1_Drehzahl_[1\\min]", 0.0).fillna(0) * (2*np.pi/60)

# 4. Estimated engine power output in **kW**
df["Power_Estimate"] = (
    df["Actual_Torque_[Nm]"] *
    df["Engine_Omega_[rad_s]"] / 1000    # kW
)




# ────────── GPS conversion ──────────────────────────────────────────────────
# ────────── GPS conversion (guarded) ────────────────────────────────────────
def ddmm_to_deg(series):
    deg  = np.floor(series / 100)
    mins = series - deg*100
    return deg + mins/60

def smart_geo(series):
    return (ddmm_to_deg(series) if series.abs().max() > 180 else series).astype(float)

if {"Latitude", "Longitude"}.issubset(df.columns):
    df["Latitude_[deg]"]  = smart_geo(df["Latitude"])
    df["Longitude_[deg]"] = smart_geo(df["Longitude"])

    for col in ["Latitude_[deg]", "Longitude_[deg]"]:
        s = df[col].replace(0, np.nan).ffill()              # no bfill (no future peeking)
        df[col] = s.ewm(alpha=0.5, adjust=False).mean()


    # WGS84 → Web-Mercator
    if HAS_PYPROJ:
        fwd = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        df["X_m"], df["Y_m"] = fwd.transform(
            df["Longitude_[deg]"].values,
            df["Latitude_[deg]"].values
        )
    else:
        lat = np.deg2rad(df["Latitude_[deg]"].astype(float).to_numpy())
        lon = np.deg2rad(df["Longitude_[deg]"].astype(float).to_numpy())
        R   = 6371000.0
        lat0 = lat[0]; lon0 = lon[0]
        df["X_m"] = R * (lon - lon0) * np.cos(lat0)
        df["Y_m"] = R * (lat - lat0)

    # robust 2D speed
    dx = df["X_m"].diff().clip(-15, 15).fillna(0.0)
    dy = df["Y_m"].diff().clip(-15, 15).fillna(0.0)
    df["V_xy_ms"] = np.hypot(dx, dy) / TICK_RATE
    df["V_xy_ms"] = df["V_xy_ms"].ewm(alpha=0.5, adjust=False).mean()

    # prefer dedicated GPS velocity if present
    if "Velocity_[km\\h]" in df.columns:
        df["V_gps_ms"] = (df["Velocity_[km\\h]"].replace(0, np.nan) / 3.6)
        df["V_gps_ms"] = df["V_gps_ms"].ewm(alpha=0.5, adjust=False).mean()
        df["V_gps_ms"] = df["V_gps_ms"].fillna(df["V_xy_ms"])
    else:
        df["V_gps_ms"] = df["V_xy_ms"]

    # heading
    if "Heading_deg" not in df.columns:
        if "Heading" in df.columns:
            df["Heading_deg"] = df["Heading"].astype(float)
        elif "True_Heading_[Grad]" in df.columns:
            df["Heading_deg"] = df["True_Heading_[Grad]"].astype(float)
        else:
            df["Heading_deg"] = np.rad2deg(np.arctan2(dy, dx))
            df["Heading_deg"] = df["Heading_deg"].bfill().fillna(0.0)
    df["Heading_unwrapped"] = np.unwrap(np.deg2rad(df["Heading_deg"]))

    # EKF / fallback smoothing
    dt = TICK_RATE

    def _fallback_path_smoothing():
        df["X_f"] = df["X_m"].rolling(5, min_periods=1).median()
        df["Y_f"] = df["Y_m"].rolling(5, min_periods=1).median()
        dx = df["X_f"].diff().fillna(0.0); dy = df["Y_f"].diff().fillna(0.0)
        df["V_f"] = np.hypot(dx, dy) / dt
        psi_unwrapped = np.unwrap(np.deg2rad(df["Heading_deg"].astype(float).to_numpy()))
        df["Psi_f"] = pd.Series(psi_unwrapped, index=df.index)
        df["r_f"]   = df["Psi_f"].diff().fillna(0.0) / dt    # rad/s, causal


    try:
        if HAS_FILTERPY:
            # TODO: drop in your EKF here and assign the 5 columns below.
            # Until then, keep behavior consistent by using the fallback:
            _fallback_path_smoothing()
        else:
            _fallback_path_smoothing()
    except Exception as e:
        print("[EKF] failed, using fallback:", e)
        _fallback_path_smoothing()


    # back to lat/lon
    if HAS_PYPROJ:
        inv = pyproj.Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
        df["Lon_filt"], df["Lat_filt"] = inv.transform(df["X_f"].values, df["Y_f"].values)
    else:
        lat0 = np.deg2rad(df["Latitude_[deg]"].iloc[0])
        R = 6371000.0
        df["Lat_filt"] = np.rad2deg(df["Y_f"]/R + lat0)
        df["Lon_filt"] = np.rad2deg(df["X_f"]/(R*np.cos(lat0)) + np.deg2rad(df["Longitude_[deg]"].iloc[0]))

    residual_m = np.hypot(df["X_f"] - df["X_m"], df["Y_f"] - df["Y_m"])
    print(f"[EKF] median residual  : {np.median(residual_m):5.2f} m")
    print(f"[EKF] 95-percentile    : {np.percentile(residual_m,95):5.2f} m")
    print(f"[EKF] worst (max) error: {residual_m.max():5.2f} m")

else:
    # No GPS → create safe placeholders so later code/plots don’t crash
    df["Latitude_[deg]"]  = 0.0
    df["Longitude_[deg]"] = 0.0
    df["X_m"] = df["Y_m"] = 0.0
    df["V_xy_ms"] = 0.0
    df["V_gps_ms"] = df.get("V_gps_ms", pd.Series(0.0, index=df.index))
    df["Heading_deg"] = 0.0
    df["Heading_unwrapped"] = 0.0
    df["X_f"] = df["Y_f"] = 0.0
    df["V_f"] = 0.0
    df["Psi_f"] = 0.0
    df["r_f"] = 0.0
    df["Lat_filt"] = df["Latitude_[deg]"]
    df["Lon_filt"] = df["Longitude_[deg]"]
    print("[GPS] No Latitude/Longitude in CSV → using zeroed placeholders.")
# ────────────────────────────────────────────────────────────────────────────

# ────────── heading-rate & curvature (Yaw_Rate_DA zeros – recreate) ─────────

# ───────── heading-rate & curvature (GPS-only speed) ───────────────────────

# 1) Heading-rate [deg / s]  – as you already had
df["Heading_rate_[deg_s]"] = (
    np.rad2deg(df["Heading_unwrapped"].diff().fillna(0)) / TICK_RATE
).ewm(alpha=0.5, adjust=False).mean()


# 2) Pure-GPS speed series  (km/h ➜ m/s)  with tiny-gap interpolation
df["V_gps_ms"] = df["V_gps_ms"].replace(0, np.nan).ffill().fillna(0.0)


# 3) Curvature κ = yaw-rate / speed   (guard against div-by-0)
# Causal curvature κ = yaw_rate / speed
df["Curvature_[1_m]"] = (
    df["r_f"] / df["V_gps_ms"].clip(lower=0.1)
).replace([np.inf, -np.inf], 0.0)

#--------------------------------------------------------------------------------------------------
def plot_prediction_diagnostics(y_true, y_pred, time,
                                 tgt_name, unit, df, features):
    """
    Two-panel diagnostics:
      1. Time series of actual vs predicted
      2. Hex-bin of Predicted vs Actual with 45° reference
    """

    # ------------------------------------------------------------------ setup
    residuals = y_true - y_pred    
    # ------------------------------------------------------------------
    # 1.  ── CREATE A *ONE-COLUMN* GRIDSPEC (2 rows × 1 column)
    # ------------------------------------------------------------------
    # ── figure & grid
    fig = plt.figure(figsize=(10, 8), dpi=200, constrained_layout=True)
    gs = fig.add_gridspec(        # <-- this line was missing
            2, 1,                 # 2 rows, 1 column
            hspace=0.30
        )


    # ── top: time-series
    ax_ts = fig.add_subplot(gs[0, 0])
    ax_ts.plot(time, y_true, label='Actual', alpha=0.7)
    ax_ts.plot(time, y_pred, label='Predicted', alpha=0.7)
    ax_ts.set_title(f'{tgt_name} – time series (R²={r2_score(y_true,y_pred):.2f})')
    ax_ts.set_ylabel(f'{tgt_name} [{unit}]')
    ax_ts.legend()

    # ── bottom: hexbin density
    ax_pa = fig.add_subplot(gs[1, 0])
    hb = ax_pa.hexbin(
        y_true, y_pred,
        gridsize=60,
        cmap='viridis',
        mincnt=1          # draw hexes only where there’s data
        # (no density=… here)
    )
    fig.colorbar(hb, ax=ax_pa, label='raw count')
    ax_pa.set_xlabel('Actual')
    ax_pa.set_ylabel('Predicted')
    ax_pa.set_title('Prediction vs Actual')

        # ------------------------------------------------------------------
        # 4.  ── DONE  (no “holder/inner/drivers” block, because we now have only two panels)
        # ------------------------------------------------------------------
    return fig

# ——————————————————————————————————————— 

# ---- global safety: clip z-scored features to ±6σ ----
def clip_features(arr, clip=6.0):
    return np.clip(arr, -clip, clip)


# ——————————————————————————————————————— 2. original feature engineering
# 2-a  basic kinematics -------------------------------------------------------
if "BR8_Laengsbeschl_[m\\s2]" in df.columns:
    df["a_long"] = df["BR8_Laengsbeschl_[m\\s2]"].fillna(0)
else:
    df["a_long"] = df["KO1_[km\\h]"].diff().fillna(0) / (3.6 * TICK_RATE)

    
for lag in (1, 2):
    df[f"KO1_[km\\h]_lag{lag}"] = df["KO1_[km\\h]"].shift(lag).ffill().fillna(0.0)
    df[f"a_long_lag{lag}"]      = df["a_long"].shift(lag).ffill().fillna(0.0)

df["a_lat"] = df.get("BR2_Querbeschl_[m/s²]", pd.Series(0, index=df.index))

if "LW1_Lenk_Gesch_[Grad\\s]" in df.columns:
    df["Steering_Power"] = np.abs(df["LW1_Lenk_Gesch_[Grad\\s]"] * df["a_lat"])

for loc in ["Front", "Rear"]:
    l = f"BR3_Rad_V{'L' if loc=='Front' else 'H'}L_[km\\h]"
    r = f"BR3_Rad_V{'L' if loc=='Front' else 'H'}R_[km\\h]"
    if l in df.columns and r in df.columns:
        df[f"Wheel_Speed_Diff_{loc}"] = df[l] - df[r]

# 2-b  physics block – load transfer, μ, friction force ----------------------
G             = 9.80665
tire_radius   = (457.2 + 2*114.75)/2000           # ~0.343 m for 255/45 R18
probe_height  = 0.13                              # PM probe height [m]
weights       = {"Front_Left":681, "Front_Right":630,
                 "Rear_Left":464.5, "Rear_Right":484}
vehicle_mass  = sum(weights.values())
L             = 3.0                               # wheel-base [m]
x_cg = ((weights["Rear_Left"] + weights["Rear_Right"]) / vehicle_mass) * L
H_CG_DEFAULT = 0.62   # tweak to 0.65–0.70 m when the van is loaded / roof rack, etc.
h_cg = float(H_CG_DEFAULT)
static_FR     = weights["Front_Right"] * G
FR_share      = weights["Front_Right"]/(weights["Front_Left"]+weights["Front_Right"])

# --- Front-right dynamic load (warning-proof version) ---

# measured front track (m) — no arbitrary +0.06
track_front = (1622.0 + 1634.0) / 2000.0  # ≈ 1.628

# ensure clean float Series
a_x = pd.to_numeric(df["a_long"], errors="coerce").astype(float)
a_y = pd.to_numeric(df["a_lat"],  errors="coerce").astype(float)

# total longitudinal transfer (positive a_x = accel → unloads front axle)
df.loc[:, "deltaFz_long"] = a_x * vehicle_mass * h_cg / L

# fraction of lateral transfer that goes to front axle (roll stiffness split)
lambda_front = 0.55
df.loc[:, "deltaFz_lat_front"] = lambda_front * (vehicle_mass * a_y * h_cg / track_front)

# sign convention: +a_y loads RIGHT side; flip sign if your data is opposite
lat_sign = pd.Series(np.sign(a_y.to_numpy()), index=df.index).fillna(0.0)

# Front-Right dynamic normal load
df.loc[:, "Fz_FR_dynamic"] = (
    static_FR
    - FR_share * df["deltaFz_long"]             # accel unloads front axle
    + 0.5 * lat_sign * df["deltaFz_lat_front"]  # outside front gets +, inside −
).clip(lower=0.1 * static_FR)


# temperature μ
if USE_DYNAMIC_MU and "KO2_Aussen_T_[Grad_Celsius]" in df.columns:
    tf = 1 + 0.002*(df["KO2_Aussen_T_[Grad_Celsius]"]-20)
    df["mu_temp_adjusted"] = np.clip(DRY_ASPHALT_BASE_MU*tf, 0.95*1.2, 1.05*1.2)
else:
    df["mu_temp_adjusted"] = DRY_ASPHALT_BASE_MU

# gear factor (original 4-speed logic) ---------------------------------------
gear_factor = 1.00
if "MO_Istgang" in df.columns:
    df["Gear"] = df["MO_Istgang"].fillna(0).clip(0,4).astype(int)
    gear_factor = df["Gear"].map({1:1.06, 2:1.04, 3:1.02, 4:1.00, 0:1.00})
else:
    df["Gear"] = 0

df["mu_gear_adjusted"] = df["mu_temp_adjusted"] * gear_factor

# ——— NEW  dynamic μ drop with slip-angle ------------------------------------
SLIP_MU_SLOPE = 0.005
if "Slip_Angle_[Grad]" in df.columns:
    slip_corr = 1 - SLIP_MU_SLOPE*df["Slip_Angle_[Grad]"].abs()
    df["mu_final"] = np.clip(df["mu_gear_adjusted"]*slip_corr, 0.6, 1.3)
else:
    df["mu_final"] = df["mu_gear_adjusted"]

# friction force & mechanical power ------------------------------------------
df["Friction_Force_Dynamic_[N]"] = df["mu_final"] * df["Fz_FR_dynamic"]

df["Probe_Height_[m]"] = probe_height / (1 + (df["KO1_[km\\h]"]/120)**2)

v_ms         = df["KO1_[km\\h]"]/3.6
df["dynamic_radius"] = np.clip(tire_radius*(1-0.001*df["Fz_FR_dynamic"]/static_FR),
                               0.25, 0.40)
omega_tyre   = v_ms / df["dynamic_radius"]
omega_engine = df.get("MO1_Drehzahl_[1\\min]", 0).fillna(0) * 0.10472
omega_use    = np.minimum(omega_tyre, omega_engine)
df["Mech_Power"] = (df["Friction_Force_Dynamic_[N]"]*df["dynamic_radius"]*
                    omega_use*0.85)/1000    # kW

# —— energy-balance QC scatter  ---------------------------------------------

# slip-rate & slip-energy -----------------------------------------------------
df["Slip_Rate"]   = df["Slip_Angle_[Grad]"].diff() / TICK_RATE
df["slip_x_speed"]= df["Slip_Angle_[Grad]"]*df["KO1_[km\\h]"]
df["slip_energy"] = (df["Slip_Angle_[Grad]"].abs()
                     *df["Friction_Force_Dynamic_[N]"]*TICK_RATE).cumsum()/1000





# ----------  Regime-feature scaffold  ---------------------------------
g = 9.80665                                      # gravity

df["lambda_x"] = df["a_long"] / (df["mu_final"]*G)      # signed long. demand
df["lambda_y"] = df["a_lat"]  / (df["mu_final"]*G)      # lateral demand

df["slip_power"] = (                                    # |α|·F  ≈ tread work
    df["Slip_Angle_[Grad]"].abs() *
    df["Friction_Force_Dynamic_[N]"]
)

regime_cols = [
    "lambda_x", "lambda_y", "KO1_[km\\h]",
    "Curvature_[1_m]", "BR5_Bremsdruck_[bar]", "slip_power"
]

# lag-correction for PM probes (unchanged) ------------------------------------
# === Robust PM shifting & masks (handles missing raw columns) ===
pm_raw_all = ["PM1_ug_per_m3", "PM2_5_ug_per_m3", "PM10_ug_per_m3"]
pm_raw = [c for c in pm_raw_all if c in df.columns]
if not pm_raw:
    print("[WARN] No PM*_ug_per_m3 columns found; PM targets will be empty.")

df["estimated_lag"] = (
    df["Probe_Height_[m]"] / (df["KO1_[km\\h]"].clip(lower=5) / 3.6)
).round().astype(int).clip(1)

for raw in pm_raw:
    tgt = raw.replace("_ug_per_m3", "_shifted")
    idx = np.arange(len(df)); lag = df["estimated_lag"].to_numpy()
    arr = df[raw].to_numpy()
    df[tgt] = arr[np.minimum(idx + lag, len(df) - 1)]
    df[f"{tgt}_mask"] = np.isfinite(df[tgt].to_numpy()).astype(np.float32)

# ensure PM10_shifted exists for downstream code (alias if missing)
if "PM10_shifted" not in df.columns:
    alias = next((x for x in ["PM2_5_shifted", "PM1_shifted"] if x in df.columns), None)
    if alias is not None:
        df["PM10_shifted"] = df[alias]
        df["PM10_shifted_mask"] = df.get(f"{alias}_mask",
                                         np.isfinite(df[alias]).astype(np.float32))
        print(f"[NOTE] Using {alias} as canonical PM10_shifted.")
    else:
        # still create a mask column so later code doesn't crash
        df["PM10_shifted_mask"] = 0.0

TARGETS = [t for t in ["PM1_shifted","PM2_5_shifted","PM10_shifted"] if t in df.columns]
PRETTY  = {"PM1_shifted":"PM₁","PM2_5_shifted":"PM₂.₅","PM10_shifted":"PM₁₀"}
PRETTY  = {k: v for k, v in PRETTY.items() if k in TARGETS}
UNIT    = "µg/m³"
# === end robust PM block ===


# slip-ratio, gear extras (unchanged from original) ---------------------------
for side in ["L","R"]:
    col = f"BR3_Rad_V{side}_[km\\h]"
    if col in df.columns:
        df[f"Slip_Ratio_F{side}"] = (df[col]-df["KO1_[km\\h]"])/df["KO1_[km\\h]"].clip(1)
if {"Slip_Ratio_FL","Slip_Ratio_FR"}.issubset(df.columns):
    df["Slip_Ratio_Avg"] = (df["Slip_Ratio_FL"]+df["Slip_Ratio_FR"])/2
else:
    df["Slip_Ratio_Avg"] = 0.0

# final friction utilisation ---------------------------------------------------
df["Friction_Utilization"] = np.sqrt((df["a_long"]/(df["mu_final"]*G))**2 +
                                     (df["a_lat"] /(df["mu_final"]*G))**2)

# —— physical sanity guards ---------------------------------------------------
viol = df['Friction_Utilization'] > 1.30
if viol.any():
    print(f"[WARN] clamping {int(viol.sum())} rows with μ-utilisation > 1.30")
    df.loc[viol, 'Friction_Utilization'] = 1.30

# === Strictly-causal imputation for model features (moved here) ===
# Make sure every feature used by the model actually exists,
# then do causal fill (ffill) and final zero-fill for any leading gaps.
for c in feat_cols_for_model:
    if c not in df.columns:
        df[c] = np.nan

df[feat_cols_for_model] = (
    df[feat_cols_for_model]
      .replace([np.inf, -np.inf], np.nan)
      .ffill()
      .fillna(0.0)
)
# ================================================================

# cluster regimes -------------------------------------------------------------

# rows that are effectively “idling” – hard-label as regime 0
idle = (
    (df["KO1_[km\\h]"] < 5) &
    (df["lambda_x"].abs() < .05) &
    (df["lambda_y"].abs() < .05)
)

dyn_mask = ~idle
scaler   = StandardScaler()
n_dyn    = int(dyn_mask.sum())

if n_dyn >= 5:
    X_reg  = scaler.fit_transform(df.loc[dyn_mask, regime_cols].fillna(0))
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    df.loc[dyn_mask, "regime"] = kmeans.fit_predict(X_reg) + 1  # 1..5
    df.loc[idle, "regime"] = 0
else:
    df["regime"] = 0
    print(f"[KMeans] skipped: only {n_dyn} dynamic rows")
df["regime"] = df["regime"].astype(int)


# ——————————————————————————————————————— 4. two driving scenarios
df1 = df[(df["X_Achse_[s]"]>=93)  & (df["X_Achse_[s]"]<=584)].copy()
df2 = df[(df["X_Achse_[s]"]>=584) & (df["X_Achse_[s]"]<=1150)].copy()

# Replace special chars in column names so LightGBM won't crash
# ——————————————————————————————————————— 5. analysis & plots
vmin, vmax = df["KO1_[km\\h]"].min(), df["KO1_[km\\h]"].max()

for tgt in TARGETS:

    X1 = df1[SELECTED].fillna(0)
    X2 = df2[SELECTED].fillna(0)
    y1, y2 = df1[tgt], df2[tgt]


    try:
        rf1, X1_safe = lgb_fit(X1, y1, tgt=tgt)
        rf2, X2_safe = lgb_fit(X2, y2, tgt=tgt)
    except ValueError as e:
        print(f"[LGB] {tgt}: skipped one/both segments — {e}")
        continue


    # Use the exact rows LightGBM saw (X*_safe index) to align y
    y1_safe = y1.loc[X1_safe.index]
    y2_safe = y2.loc[X2_safe.index]

    perm1 = permutation_importance(rf1, X1_safe[rf1.feature_name_], y1_safe, n_repeats=5, random_state=42, n_jobs=-1)
    perm2 = permutation_importance(rf2, X2_safe[rf2.feature_name_], y2_safe, n_repeats=5, random_state=42, n_jobs=-1)



        # 5-a feature importance --------------------------------------------------
    # Features actually used by each model (sanitized names)
    f1 = list(rf1.feature_name_)
    f2 = list(rf2.feature_name_)

    s1 = pd.Series(perm1.importances_mean, index=f1, name="Imp_93_584")
    s2 = pd.Series(perm2.importances_mean, index=f2, name="Imp_584_1150")

    imp = pd.concat([s1, s2], axis=1).fillna(0.0).reset_index().rename(columns={"index":"Feature"})

    # Map back to original (unsanitized) names for pretty labels/units
    safe = pd.Index([re.sub(r'[^A-Za-z0-9_]', '_', c) for c in SELECTED])
    name_map = dict(zip(safe, SELECTED))  # safe → original

    imp["Feature_pretty"] = imp["Feature"].map(name_map).fillna(imp["Feature"])
    feature_labels = [
        f"{f} [{FEATURE_UNITS.get(f,'')}]" if FEATURE_UNITS.get(f,"") else f
        for f in imp["Feature_pretty"]
    ]

    imp.sort_values("Imp_93_584", ascending=False, inplace=True)
    imp.to_csv(f"ml_charts/feature_importance/feature_importance_{tgt}.csv", index=False)

    # ── PLOT (uses joined & aligned importances)
    idx = np.arange(len(imp)); width = 0.4
    plt.figure(figsize=(12,6), dpi=200)
    plt.barh(idx-width/2, imp["Imp_93_584"],   height=width, label="93–584 s",   alpha=.85)
    plt.barh(idx+width/2, imp["Imp_584_1150"], height=width, label="584–1150 s", alpha=.85)
    plt.yticks(idx, feature_labels)
    plt.xscale("symlog", linthresh=1e-6)
    plt.xlabel("Permutation importance (ΔR², log scale)")
    plt.title(f"Permutation importance — {PRETTY[tgt]} (log scale)")
    plt.legend(); plt.tight_layout()
    plt.savefig(f"ml_charts/feature_importance/feature_importance_{tgt}_log.png", dpi=300)
    plt.close()


    # 5-b  utilisation vs emission  (per regime)  -------------------------------
    fig, ax = plt.subplots(figsize=(9, 6), dpi=200)

    cmap = plt.cm.viridis
    size = 35

    for reg in sorted(df['regime'].unique()):
        s1 = df1[df1['regime'] == reg]        # ← now defined
        s2 = df2[df2['regime'] == reg]

        ax.scatter(
            s1["Friction_Utilization"], s1[tgt],
            c=s1["KO1_[km\\h]"], cmap=cmap, vmin=vmin, vmax=vmax,
            marker="o", s=size, alpha=.7, edgecolors="black", linewidths=.3,
            label=f"Reg {reg} (93–584 s)" if reg == 0 else None
        )
        ax.scatter(
            s2["Friction_Utilization"], s2[tgt],
            c=s2["KO1_[km\\h]"], cmap=cmap, vmin=vmin, vmax=vmax,
            marker="^", s=size, alpha=.7, edgecolors="black", linewidths=.3,
            label=f"Reg {reg} (584–1150 s)" if reg == 0 else None
        )

    norm = plt.Normalize(vmin, vmax)
    sm   = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Speed [km/h]")

    ax.set_xlabel("Friction utilisation [–]")
    ax.set_ylabel(f"{PRETTY[tgt]} [{UNIT}]")
    ax.set_title(f"{PRETTY[tgt]} vs tyre-grip utilisation")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(f"ml_charts/friction_vs_emission/util_vs_{tgt}.png", dpi=300)
    plt.close(fig)


 # 5-c prediction-error display -------------------------------------------

# ─────────────────────────────────────────────────────────────────────────────
PREDICTORS = [
    "KO1_[km\\h]", "Pitch_Angle_[Grad]", "BR5_Bremsdruck_[bar]","Velocity_[km\\h]",
    "KO1_[km\\h]_lag1", "Pitch_Angle_[Grad]_lag1", "BR5_Bremsdruck_[bar]_lag1",
    "KO1_[km\\h]_lag2", "Pitch_Angle_[Grad]_lag2", "BR5_Bremsdruck_[bar]_lag2" , "Heading_rate_[deg_s]", "Curvature_[1_m]"
]
# --- ensure lags exist for key channels (global once)
for lag in (1, 2):
    for f in ["KO1_[km\\h]", "Pitch_Angle_[Grad]", "BR5_Bremsdruck_[bar]"]:
        if f in df.columns:
            df[f"{f}_lag{lag}"] = df[f].shift(lag).ffill().fillna(0.0)

# reduce to columns that actually exist
PREDICTORS = [c for c in PREDICTORS if c in df.columns]
TEACH_PREDICTORS = [c for c in feat_cols_for_model if c in df.columns]


TARGETS = ["PM1_shifted", "PM2_5_shifted", "PM10_shifted"]
PRETTY  = dict(zip(TARGETS, ["PM₁", "PM₂.₅", "PM₁₀"]))
UNIT    = "µg/m³"


# ── Optuna per-target tuning (with time gap) ─────────────────────────
def make_objective(tgt_name: str):
    def objective(trial):
        params = {
    "n_estimators": trial.suggest_int("n_estimators", 3000, 19000, step=2000),
    "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
    "num_leaves": trial.suggest_int("num_leaves", 31, 255),
    "max_depth": trial.suggest_int("max_depth", 4, 12),
    "min_child_samples": trial.suggest_int("min_child_samples", 10, 120),
    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
    "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
    "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.5),
    "random_state": 42,
    "n_jobs": -1,
    }

        tscv = TimeSeriesSplit(n_splits=5, gap=WIN)
        maes = []

        for tr_idx, te_idx in tscv.split(df):
            train_df = df.iloc[tr_idx].copy()
            test_df  = df.iloc[te_idx].copy()

            # build lags INSIDE the fold
            for lag in (1, 2):
                for f in ["KO1_[km\\h]", "Pitch_Angle_[Grad]", "BR5_Bremsdruck_[bar]"]:
                    train_df[f"{f}_lag{lag}"] = train_df[f].shift(lag).ffill().fillna(0.0)
                    test_df[f"{f}_lag{lag}"]  = test_df[f].shift(lag).ffill().fillna(0.0)

            cols = [c for c in PREDICTORS if c in train_df.columns]
            if not cols:
                continue

            y_tr = train_df[tgt_name].astype(float)
            y_te = test_df[tgt_name].astype(float)

            mtr = np.isfinite(y_tr.to_numpy())
            if mtr.sum() < 20:
                continue

            X_tr_orig = train_df.loc[mtr, cols].copy()
            y_tr = y_tr.loc[mtr]
            X_te_orig = test_df[cols].copy()

            # drop constant columns on THIS fold (use original names)
            non_const = X_tr_orig.columns[X_tr_orig.nunique(dropna=False) > 1]
            X_tr_orig = X_tr_orig[non_const]
            X_te_orig = X_te_orig[non_const].copy()
            if X_tr_orig.shape[1] == 0 or pd.Series(y_tr).nunique() <= 1:
                continue

            # build constraints AFTER pruning (original names)
            mono = build_monotone_list(list(X_tr_orig.columns))

            # sanitize AFTER pruning
            X_tr = X_tr_orig.copy(); X_tr.columns = safe_cols(X_tr.columns)
            X_te = X_te_orig.copy(); X_te.columns = X_tr.columns

            model = lgb.LGBMRegressor(
                objective="regression",
                random_state=42,
                n_jobs=-1,
                **params,
                monotone_constraints=mono,
            )

            # in-fold split for early stopping
            val_split = max(1, min(int(0.8 * len(X_tr)), len(X_tr) - 1))
            early = lgb.early_stopping(100, verbose=False)
            model.fit(
                X_tr.iloc[:val_split], y_tr.iloc[:val_split],
                eval_set=[(X_tr.iloc[val_split:], y_tr.iloc[val_split:])],
                eval_metric="l2",
                callbacks=[early]
            )

            mte = np.isfinite(y_te.to_numpy())
            if not mte.any():
                continue

            preds = model.predict(X_te.loc[mte])
            maes.append(mean_absolute_error(y_te.loc[mte], preds))

        return float(np.mean(maes)) if maes else 1e9
    return objective

# ---- run the studies (small budget to start) ----
if HAS_OPTUNA:
    BEST_LGB_PARAMS_BY_TGT = {}
    for tgt in ["PM1_shifted", "PM2_5_shifted", "PM10_shifted"]:
        study = optuna.create_study(direction="minimize")
        study.optimize(make_objective(tgt), n_trials=20)
        BEST_LGB_PARAMS_BY_TGT[tgt] = study.best_params
        print(f"[Optuna] Best LightGBM params for {tgt}: {study.best_params}")
else:
    print("[Optuna] not available → using defaults per target.")
    BEST_LGB_PARAMS_BY_TGT = {
        "PM1_shifted":   dict(n_estimators=5000, learning_rate=0.01, num_leaves=63, max_depth=8,
                              min_child_samples=40, subsample=0.8, colsample_bytree=0.8, reg_lambda=0.2),
        "PM2_5_shifted": dict(n_estimators=5000, learning_rate=0.01, num_leaves=63, max_depth=8,
                              min_child_samples=40, subsample=0.8, colsample_bytree=0.8, reg_lambda=0.2),
        "PM10_shifted":  dict(n_estimators=5000, learning_rate=0.01, num_leaves=63, max_depth=8,
                              min_child_samples=40, subsample=0.8, colsample_bytree=0.8, reg_lambda=0.2),
    }



# ───────────────────────────────── 3. model & rolling CV ─────────────────────
def run_time_series_cv(df, predictors, targets, time_col='X_Achse_[s]'):
    """Time-series CV with sanitized columns for LightGBM; monotone constraints
    are built from ORIGINAL column names (order preserved), and alignment is
    done after masking non-finite train targets.
    """
    results = {t: {'true': [], 'pred': [], 'time': []} for t in targets}
    tscv = TimeSeriesSplit(n_splits=5, gap=WIN)

    for tr_idx, te_idx in tscv.split(df):
        train_df = df.iloc[tr_idx].copy()
        test_df  = df.iloc[te_idx].copy()

        # create lags INSIDE the fold
        for lag in (1, 2):
            for f in ["KO1_[km\\h]", "Pitch_Angle_[Grad]", "BR5_Bremsdruck_[bar]"]:
                train_df[f"{f}_lag{lag}"] = train_df[f].shift(lag).ffill().fillna(0.0)
                test_df[f"{f}_lag{lag}"]  = test_df[f].shift(lag).ffill().fillna(0.0)

        if train_df.empty or test_df.empty:
            continue

        # Keep original names to compute constraints (order matters)
        X_tr_orig_all = train_df[predictors]
        X_te_orig_all = test_df[predictors]
        cols_orig = list(X_tr_orig_all.columns)
        mono = build_monotone_list(cols_orig)

        for tgt in targets:
            y_tr = train_df[tgt].astype(float)
            y_te = test_df[tgt].astype(float)

            # mask NaNs in train target and ALIGN features with that mask
            mtr = np.isfinite(y_tr.to_numpy())
            if mtr.sum() < 20:
                continue
            X_tr_orig = X_tr_orig_all.loc[mtr].copy()
            y_tr = y_tr.loc[mtr]

            # --- drop constant / all-NaN columns on THIS fold (use original names) ---
            non_const = X_tr_orig.columns[X_tr_orig.nunique(dropna=False) > 1]
            X_tr_orig = X_tr_orig[non_const]
            X_te_orig = X_te_orig_all[non_const].copy()

            if X_tr_orig.shape[1] == 0:
                print("[LGB] skip fold: all features constant after cleaning")
                continue

            # --- skip fold if target has no variation ---
            import pandas as _pd
            if _pd.Series(y_tr).nunique() <= 1:
                print("[LGB] skip fold: target is constant")
                continue

            # rebuild constraints for remaining columns
            mono = build_monotone_list(list(X_tr_orig.columns))

            # sanitize names after pruning (mirror to test)
            X_tr = X_tr_orig.copy()
            X_tr.columns = safe_cols(X_tr.columns)
            X_te = X_te_orig.copy()
            X_te.columns = X_tr.columns


            # sanitize column names for LightGBM and mirror them on test
            assert len(mono) == X_tr.shape[1], "Monotone constraint length mismatch."

            model = make_lgb_fast(X_tr.columns, device="cpu",
                      params_override=BEST_LGB_PARAMS_BY_TGT.get(tgt))
            model.set_params(monotone_constraints=mono)


            # small in-fold validation split + early stopping
            val_split = max(1, min(int(0.8 * len(X_tr)), len(X_tr) - 1))
            early = lgb.early_stopping(100, verbose=False)
            model.fit(
                X_tr.iloc[:val_split], y_tr.iloc[:val_split],
                eval_set=[(X_tr.iloc[val_split:], y_tr.iloc[val_split:])],
                eval_metric="l2",
                callbacks=[early]
            )

            # predict only where test target is finite (keeps alignment)
            mte = np.isfinite(y_te.to_numpy())
            if not mte.any():
                continue
            X_te_eval = X_te.loc[mte]
            y_te_eval = y_te.loc[mte]

            y_hat = model.predict(X_te_eval)

            results[tgt]['true'].extend(y_te_eval)
            results[tgt]['pred'].extend(y_hat)
            results[tgt]['time'].extend(test_df.loc[mte, time_col])

    return results

if HAS_LGB:
    cv_results = run_time_series_cv(df, PREDICTORS, TARGETS)
    gb_metrics = {}
    for tgt in TARGETS:
        mask = np.isfinite(cv_results[tgt]['true']) & np.isfinite(cv_results[tgt]['pred'])
        y_true = np.array(cv_results[tgt]['true'])[mask]
        y_pred = np.array(cv_results[tgt]['pred'])[mask]
        time = np.array(cv_results[tgt]['time'])[mask]
        gb_metrics[tgt] = (mean_absolute_error(y_true, y_pred), rmse(y_true, y_pred))

        fig = plot_prediction_diagnostics(y_true, y_pred, time, PRETTY[tgt], UNIT, df, PREDICTORS)
        fig.savefig(f"ml_charts/diagnostics/diagnostics_{tgt}.png")
        plt.close(fig)

    print("\n📊  Gradient-Boosting cross-validated error summary")
    for tgt in TARGETS:
        mask = (np.isfinite(cv_results[tgt]['true']) & np.isfinite(cv_results[tgt]['pred']))
        y_true_cv = np.array(cv_results[tgt]['true'])[mask]
        y_pred_cv = np.array(cv_results[tgt]['pred'])[mask]
        mae_val  = mean_absolute_error(y_true_cv, y_pred_cv)
        rmse_val = rmse(y_true_cv, y_pred_cv)
        print(f"  {tgt:<12}  MAE = {mae_val:6.2f}   RMSE = {rmse_val:6.2f}")
    print()
else:
    print("[WARN] Skipping LightGBM CV/diagnostics (lightgbm not installed).")
    cv_results = {t: {'true': [], 'pred': [], 'time': []} for t in ["PM1_shifted","PM2_5_shifted","PM10_shifted"]}
    gb_metrics = {}
    
# --- GBM LOFO on full series with a time gap (safe, single split) ---
if HAS_LGB and "PM10_shifted" in df.columns:
    tgt = "PM10_shifted"
    X_all = df[SELECTED].fillna(0)
    y_all = df[tgt].astype(float)

    N   = len(df)
    gap = WIN
    tr_end = int(0.8 * N) - gap
    tr_idx = np.arange(0, max(tr_end, 1))
    te_idx = np.arange(int(0.8 * N), N)

    params = dict(objective="regression", random_state=42, n_jobs=-1)
    params.update(BEST_LGB_PARAMS_BY_TGT.get(tgt, {}))
    gbm = lgb.LGBMRegressor(**params)
    gbm.fit(X_all.iloc[tr_idx], y_all.iloc[tr_idx])

    lofo, lofo_base = lofo_mae(gbm, X_all, y_all, te_idx, feature_names=list(X_all.columns))
    for f, d in lofo:
        print(f"[LOFO] {f:>28}  ΔMAE(TEST) = {d:+.5f}")

# — cumulative emission KPI ---------------------------------------------------
if "KO1_[km\\h]" in df.columns:
    dist_km = (df["KO1_[km\\h]"]*TICK_RATE/3600).cumsum()
    for tgt in TARGETS:
        plt.figure(figsize=(8,4),dpi=200)
        plt.plot(dist_km, df[tgt].cumsum()*TICK_RATE, lw=1)
        plt.xlabel("Distance travelled [km]")
        plt.ylabel(f"Cumulative {PRETTY[tgt]} [µg·s/m³]")
        plt.title(f"Cumulative {PRETTY[tgt]} vs distance")
        plt.tight_layout()
        plt.savefig(f"ml_charts/friction_vs_emission/cum_{tgt}.png",dpi=300)
        plt.close()

print("✅  Plots & CSVs written under ./ml_charts/")
#ßßßßßßßßßßßßßßßßßßßßßßßßß
def save_geo_multi_pm_plot(df, mask=None):
    """Scatter emissions on top of an OpenStreetMap basemap."""
    # default: all rows
    if mask is None:
        mask = df.index

    # If a boolean mask (Series/ndarray) is passed, convert to index positions.
    if isinstance(mask, (pd.Series, np.ndarray)):
        # pandas-friendly check for boolean dtype
        from pandas.api.types import is_bool_dtype
        if is_bool_dtype(mask):
            if len(mask) != len(df):
                raise ValueError(f"Mask length {len(mask)} != df length {len(df)}")
            mask = df.index[mask]

    # required columns
    needed = {"Lat_filt", "Lon_filt"}
    missing = needed - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns for map: {missing} "
                       "(run the GPS preprocessing that creates Lat_filt/Lon_filt)")

    fig, axes = plt.subplots(3, 1, figsize=(11, 14), dpi=300,
                             sharex=True, gridspec_kw=dict(hspace=0.25))

    spec = [
        ("PM₁",   "PM1_shifted",   "Blues",  0),
        ("PM₂.₅", "PM2_5_shifted", "Greens", 1),
        ("PM₁₀",  "PM10_shifted",  "Reds",   2),
    ]

    for label, col, cmap, idx in spec:
        ax = axes[idx]

        if col not in df.columns:
            ax.set_title(f"{label} (missing)")
            continue

        sc = ax.scatter(
            df.loc[mask, "Lon_filt"],
            df.loc[mask, "Lat_filt"],
            c=df.loc[mask, col],
            cmap=cmap, s=8, alpha=0.8, edgecolors="none"
        )

        # Optional basemap
        if HAS_CTX and USE_BASEMAPS:
            try:
                ctx.add_basemap(ax, crs="EPSG:4326",
                                source=ctx.providers.OpenStreetMap.Mapnik)
            except Exception as e:
                print("[map] basemap skipped:", e)

        cb = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.01)
        cb.ax.set_ylabel(f"{label} [{UNIT}]")
        ax.set_aspect("equal")
        ax.set_title(label)

    axes[-1].set_xlabel("Longitude [°]")
    axes[1].set_ylabel("Latitude [°]")
    fig.suptitle("Tyre-wear particulate concentration along driven path", fontsize=15)

    out = "ml_charts/diagnostics/pm_pathmap_ALL.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


save_geo_multi_pm_plot(df)


def plot_overlay(df, out="ml_charts/diagnostics/raw_vs_ekf.png"):
    
    fig, ax = plt.subplots(figsize=(8, 8), dpi=200)

    # raw points – light & tiny
    ax.plot(df["Longitude_[deg]"], df["Latitude_[deg]"],
            ".", ms=2, alpha=0.25, label="raw GPS")

    # EKF path – prominent
    ax.plot(df["Lon_filt"], df["Lat_filt"],
            "-", lw=1.6, color="tab:red", label="EKF track")

    if HAS_CTX and USE_BASEMAPS:
        try:
            ctx.add_basemap(ax, crs="EPSG:4326",
                            source=ctx.providers.OpenStreetMap.Mapnik)
        except Exception as e:
            print("[map] basemap skipped:", e)

    ax.set_xlabel("Longitude [°]")
    ax.set_ylabel("Latitude [°]")
    ax.set_title("Raw GPS fixes vs EKF-smoothed path")
    ax.set_aspect("equal")
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print("🗺️  overlay map saved →", out)


plot_overlay(df)

#ßßßßßßßßßßßßßßßßßßßß


tgt_col = "PM10_shifted"
feat_mat = df[SELECTED].fillna(0).to_numpy(dtype=np.float32)
tgt_vec  = df[tgt_col].to_numpy(dtype=np.float32)

# chronological 80 / 20 split index
split_idx = int(0.8 * len(feat_mat))

# ---- feature scaler (fit on train only) ----
# ---- feature scaler (fit on train only)  · robust to zero std / NaNs ----
x_scaler = StandardScaler()
x_scaler.fit(feat_mat[:split_idx])

# clamp zero/invalid scales and clean means
if hasattr(x_scaler, "scale_"):
    x_scaler.scale_ = np.where((~np.isfinite(x_scaler.scale_)) | (x_scaler.scale_ == 0.0), 1.0, x_scaler.scale_)
if hasattr(x_scaler, "mean_"):
    x_scaler.mean_ = np.nan_to_num(x_scaler.mean_, nan=0.0, posinf=0.0, neginf=0.0)

feat_mat[:split_idx] = x_scaler.transform(feat_mat[:split_idx])
feat_mat[split_idx:] = x_scaler.transform(feat_mat[split_idx:])

# final safety: no NaN/inf in features
feat_mat = np.nan_to_num(feat_mat, nan=0.0, posinf=0.0, neginf=0.0)

# ---- target scaler (fit on finite TRAIN labels only; REAL labels) ----
y_pm10_full = df["PM10_shifted"].to_numpy(np.float32)   # <-- real labels

y_train_raw    = y_pm10_full[:split_idx]
y_train_finite = y_train_raw[np.isfinite(y_train_raw)]
if y_train_finite.size == 0:
    raise ValueError("No finite training labels to fit target scaler.")

y_scaler = StandardScaler().fit(y_train_finite.reshape(-1, 1))

def safe_transform_y(y_arr: np.ndarray) -> np.ndarray:
    # Replace non-finite values with the TRAIN-label median so transform is defined.
    med = float(np.median(y_train_finite))
    y_arr = np.nan_to_num(y_arr.reshape(-1, 1), nan=med, posinf=med, neginf=med)
    y_t = y_scaler.transform(y_arr)
    return np.nan_to_num(y_t, nan=0.0, posinf=0.0, neginf=0.0).ravel()


# ---- window maker (define once) ----
def make_windows_masked(features, target, target_mask, win=512, horizon=1):
    Xs, ys, ws = [], [], []
    T = len(target)
    for i in range(T - win - horizon + 1):
        Xs.append(features[i:i+win, :])
        ys.append(target[i+win+horizon-1])
        ws.append(target_mask[i+win+horizon-1])  # 1 real, 0 missing→pseudo
    return (
        torch.tensor(np.stack(Xs), dtype=torch.float32),
        torch.tensor(np.array(ys), dtype=torch.float32),
        torch.tensor(np.array(ws), dtype=torch.float32)
    )

def make_windows_masked_with_next(features, target, target_mask, win=512, horizon=1):
    """
    Like make_windows_masked, but also returns X_next = features at t+win+horizon (next-step).
    Shapes:
      X:      (N, win, F)
      y:      (N,)
      w:      (N,)
      X_next: (N, F)
    """
    Xs, ys, ws, Xn = [], [], [], []
    T, F = len(target), features.shape[1]
    for i in range(T - win - horizon):
        Xs.append(features[i:i+win, :])
        ys.append(target[i+win+horizon-1])
        ws.append(target_mask[i+win+horizon-1])
        Xn.append(features[i+win+horizon, :])

    if not Xs:  # return valid empty tensors instead of crashing
        empty = lambda *shape: torch.empty(*shape, dtype=torch.float32)
        return empty(0, win, F), empty(0), empty(0), empty(0, F)

    return (
        torch.tensor(np.stack(Xs), dtype=torch.float32),
        torch.tensor(np.array(ys), dtype=torch.float32),
        torch.tensor(np.array(ws), dtype=torch.float32),
        torch.tensor(np.stack(Xn), dtype=torch.float32)
    )

def make_ssl_windows(features, win=WIN):
    """
    Self-supervised windows for next-step prediction.
    Returns (X_ssl, Y_ssl) or empty tensors if sequence is too short.
    """
    Xs, ys = [], []
    T, F = features.shape
    for i in range(T - win):
        Xs.append(features[i:i+win, :])
        ys.append(features[i+win, :])

    if not Xs:
        return torch.empty(0, win, F), torch.empty(0, F)

    return torch.tensor(np.stack(Xs), dtype=torch.float32), \
           torch.tensor(np.stack(ys), dtype=torch.float32)

# Use the imputed features for SSL so every row participates
# Use only the chronological train region (same split_idx you already computed)
# FIX (use the same 80% split you already computed for scalers)
feats_ssl_train = x_scaler.transform(
    df.iloc[:split_idx, :][feat_cols_for_model].to_numpy(np.float32)
)
feats_ssl_train = clip_features(feats_ssl_train, 6.0)
X_ssl, Y_ssl = make_ssl_windows(feats_ssl_train.astype(np.float32), win=WIN)


ssl_split = int(0.9 * len(X_ssl))

ssl_train = TensorDataset(X_ssl[:ssl_split], Y_ssl[:ssl_split])
ssl_loader = DataLoader(ssl_train, batch_size=64, shuffle=True)

# ===  ==============================
n_feat_ssl = X_ssl.shape[-1]
encoder  = TCNEncoder(input_size=n_feat_ssl, num_channels=CHANNELS, kernel_size=5, dropout=0.15).to(DEVICE)
ssl_head = SSLHead(in_dim=encoder.outdim, out_dim=n_feat_ssl).to(DEVICE)

register_nan_hooks(encoder,  name="SSL_Encoder")
register_nan_hooks(ssl_head, name="SSL_Head")
for m in encoder.modules():
    if isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight)

ssl_optim = torch.optim.Adam(list(encoder.parameters()) + list(ssl_head.parameters()), lr=1e-3)
ssl_loss  = nn.MSELoss()

encoder.train(); ssl_head.train()
SSL_EPOCHS = 15
for e in range(1, SSL_EPOCHS + 1):
    running = 0.0
    for xb, yb in ssl_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        # light denoising by random masking (improves robustness)
        with torch.no_grad():
            mask = (torch.rand_like(xb) < 0.03).float()
            xb = xb * (1 - mask)

        ssl_optim.zero_grad()
        z    = encoder(xb)
        yhat = ssl_head(z)                           # both (B, F)
        with torch.no_grad():
            M = (torch.rand_like(yhat) < MASK_RATE).float()
        loss = ((yhat - yb)**2 * M).sum() / (M.sum() + 1e-8)

        loss.backward()
        clip_grad_norm_(list(encoder.parameters()) + list(ssl_head.parameters()), 1.0)
        ssl_optim.step()
        running += loss.item()
    print(f"[SSL] epoch {e:02d}  loss={running/len(ssl_loader):.4f}")
# keep the pretrained weights; ssl_head not used afterwards


# keep the pretrained weights; we won't use ssl_head anymore
# ================================================================


# memory guard – keep it, but stricter


# ---- Correlation-based gate init (train-only) ----
feat_df_train = pd.DataFrame(feat_mat[:split_idx], columns=SELECTED)
y_train_raw   = tgt_vec[:split_idx]  # unscaled target for correlations

corrs = []
for c in SELECTED:
    x = np.asarray(feat_df_train[c], dtype=np.float64)
    if np.allclose(x.std(), 0):
        corrs.append(0.0)
        continue
    xc = (x - x.mean()) / (x.std() + 1e-8)
    yc = (y_train_raw - y_train_raw.mean()) / (y_train_raw.std() + 1e-8)
    corrs.append(float(np.clip(np.corrcoef(xc, yc)[0, 1], -1, 1)))

corr_abs = np.abs(np.nan_to_num(corrs, nan=0.0))
corr_norm = corr_abs / (corr_abs.max() + 1e-8) if corr_abs.max() > 0 else corr_abs
gate_init_vec = 0.2 + 0.8 * corr_norm  # keep >0 to avoid dead gates


# --- supervised windows for PM10 (use z-scored target and real-label mask) ---
# y_pm10_full, y_scaler, feat_mat, WIN are already defined above
w_pm10_full = df.get("PM10_shifted_mask", pd.Series(0.0, index=df.index)).to_numpy(np.float32)
y_z = safe_transform_y(y_pm10_full)  # map raw PM10 to Z-space using TRAIN-only scaler

X, y, w, X_next = make_windows_masked_with_next(
    features=feat_mat.astype(np.float32),   # already scaled & clipped earlier
    target=y_z.astype(np.float32),
    target_mask=w_pm10_full.astype(np.float32),
    win=WIN,
    horizon=1,
)

if X.numel() == 0:
    raise ValueError(f"Not enough rows to make windows (need > {WIN+1}).")


# memory guard – keep it, but stricter
if X.numel() > 2e8:                       # 100 M floats ≈ 400 MB
    raise MemoryError("Lower WIN or CHANNELS; GPU would OOM.")
# ────────────────────────────────────────────────────────────────────────────

if len(X) < 10:
    raise ValueError(f"Only {len(X)} windows produced; "
                     "reduce `win` or collect longer drives.")

# 70% train, 15% val, 15% test (chronological)
n = len(X)
i_tr = int(0.70 * n)
i_va = int(0.85 * n)

train_ds = TensorDataset(X[:i_tr],   y[:i_tr],   w[:i_tr],   X_next[:i_tr])
val_ds   = TensorDataset(X[i_tr:i_va], y[i_tr:i_va], w[i_tr:i_va], X_next[i_tr:i_va])
test_ds  = TensorDataset(X[i_va:],   y[i_va:],   w[i_va:],   X_next[i_va:])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

n_feat = X.shape[-1]

# === REPLACE OLD SUPERVISED TCN TRAINING WITH THIS =========================
# === SELF-/SEMI-SUPERVISED FINETUNE (robust + SSL aux) =====================
pm_model = PMModel(n_feat=X.shape[-1], channels=CHANNELS, kernel_size=5, dropout=0.10).to(DEVICE)
register_nan_hooks(pm_model, name="PMModel")

# apply data-driven init to gates
with torch.no_grad():
    g0 = torch.tensor(gate_init_vec[:pm_model.gate.log_g.numel()],
                      dtype=torch.float32, device=pm_model.gate.log_g.device)
    pm_model.gate.log_g.copy_(torch.log(g0))

# init encoder from the SSL-pretrained encoder
# init encoder from the SSL-pretrained encoder
pm_model.encoder.load_state_dict(encoder.state_dict(), strict=True)

# NEW: carry over the pretrained SSL head weights into the in-model head
pm_model.ssl_aux.load_state_dict(ssl_head.state_dict())


# optional warm start: freeze encoder for a few epochs
# optional warm start: freeze encoder for a longer period
for p in pm_model.encoder.parameters():
    p.requires_grad = False
WARM_EPOCHS = 5

# two LR param groups; encoder gets a smaller LR after unfreeze
optim = torch.optim.Adam([
    {"params": pm_model.gate.parameters(),    "lr": 3e-5, "weight_decay": 5e-5},
    {"params": pm_model.head.parameters(),    "lr": 3e-5, "weight_decay": 5e-5},
    {"params": pm_model.encoder.parameters(), "lr": 1e-6, "weight_decay": 2e-5},
])


loss_fn   = nn.MSELoss()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=EPOCHS, eta_min=1e-6)

# Fine-tune SSL is always off (we only use SSL for pretraining)
ALPHA_SSL_START = 0.05
ALPHA_SSL_END   = 0.00

# Gate sparsity: stronger by default; softer when stabilizing
L1_GATES = 1e-4 if STABLE_MODE else 2e-4


# ============================================

def ssl_weight(epoch, total=EPOCHS):
    import math
    if total <= 1:
        return ALPHA_SSL_END
    t = (epoch - 1) / (total - 1)
    w = (1 + math.cos(math.pi * t)) / 2  # 1→0
    return ALPHA_SSL_END + (ALPHA_SSL_START - ALPHA_SSL_END) * w


grad_history = defaultdict(list)


    # unfreeze encoder after warmup (uses its own smaller LR)
step_i = 0  # gradient logging step counter
PATIENCE     = 6
_best_mae    = float("inf")
_best_epoch  = 0
_bad_epochs  = 0
_best_path   = "tcn_pm10_best.pth"

loss_curves = {
    "train_total": [], "train_sup": [], "train_ssl": [], "train_gate": [],
    "val_sup": [], "val_mae_all": [], "val_mae_real": []
}


for epoch in range(1, EPOCHS + 1):
    with anomaly_guard():
        pm_model.train()
        num_real, num_total = 0, 0

        if epoch == WARM_EPOCHS + 1:
            for p in pm_model.encoder.parameters():
                p.requires_grad = True
            

        epoch_sup = epoch_ssl = epoch_gate = epoch_total = 0.0
        train_batches = 0

        for xb, yb, wb, xb_next in train_loader:

            # (A) augmentation — training only
            if (not AUG_DISABLED) and pm_model.training:
                scale = aug_scale(epoch) if 'aug_scale' in globals() else 1.0
                aug_idx = [SELECTED.index(c) for c in AUG_COLS if c in SELECTED]

                with torch.no_grad():
                    # generic masked Gaussian noise on a few features
                    if aug_idx and AUG_NOISE_STD > 0:
                        mask = torch.zeros(1, 1, n_feat, device=xb.device)
                        mask[..., aug_idx] = 1.0
                        xb.add_(mask * torch.randn_like(xb) * (AUG_NOISE_STD * scale))

                    # KO1-only jitter
                    if JITTER_KO1 and ("KO1_[km\\h]" in SELECTED) and KO1_JITTER_STD > 0:
                        j = SELECTED.index("KO1_[km\\h]")
                        xb[..., j].add_(KO1_JITTER_STD * scale * torch.randn_like(xb[..., j]))




            xb, yb, wb = xb.to(DEVICE), yb.to(DEVICE), wb.to(DEVICE)
            xb = torch.nan_to_num(xb); yb = torch.nan_to_num(yb)

            optim.zero_grad()
            pred_z, g, recon = pm_model(xb, return_ssl=True)
            pred_z = torch.nan_to_num(pred_z); recon = torch.nan_to_num(recon)

            num_real  += int((wb > 0.5).sum().item())
            num_total += int(wb.numel())
            # --- speed-aware weighting (upweight near-idle scenarios)
            speed_idx = SELECTED.index("KO1_[km\\h]") if "KO1_[km\\h]" in SELECTED else None
            if speed_idx is not None:
                # last step speed in the window (z-scored units here)
                v_last = xb[:, -1, speed_idx]
                # convert back to approx km/h scale using scaler stats
                mu, sigma = x_scaler.mean_[speed_idx], x_scaler.scale_[speed_idx]
                v_kmh = (v_last * sigma + mu)
                # weight: emphasize <10 km/h (clip to [1, 2])
                w_speed = torch.clamp(1.0 + (10.0 - v_kmh) / 10.0, 1.0, 2.0)
            else:
                w_speed = torch.ones_like(wb)

            # combine with label mask
            # combine with label mask and speed weighting
            w_eff = wb * w_speed  # shape (B,)

            # fallback if a batch has no supervised signal at all
            if w_eff.sum() < 1e-6:
                w_eff = torch.ones_like(w_eff) * 0.1  # weak supervision fallback

            per_elem = F.smooth_l1_loss(pred_z, yb, beta=1.0, reduction="none")
            sup_loss = (w_eff * per_elem).sum() / (w_eff.sum() + 1e-8)

            # ---- Masked Feature Modeling (score only masked feature dims) ----
            xb_next_ = xb_next.to(DEVICE)                 # (B, F)
            with torch.no_grad():
                M = (torch.rand_like(recon) < MASK_RATE).float()  # (B, F)
            ssl_recon_loss = ((recon - xb_next_)**2 * M).sum() / (M.sum() + 1e-8)
            # ---------------------------------------------------------------

            gate_loss = L1_GATES * g.abs().mean()

            alpha_ssl = ssl_weight(epoch, total=EPOCHS)
            loss = sup_loss + alpha_ssl * ssl_recon_loss + gate_loss

            if not torch.isfinite(loss):
                xb_mean = float(torch.nan_to_num(xb).mean())
                xb_std  = float(torch.nan_to_num(xb).std())
                yb_mean = float(torch.nan_to_num(yb).mean())
                yb_std  = float(torch.nan_to_num(yb).std())
                print(f"[SKIP] non-finite loss. xb μ±σ={xb_mean:.3f}±{xb_std:.3f}  y μ±σ={yb_mean:.3f}±{yb_std:.3f}")
                continue

            loss.backward()
            preclip = clip_grad_norm_(pm_model.parameters(), 1.0)

            # gradient logging (unchanged)
            step_i += 1
            if step_i % GRAD_LOG_EVERY == 0:
                import re
                keep = set()
                for pat in TRACKED_PARAM_PATTERNS:
                    rx = re.compile(pat)
                    for name, param in pm_model.named_parameters():
                        if param.grad is not None and rx.search(name):
                            keep.add(name)
                for name, param in pm_model.named_parameters():
                    if name in keep and param.grad is not None:
                        grad_history[name].append(param.grad.data.norm(2).item())
                grad_history["__global__"].append(float(preclip))

            optim.step()  
            # accumulate batch losses
            epoch_sup   += float(sup_loss)
            epoch_ssl   += float(ssl_recon_loss)
            epoch_gate  += float(gate_loss)
            epoch_total += float(loss)
            train_batches += 1




    
    pm_model.eval()
    with torch.no_grad():
        val_pred_z = torch.cat([pm_model(b_x.to(DEVICE))[0] for (b_x, _, _, _) in val_loader])
        val_true_z = torch.cat([b_y.to(DEVICE)             for (_, b_y, _, _) in val_loader])
        val_w      = torch.cat([b_w                         for (_, _, b_w, _) in val_loader]).cpu().numpy()


    val_pred = y_scaler.inverse_transform(val_pred_z.cpu().numpy().reshape(-1, 1)).ravel()
    val_true = y_scaler.inverse_transform(val_true_z.cpu().numpy().reshape(-1, 1)).ravel()

    # Coerce any non-finite values to safe numbers before metrics
    val_true = np.nan_to_num(val_true, nan=0.0, posinf=0.0, neginf=0.0)
    val_pred = np.nan_to_num(val_pred, nan=0.0, posinf=0.0, neginf=0.0)

    from sklearn.metrics import mean_absolute_error
    real_mask = val_w > 0.5
    if np.isfinite(val_true).any() and np.isfinite(val_pred).any():
        all_mae  = mean_absolute_error(val_true, val_pred)
        real_mae = mean_absolute_error(val_true[real_mask], val_pred[real_mask]) if real_mask.any() else float("nan")
        # --- validation supervised loss in Z-space (same as training sup_loss) ---
        # use tensors already built above: val_pred_z, val_true_z, and numpy val_w
        val_per_elem = F.smooth_l1_loss(val_pred_z, val_true_z, beta=1.0, reduction="none")
        val_w_t = torch.from_numpy(val_w).to(val_per_elem.device).float()
        val_sup = ((val_w_t * val_per_elem).sum() / (val_w_t.sum() + 1e-8)).item()

        # --- log epoch means ---
        m_total = epoch_total / max(1, train_batches)
        m_sup   = epoch_sup   / max(1, train_batches)
        m_ssl   = epoch_ssl   / max(1, train_batches)
        m_gate  = epoch_gate  / max(1, train_batches)

        loss_curves["train_total"].append(m_total)
        loss_curves["train_sup"].append(m_sup)
        loss_curves["train_ssl"].append(m_ssl)
        loss_curves["train_gate"].append(m_gate)
        loss_curves["val_sup"].append(val_sup)
        loss_curves["val_mae_all"].append(float(all_mae))
        loss_curves["val_mae_real"].append(float(real_mae))

    else:
        all_mae, real_mae = float("nan"), float("nan")

    real_frac_train = (num_real / max(1, num_total))
    print(f"Epoch {epoch:02d}  ALL MAE={all_mae:.2f} | REAL MAE={real_mae:.2f} | "
      f"n_real_val={int(real_mask.sum())}/{len(val_true)} | real_frac_train={real_frac_train:.2%}")

    # ---- Early stopping on REAL MAE (fallback to ALL if NaN)
    metric = float(real_mae) if np.isfinite(real_mae) else float(all_mae)
    if np.isfinite(metric) and (metric + 1e-6 < _best_mae):
        _best_mae, _best_epoch, _bad_epochs = metric, epoch, 0
        torch.save(pm_model.state_dict(), _best_path)
        print(f"[ES] new best @ epoch {epoch:02d}: MAE={metric:.3f}")
    else:
        _bad_epochs += 1
        if _bad_epochs >= PATIENCE:
            print(f"[ES] stop after {PATIENCE} bad epochs (best={_best_mae:.3f} @ {_best_epoch})")
            break

    scheduler.step()





# save AFTER the training loop
# restore best checkpoint before saving the final artifact
if os.path.exists(_best_path):
    pm_model.load_state_dict(torch.load(_best_path, map_location=DEVICE))

# --- Step 4: save/plot loss & metric history (do this BEFORE saving the final .pth)
import pandas as pd, matplotlib.pyplot as plt, numpy as np, os
hist = pd.DataFrame(loss_curves)
hist.index.name = "epoch"
os.makedirs("ml_charts/diagnostics", exist_ok=True)
hist.to_csv("ml_charts/diagnostics/loss_curves.csv")

epochs = np.arange(1, len(hist) + 1)
plt.figure(figsize=(9,5), dpi=200)
plt.plot(epochs, hist["train_sup"], label="train_sup (SmoothL1)")
plt.plot(epochs, hist["val_sup"],   label="val_sup (SmoothL1)")
plt.axvline(_best_epoch, ls="--", alpha=0.5, label=f"best @ {int(_best_epoch)}")
plt.xlabel("epoch"); plt.ylabel("loss (Z-space)"); plt.title("Training vs Validation Loss")
plt.legend(frameon=False); plt.tight_layout()
plt.savefig("ml_charts/diagnostics/loss_curves_sup.png", dpi=300); plt.close()

plt.figure(figsize=(9,5), dpi=200)
plt.plot(epochs, hist["train_total"], label="train_total (sup + α·ssl + gate)")
plt.plot(epochs, hist["val_sup"],     label="val_sup")
plt.axvline(_best_epoch, ls="--", alpha=0.5)
plt.xlabel("epoch"); plt.ylabel("loss"); plt.title("Total train loss vs val sup loss")
plt.legend(frameon=False); plt.tight_layout()
plt.savefig("ml_charts/diagnostics/loss_curves_total.png", dpi=300); plt.close()

torch.save(pm_model.state_dict(), "tcn_pm10_gated.pth")


# --------------------------------------------------------------
TRAIN_WIN = WIN 
HORIZON  = 1
SCALER   = x_scaler      # <- feature scaler used to train the TCN
Y_SCALER = y_scaler      # <- target scaler used to train the TCN
MODEL_PM = pm_model.eval() # finished network


@torch.no_grad()
def tcn_predict_series(df_src: pd.DataFrame, feature_cols=SELECTED, win=TRAIN_WIN):
    # 1) same preprocessing as training
    feats = df_src[feature_cols].fillna(0).to_numpy(np.float32)
    feats = SCALER.transform(feats)
    feats = np.clip(feats, -6.0, 6.0)  # mirror training clip

    # need at least (win + 1) rows to produce the first next-step prediction
    if feats.shape[0] <= win:
        return np.full(len(df_src), np.nan, dtype=float)

    # 2) windows are [i : i+win) → model predicts target at (i+win)
    X_tmp = np.stack([feats[i:i+win, :] for i in range(0, len(feats) - win)]).astype(np.float32)
    ds = DataLoader(TensorDataset(torch.from_numpy(X_tmp)), batch_size=256, shuffle=False)

    # 3) run model and inverse-scale
    pred_z = torch.cat([MODEL_PM(b.to(DEVICE))[0].cpu() for (b,) in ds]).numpy()
    pred   = Y_SCALER.inverse_transform(pred_z.reshape(-1, 1)).ravel()

    # 4) write predictions at indices [win :] (because label is at i+win)
    full = np.full(len(df_src), np.nan, dtype=float)
    full[win:] = pred
    return full
pred_full = tcn_predict_series(df, feature_cols=SELECTED, win=TRAIN_WIN)
_idx = np.flatnonzero(np.isfinite(pred_full))
print("first finite pred index:", int(_idx[0]) if len(_idx) else "none")
 # should be == TRAIN_WIN

# Show learned feature weights (the “importance” your model discovered)
# Show learned feature weights (the “importance” your model discovered)
# --- Show learned feature weights (gates) ---
# Place this immediately after saving the model.

with torch.no_grad():
    # gates are parameterized in log space → softplus to get positive weights
    gate_vals = torch.nn.functional.softplus(pm_model.gate.log_g).detach().cpu().numpy().astype(np.float32)

# Align to SELECTED and save/print
idx_names = SELECTED[:len(gate_vals)]
gate_series = pd.Series(gate_vals[:len(idx_names)], index=idx_names).sort_values(ascending=False)
gate_series.name = "gate"  # give the Series a column name
print("\nTop learned feature weights (gates):\n", gate_series.head(12))
out_csv = "ml_charts/diagnostics/learned_feature_gates.csv"
gate_series.to_csv(out_csv, header=True)

mu_gate = float(gate_series.get("mu_final", np.nan))
if np.isfinite(mu_gate):
    print(f"[GATE] mu_final = {mu_gate:.6f}")
else:
    print("[GATE] mu_final not found in learned gates (check SELECTED)")

# ============================================================================
def log_gradient_norms(grad_history, out_path, max_series=25):
    import numpy as np
    from pathlib import Path
    import matplotlib.pyplot as plt

    if not grad_history or sum(len(v) for v in grad_history.values()) == 0:
        print("[grad debug] nothing to plot")
        return

    plt.figure(figsize=(12, 6), dpi=200)
    keys = list(grad_history.keys())
    keys_sorted = ["__global__"] + [k for k in keys if k != "__global__"]
    keys_sorted = keys_sorted[:max_series]

    if len(keys_sorted) <= max_series:
        for name in keys_sorted:
            norms = grad_history.get(name, [])
            if len(norms) > 1:
                plt.plot(norms, label=name, alpha=0.7)
        plt.legend(fontsize=6, ncol=2, frameon=False)
        title = "Gradient Norms (log scale)"
    else:
        steps = max((len(v) for v in grad_history.values()), default=0)
        if steps > 0:
            stacked = np.vstack([
                np.array(v + [np.nan]*(steps - len(v)))
                for v in grad_history.values()
            ])
            mean_curve = np.nanmean(stacked, axis=0)
            p90_curve  = np.nanpercentile(stacked, 90, axis=0)
            p10_curve  = np.nanpercentile(stacked, 10, axis=0)
            plt.plot(mean_curve, label="mean grad norm")
            plt.plot(p90_curve,  label="p90")
            plt.plot(p10_curve,  label="p10")
            plt.legend(frameon=False)
        title = "Gradient Norms (aggregated, log scale)"

    plt.yscale("log")
    plt.xlabel("Training Step (subsampled)")
    plt.ylabel("Gradient L2 Norm")
    plt.title(title)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[grad debug] Saved gradient plot → {Path(out_path).resolve()}")


# --- sanity checks before plotting grads
diag_dir = Path("ml_charts/diagnostics")
diag_dir.mkdir(parents=True, exist_ok=True)

num_params = sum(p.numel() for p in pm_model.parameters())
num_keys = len(grad_history)
num_points = sum(len(v) for v in grad_history.values())
print(f"[grad debug] train batches={len(train_loader)} | params={num_params} | "
      f"tracked={num_keys} | total_points={num_points}")

if num_points == 0:
    print("[grad debug] No gradient points recorded. Likely no training batches "
          "or backward() not executed. Check split size and DataLoader.")


if ENABLE_GRAD_PLOTS:
    log_gradient_norms(grad_history, "ml_charts/diagnostics/gradient_norms.png", max_series=MAX_GRAD_SERIES)



# ── Evaluate on the held-out test windows and map back to full df  (REAL-ONLY MAE)
pm_model.eval()
with torch.no_grad():
    val_pred_z = torch.cat([pm_model(b_x.to(DEVICE))[0] for (b_x, _, _, _) in test_loader])
    val_true_z = torch.cat([b_y.to(DEVICE)             for (_, b_y, _, _) in test_loader])
    val_w      = torch.cat([b_w                        for (_, _, b_w, _) in test_loader]).cpu().numpy()  # << add

pm_pred = y_scaler.inverse_transform(val_pred_z.cpu().numpy().reshape(-1,1)).ravel()
pm_true = y_scaler.inverse_transform(val_true_z.cpu().numpy().reshape(-1,1)).ravel()

from sklearn.metrics import mean_absolute_error
# REAL labels only (mask comes from the DataLoader)
real_mask = val_w > 0.5
finite = np.isfinite(pm_pred) & np.isfinite(pm_true) & real_mask
mae_test = mean_absolute_error(pm_true[finite], pm_pred[finite]) if finite.any() else float("nan")

from pathlib import Path
Path("ml_charts/diagnostics").mkdir(parents=True, exist_ok=True)
csv_path = "ml_charts/diagnostics/ssl_ablation.csv"
header_needed = not os.path.exists(csv_path)

with open(csv_path, "a", encoding="utf-8") as f:
    if header_needed:
        f.write("ssl_enabled,seed,mae_test_real\n")   # << column name clarifies "real-only"
    f.write(f"{int(EXPERIMENT_SSL)},{SEED},{mae_test:.6f}\n")

print(f"[SSL_ABL] ssl={EXPERIMENT_SSL} seed={SEED}  REAL-ONLY test MAE={mae_test:.3f}")




# ================================================

# (Optional) compare with GB metric if you want:
if "PM10_shifted" in gb_metrics:
    gb_mae, gb_rmse = gb_metrics["PM10_shifted"]
    from sklearn.metrics import mean_absolute_error
    finite = np.isfinite(pm_pred) & np.isfinite(pm_true)
    if finite.any():
        print(f"PM10_shifted: GB  MAE={gb_mae:.2f}  RMSE={gb_rmse:.2f} | "
            f"Gated-TCN  MAE={mean_absolute_error(pm_true[finite], pm_pred[finite]):.2f}  "
            f"RMSE={rmse(pm_true[finite], pm_pred[finite]):.2f}")
    else:
        print("[ERROR] all TCN predictions are non-finite; check training logs.")

# Map predictions to their original row positions
win, horizon = TRAIN_WIN, 1
n_rows = len(df)
target_pos = np.arange(win + horizon - 1, n_rows)  # positions with labels
pm10_hat_full = np.full(n_rows, np.nan, dtype=float)
pm10_hat_full[target_pos[-len(pm_pred):]] = pm_pred  # align tail to test set

df_pred = df.copy()
df_pred["PM10_hat"] = pm10_hat_full
for col in ["PM1_hat", "PM2_5_hat"]:
    if col not in df_pred.columns:
        df_pred[col] = np.nan

def audit_coverage(df_src, df_pred_src, WIN, test_frac=0.2):
    T = len(df_src)
    n_real = int(df_src.get("PM10_shifted_mask", pd.Series(np.zeros(T))).sum())
    n_pseudo = T - n_real
    n_win_total = max(0, T - WIN)
    n_train = int((1 - test_frac) * n_win_total)
    n_test  = n_win_total - n_train
    n_pred_eval = int(np.isfinite(df_pred_src.get("PM10_hat", pd.Series([np.nan]*T))).sum())
    n_pred_full = int(np.isfinite(df_pred_src.get("PM10_hat_all", pd.Series([np.nan]*T))).sum())
    print(f"[COVERAGE] rows total           : {T}")
    print(f"[COVERAGE] real labels          : {n_real}  ({n_real/T:.1%})")
    print(f"[COVERAGE] pseudo labels        : {n_pseudo}  ({n_pseudo/T:.1%})")
    print(f"[COVERAGE] windows total        : {n_win_total}  (skip first {WIN-1})")
    print(f"[COVERAGE] train/test windows   : {n_train} / {n_test}")
    print(f"[COVERAGE] preds (test-only map): {n_pred_eval}")
    print(f"[COVERAGE] preds (full series)  : {n_pred_full}")

# ---- Full-drive predictions from WIN-1 onward (no skipping after warmup)
df_pred["PM10_hat_all"] = tcn_predict_series(df, feature_cols=SELECTED, win=TRAIN_WIN)
audit_coverage(df, df_pred, WIN)

# ========================= SANITY CHECK SUITE =========================
from sklearn.metrics import mean_absolute_error
import numpy as np, pandas as pd, os, matplotlib.pyplot as plt

SANITY_DIR = "ml_charts/diagnostics/sanity"
os.makedirs(SANITY_DIR, exist_ok=True)

# --- 0) helper: map window indices → row masks (for fair splits)
def split_row_masks_from_windows(n_win, win, i_tr, i_va, horizon=1, n_rows=None):
    # label row for window j is j + win + horizon - 1 (here horizon=1)
    lab = np.arange(n_win) + (win + horizon - 1)
    train_mask = np.zeros(n_rows, bool)
    val_mask   = np.zeros(n_rows, bool)
    test_mask  = np.zeros(n_rows, bool)
    train_mask[lab[:i_tr]]        = True
    val_mask[lab[i_tr:i_va]]      = True
    test_mask[lab[i_va:]]         = True
    return train_mask, val_mask, test_mask

# recover window counts & row masks from objects you already built
n_win = len(X)                           # number of TCN windows you produced
train_rows, val_rows, test_rows = split_row_masks_from_windows(
    n_win, TRAIN_WIN, i_tr, i_va, horizon=HORIZON, n_rows=len(df)
)

# mask where we have real labels and finite preds
y_col = "PM10_shifted"
y_true_full = df[y_col].to_numpy(np.float32)
y_mask_real = (df[f"{y_col}_mask"].to_numpy(np.float32) > 0.5)

# you already wrote full-series TCN predictions into df_pred["PM10_hat_all"]
y_pred_full = df_pred["PM10_hat_all"].to_numpy(np.float32)

def safe_mae(y_true, y_pred, m):
    m = m & np.isfinite(y_true) & np.isfinite(y_pred)
    return mean_absolute_error(y_true[m], y_pred[m]) if m.any() else np.nan

# --- 1) Overfitting check: train vs val vs test (REAL labels only)
mae_train = safe_mae(y_true_full, y_pred_full, train_rows & y_mask_real)
mae_val   = safe_mae(y_true_full, y_pred_full, val_rows   & y_mask_real)
mae_test  = safe_mae(y_true_full, y_pred_full, test_rows  & y_mask_real)

of_idx  = (mae_train - mae_val)/mae_val if np.isfinite(mae_train) and np.isfinite(mae_val) and mae_val>0 else np.nan
of_idx2 = (mae_val   - mae_test)/mae_test if np.isfinite(mae_val) and np.isfinite(mae_test) and mae_test>0 else np.nan

with open(os.path.join(SANITY_DIR, "overfit_report.txt"), "w", encoding="utf-8") as f:
    f.write(f"TCN MAE (REAL labels only)\n")
    f.write(f"  train: {mae_train:.3f}\n  val  : {mae_val:.3f}\n  test : {mae_test:.3f}\n")
    f.write(f"Overfit index (train→val): {of_idx:.3f}\n")
    f.write(f"Generalization drift (val→test): {of_idx2:.3f}\n")

print(f"[SANITY] TCN MAE train/val/test = {mae_train:.2f}/{mae_val:.2f}/{mae_test:.2f}  (overfit idx={of_idx:.3f})")

# --- 2) Gap-CV option for your GBM CV to avoid near-leakage
# (Use this if you're re-running CV here; otherwise, add gap=WIN where you define TimeSeriesSplit)
# Example patch:  TimeSeriesSplit(n_splits=5, gap=WIN)

# --- 3) Baselines: persistence & rolling-mean (to ensure we beat trivial models)
# --- 3) Baselines: persistence & rolling-mean (strictly causal, TEST only)
y = pd.Series(y_true_full)
m_real = pd.Series(y_mask_real)

# previous *real* label exists at t-1?
prev_real = m_real.shift(1, fill_value=False)

# persistence = last real observation at t-1; otherwise NaN
persist = y.shift(1)
persist[~prev_real] = np.nan

# rolling mean of last K seconds using only past values
K = int(5.0 / TICK_RATE)
roll = (y.where(m_real)
          .shift(1)                       # forbid peeking at current y
          .rolling(K, min_periods=1)
          .mean())

def _mae(mask, yhat):
    m = mask & np.isfinite(y) & np.isfinite(yhat)
    return mean_absolute_error(y[m], yhat[m]) if m.any() else np.nan

mae_persist_test = _mae(test_rows & y_mask_real, persist)
mae_roll_test    = _mae(test_rows & y_mask_real, roll)

with open(os.path.join(SANITY_DIR, "baseline_report.txt"), "w", encoding="utf-8") as f:
    f.write("Baselines on TEST (REAL labels only)\n")
    f.write(f"  persistence: {mae_persist_test:.3f}\n")
    f.write(f"  rolling-{K} samples: {mae_roll_test:.3f}\n")
    f.write(f"  TCN: {mae_test:.3f}\n")

print(f"[SANITY] TEST baselines — persistence: {mae_persist_test:.2f}  rolling: {mae_roll_test:.2f}")

# --- 4) Target-permutation smoke test (time-series split with gap)
try:
    fold_lo = int(0.65 * len(df))
    fold_hi = int(0.80 * len(df))
    Xp  = df.iloc[fold_lo:fold_hi][SELECTED].fillna(0)
    yr  = df.iloc[fold_lo:fold_hi]["PM10_shifted"].astype(float)
    m   = np.isfinite(yr.to_numpy())
    Xp, yr = Xp.loc[m], yr.loc[m]

    # sanitize once
    Xp_s = Xp.copy()
    Xp_s.columns = Xp_s.columns.str.replace(r'[^0-9a-zA-Z_]', '_', regex=True)
    mono = build_monotone_list(list(Xp.columns))

    tscv = TimeSeriesSplit(n_splits=3, gap=WIN)
    mae_true_all, mae_perm_all = [], []

    # one global permutation of labels
    yr_perm = yr.sample(frac=1.0, random_state=13).reset_index(drop=True)

    for tr, te in tscv.split(Xp_s):
        split = int(0.8 * len(tr))
        tr_idx, va_idx = tr[:split], tr[split:]

        m1 = make_lgb_fast(Xp_s.columns, device="cpu")
        m1.set_params(monotone_constraints=mono)
        m1.fit(Xp_s.iloc[tr_idx], yr.iloc[tr_idx],
               eval_set=[(Xp_s.iloc[va_idx], yr.iloc[va_idx])],
               eval_metric="l2",
               callbacks=[lgb.early_stopping(50, verbose=False)])
        mae_true_all.append(mean_absolute_error(yr.iloc[te], m1.predict(Xp_s.iloc[te])))

        m2 = make_lgb_fast(Xp_s.columns, device="cpu")
        m2.set_params(monotone_constraints=mono)
        m2.fit(Xp_s.iloc[tr_idx], yr_perm.iloc[tr_idx],
               eval_set=[(Xp_s.iloc[va_idx], yr_perm.iloc[va_idx])],
               eval_metric="l2",
               callbacks=[lgb.early_stopping(50, verbose=False)])
        mae_perm_all.append(mean_absolute_error(yr.iloc[te], m2.predict(Xp_s.iloc[te])))

    if mae_true_all and mae_perm_all:
        print(f"[SANITY] Permutation test (folded) — real: {np.mean(mae_true_all):.2f}  permuted: {np.mean(mae_perm_all):.2f}")

except Exception as e:
    print("[SANITY] permutation test skipped:", e)

# --- 5) Feature drift summary (train vs test) — mean/std/quantiles per feature
summ = []
for f in SELECTED:
    a = df.loc[train_rows, f].to_numpy(np.float64)
    b = df.loc[test_rows , f].to_numpy(np.float64)
    if a.size < 10 or b.size < 10: 
        continue
    q = lambda x: np.nanquantile(x, [0.05, 0.50, 0.95])
    summ.append({
        "feature": f,
        "train_mean": np.nanmean(a), "test_mean": np.nanmean(b),
        "train_std":  np.nanstd(a),  "test_std":  np.nanstd(b),
        "train_q05":  q(a)[0],       "test_q05":  q(b)[0],
        "train_q50":  q(a)[1],       "test_q50":  q(b)[1],
        "train_q95":  q(a)[2],       "test_q95":  q(b)[2],
        "mean_shift_abs": abs(np.nanmean(a) - np.nanmean(b))
    })
pd.DataFrame(summ).sort_values("mean_shift_abs", ascending=False)\
  .to_csv(os.path.join(SANITY_DIR, "feature_drift_train_vs_test.csv"), index=False)

# --- 6) Residual autocorrelation (should decay fast; long tails hint leakage/underfit)
valid_test = test_rows & y_mask_real & np.isfinite(y_true_full) & np.isfinite(y_pred_full)
res = (y_pred_full - y_true_full)[valid_test]
if res.size > 50:
    L = min(200, res.size-1)  # up to ~20 s if TICK_RATE=0.1
    ac = np.correlate(res - res.mean(), res - res.mean(), mode="full")
    ac = ac[ac.size//2:ac.size//2+L+1]
    ac = ac / ac[0]
    t  = np.arange(L+1) * TICK_RATE
    plt.figure(figsize=(7,4), dpi=200); plt.plot(t, ac); plt.ylim(-0.2, 1.0)
    plt.xlabel("Lag [s]"); plt.ylabel("Residual ACF"); plt.title("Residual autocorrelation (TEST)")
    plt.tight_layout(); plt.savefig(os.path.join(SANITY_DIR, "residual_acf_test.png")); plt.close()

# --- 7) LightGBM monotonicity spot-check (vary one feature, hold others at median)
def check_monotone_1d(model, X_frame, col, n=50, eps=1e-6):
    X0 = X_frame.median(numeric_only=True).to_frame().T
    xs = np.linspace(X_frame[col].quantile(0.02), X_frame[col].quantile(0.98), n)
    Xs = pd.concat([X0]*n, ignore_index=True)
    Xs[col] = xs
    Z = Xs.copy(); Z.columns = Z.columns.str.replace(r'[^0-9a-zA-Z_]', '_', regex=True)
    preds = model.predict(Z)
    diffs = np.diff(preds)
    inc = np.all(diffs >= -eps); dec = np.all(diffs <= eps)
    return inc, dec

# optional: run on the last fitted GBM models rf1/rf2 if available
try:
    cols = X1_safe.columns  # from your earlier section
    # Check a couple monotone features you specified (+1): speed & pitch
    for col in ["KO1_[km\\h]","Pitch_Angle_[Grad]"]:
        if col in X1.columns:
            inc, dec = check_monotone_1d(rf1, X1, col)
            verdict = "INC" if inc else ("DEC" if dec else "VIOLATION")
            print(f"[SANITY] Monotonicity {col}: {verdict}")
except Exception:
    pass

# --- 8) Gate sanity — warn if too many features are effectively off
try:
    small = (gate_series < 1e-3).sum()
    if small >= max(1, int(0.5*len(gate_series))):
        print(f"[SANITY] ⚠ many gates are ~0: {small}/{len(gate_series)} — consider easing L1_GATES or re-check scaling.")
except Exception:
    pass

# --- 9) Look-ahead smoke test — shift all predictors by +1 step; performance should drop
df_lookahead = df.copy()
for c in SELECTED:
    df_lookahead[c] = df_lookahead[c].shift(-1)  # (uses future)
pred_la = tcn_predict_series(df_lookahead, feature_cols=SELECTED, win=TRAIN_WIN)
mae_la  = safe_mae(y_true_full, pred_la, test_rows & y_mask_real)
with open(os.path.join(SANITY_DIR, "lookahead_smoke.txt"), "w") as f:
    f.write(f"TEST MAE with +1 step *future* features injected: {mae_la:.3f}\n")
    f.write("This number SHOULD be noticeably BETTER than baseline; "
            "if your normal model is *similar*, you might be leaking future info.\n")
mae_bad, mae_cheat, n1, n2 = lookahead_smoke_checks(X_all, y_all, tr_idx, te_idx, K=5, seed=42)
print(f"[SANITY] look-ahead smoke (shift+K worse, cheat better): bad={mae_bad:.2f} (n={n1}) | cheat={mae_cheat:.2f} (n={n2})")
# ======================================================================


# === [NEW] TCN LOFO importance on TEST ONLY (real labels only) ===
y_true_full = df["PM10_shifted"].to_numpy(np.float32)
y_mask_full = df["PM10_shifted_mask"].to_numpy(np.float32) > 0.5
base_pred_full = df_pred["PM10_hat_all"].to_numpy(np.float32)

# test-row mask same as before
n_rows = len(df)
test_start_row = TRAIN_WIN + i_va - 1
test_row_mask = np.zeros(n_rows, dtype=bool)
if test_start_row < n_rows:
    context_start = max(0, test_start_row - TRAIN_WIN + 1)
    test_row_mask[context_start:] = True

mask_eval = test_row_mask & y_mask_full & np.isfinite(y_true_full) & np.isfinite(base_pred_full)
base_mae = mean_absolute_error(y_true_full[mask_eval], base_pred_full[mask_eval])

@torch.no_grad()
def predict_with_gate_zero(feature_name: str):
    idx = SELECTED.index(feature_name)
    saved = pm_model.gate.log_g[idx].item()
    try:
        pm_model.gate.log_g[idx] = -20.0  # softplus(-20) ~ 0
        return tcn_predict_series(df, feature_cols=SELECTED, win=TRAIN_WIN)
    finally:
        pm_model.gate.log_g[idx] = saved


delta = {}
for f in SELECTED:
    pred_f = predict_with_gate_zero(f)
    mae_f  = mean_absolute_error(y_true_full[mask_eval], pred_f[mask_eval])
    delta[f] = mae_f - base_mae
    print(f"[LOFO] {f:>28s}  ΔMAE(TEST, real-only) = {delta[f]:.5f}")

tcn_lofo = pd.Series(delta).sort_values(ascending=False)
tcn_lofo.to_csv("ml_charts/diagnostics/tcn_lofo_importance.csv")

plt.figure(figsize=(10,6), dpi=200)
tcn_lofo.plot(kind="barh")
plt.gca().invert_yaxis()
plt.xlabel("ΔMAE vs TEST baseline (higher = more important)")
plt.title("TCN LOFO importance (PM10, TEST, real-only)")
plt.tight_layout()
plt.savefig("ml_charts/diagnostics/tcn_lofo_importance.png", dpi=300)
plt.close()

# ===  Rank alignment between GBM permutation and TCN-LOFO (optional) ===
# ===  Rank alignment between GBM permutation and TCN-LOFO (robust) ===
imp_path = "ml_charts/feature_importance/feature_importance_PM10_shifted.csv"
if os.path.exists(imp_path):
    imp_gbm = pd.read_csv(imp_path)
    imp_gbm["avg_perm"] = 0.5*imp_gbm["Imp_93_584"] + 0.5*imp_gbm["Imp_584_1150"]
    gbm_rank = imp_gbm.set_index("Feature")["avg_perm"]

    common = tcn_lofo.index.intersection(gbm_rank.index)
    if len(common) >= 3:
        a = tcn_lofo.loc[common].rank()
        b = gbm_rank.loc[common].rank()

        if a.nunique() < 2 or b.nunique() < 2:
            print(f"[RANK] Spearman undefined over {len(common)} features — "
                  f"constant ranks (TCN uniq={a.nunique()}, GBM uniq={b.nunique()}).")
        else:
            rho = a.corr(b, method="spearman")
            print(f"[RANK] Spearman(GBM-permutation, TCN-LOFO TEST) = {rho:.2f} over {len(common)} features")

        out = pd.DataFrame({
            "feature": common,
            "tcn_lofo_delta_mae": tcn_lofo.loc[common].values,
            "gbm_perm_avg": gbm_rank.loc[common].values,
            "rank_tcn": a.values,
            "rank_gbm": b.values,
        }).sort_values("rank_tcn")
        out.to_csv("ml_charts/diagnostics/rank_alignment.csv", index=False)

# ===  ===
# === [NEW] Correlation and ablation for mu_final ===
cols_corr = [c for c in SELECTED if c in df.columns] + ["PM10_shifted"]
corr = df[cols_corr].corr(numeric_only=True)
if "mu_final" in corr.columns:
    print("[CORR] mu_final correlations (top 8):")
    print(corr["mu_final"].sort_values(ascending=False).head(8))

# Ablation: keep shape intact (important for SCALER), but remove information content
if "mu_final" in df.columns:
    df_mu_frozen = df.copy()
    df_mu_frozen["mu_final"] = np.nanmedian(df_mu_frozen["mu_final"].to_numpy())
    pred_mu_frozen = tcn_predict_series(df_mu_frozen, feature_cols=SELECTED, win=TRAIN_WIN)
    mae_mu_frozen  = mean_absolute_error(y_true_full[mask_eval], pred_mu_frozen[mask_eval])
    print(f"[ABLATE] MAE with mu_final frozen = {mae_mu_frozen:.3f} "
          f"(baseline {base_mae:.3f}, Δ={mae_mu_frozen - base_mae:.3f})")
# === ===

# Optional: save for inspection
df_pred[["X_Achse_[s]", "PM10_shifted", "PM10_hat", "PM10_hat_all"]].to_csv(
    "ml_charts/prediction/pm10_full_series.csv", index=False
)
plt.figure(figsize=(12,5), dpi=200)
plt.plot(df_pred["X_Achse_[s]"], df_pred["PM10_shifted"], lw=0.8, label="actual")
plt.plot(df_pred["X_Achse_[s]"], df_pred["PM10_hat_all"], lw=0.8, label="TCN full-series")
plt.legend(); plt.title(f"PM₁₀ – full series (win={WIN})"); plt.tight_layout()
plt.savefig("ml_charts/prediction/pm10_full_series.png", dpi=300); plt.close()



TARGETS_TO_PLOT = ["PM10_shifted"]       # plot PM10 only
PRETTY_LOCAL    = {t: PRETTY[t] for t in TARGETS_TO_PLOT}

TARGETS_TO_PLOT = ["PM10_shifted"]          # ← what to draw
PRETTY_PLOT     = {t: PRETTY[t] for t in TARGETS_TO_PLOT}
COLORS          = {"PM10_shifted": "tab:red"}

# ------------------------------------------------------------------
# FINAL -- PLOT ONLY THE TARGETS YOU REALLY HAVE
# ------------------------------------------------------------------
TARGETS_TO_PLOT = ["PM10_shifted"]               # draw PM10 only
PRETTY_PLOT     = {t: PRETTY[t] for t in TARGETS_TO_PLOT}
COLORS          = {"PM10_shifted": "tab:red"}

for tgt in TARGETS_TO_PLOT:          # ["PM10_shifted"]
    pretty = PRETTY_PLOT[tgt]
    hat    = tgt.replace("_shifted", "_hat")

    fig = plt.figure(figsize=(12, 8), dpi=200, constrained_layout=True)
    gs  = fig.add_gridspec(3, 1, hspace=0.35)

    # ① time series ----------------------------------------------------------
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(df_pred["X_Achse_[s]"], df_pred[tgt],
             lw=.8, label="actual", color=COLORS[tgt], alpha=.7)
    ax1.plot(df_pred["X_Achse_[s]"], df_pred[hat],
             lw=.8, label="pred",   color="k",        alpha=.7)
    ax1.set_ylabel(f"{pretty} [{UNIT}]")
    ax1.set_title(f"{pretty} – full-drive prediction (TCN, win={WIN})")
    ax1.legend()

    # --------------------------------------------------------------
    # rows where both actual *and* prediction are finite
    # --------------------------------------------------------------
    valid = np.isfinite(df_pred[tgt]) & np.isfinite(df_pred[hat])

    # ② predicted vs actual --------------------------------------------------
    ax2 = fig.add_subplot(gs[1])
    sns.kdeplot(x=df_pred.loc[valid, tgt],
                y=df_pred.loc[valid, hat],
                fill=True, thresh=.05, cmap="mako", ax=ax2)
    ax2.plot([df_pred.loc[valid, tgt].min(), df_pred.loc[valid, tgt].max()],
             [df_pred.loc[valid, tgt].min(), df_pred.loc[valid, tgt].max()],
             "--", color="grey")
    ax2.set_xlabel("actual")
    ax2.set_ylabel("predicted")
    ax2.set_title("Density plot – ideal = dashed line")

    # ③ residual distribution -------------------------------------------------
    ax3 = fig.add_subplot(gs[2])
    residuals = df_pred.loc[valid, hat] - df_pred.loc[valid, tgt]
    sns.histplot(residuals, bins=60, kde=True, color=COLORS[tgt], ax=ax3)
    ax3.set_xlabel("prediction error (µg/m³)")
    ax3.set_title(
        f"Residuals – MAE {abs(residuals).mean():.2f} · "
        f"RMSE {rmse(df_pred.loc[valid, tgt], df_pred.loc[valid, hat]):.2f}"
    )

    fig.tight_layout()
    fig.savefig(f"ml_charts/diagnostics/tcn_full_{tgt}.png", dpi=300)
    plt.close(fig)




# --------------------------------------------------------------
# BOX A  ·  wrap the trained TCN into a convenient predictor


# ------------------------------------------------------------------
# BOX B  ·  Booster vs Classic using the *TCN* instead of LightGBM
# ------------------------------------------------------------------
def prep_traj(df_traj_raw: pd.DataFrame) -> pd.DataFrame:
    """Adds the same minimal features as in the main file."""
    df = df_traj_raw.copy()
    df["KO1_[km\\h]"] = df["v_ms"] * 3.6
    df["a_long"]      = df["KO1_[km\\h]"].diff().fillna(0) / 3.6 / 0.1
    for lag in (1, 2):
        df[f"KO1_[km\\h]_lag{lag}"] = df["KO1_[km\\h]"].shift(lag).ffill().fillna(0.0)
        df[f"a_long_lag{lag}"]      = df["a_long"].shift(lag).ffill().fillna(0.0)

    return df



# ---------- BOX F1 : make sure external trajectories
#                     contain *all* SELECTED features ----------
def ensure_all_features(df_src: pd.DataFrame,
                        feature_cols=list(SELECTED)) -> pd.DataFrame:
    """
    Make sure df_src contains *every* feature used during TCN training
    and also a 't' time column that later plotting code relies on.
    Missing columns are filled with zeros.
    """
    df_out = df_src.copy()

    # ------------------------------------------------------------------ #
    # 1.  Guarantee the TIME column
    # ------------------------------------------------------------------ #
    if "t" not in df_out.columns:
        # assume a 0.1-s tick like the training data → derive from row-index
        df_out["t"] = df_out.index * 0.1

    # ------------------------------------------------------------------ #
    # 2.  Guarantee every feature column
    # ------------------------------------------------------------------ #
    for col in feature_cols:
        if col not in df_out.columns:
            df_out[col] = 0.0       # neutral filler

    # ------------------------------------------------------------------ #
    # 3.  Return with the right column order:
    #     first 't', then the exact SELECTED list (no duplicates)
    # ------------------------------------------------------------------ #
    ordered = ["t"] + [c for c in feature_cols if c != "t"]
    return df_out[ordered]

# ---------- end BOX F1 ----------
# Summarize results if we have multiple runs logged ===
try:
    df_ssl = pd.read_csv("ml_charts/diagnostics/ssl_ablation.csv")
    if len(df_ssl) >= 2:
        for flag, g in df_ssl.groupby("ssl_enabled"):
            m = g["mae_test_real"].mean()
            s = g["mae_test_real"].std(ddof=1) if len(g) > 1 else 0.0
            print(f"[SSL_SUMMARY] ssl={int(flag)}  n={len(g)}  REAL-ONLY MAE mean±std = {m:.3f} ± {s:.3f}")
except Exception as _e:
    pass

# ================================================================

