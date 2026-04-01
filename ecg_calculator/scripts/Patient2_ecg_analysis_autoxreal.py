#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

import numpy as np
from numba import njit
from scipy.stats import pearsonr

# -----------------------------------------------------------------------------
# 1. Configuration constants
# -----------------------------------------------------------------------------
PROJECT_DIR = Path("./P2")
STATIC_FILE = PROJECT_DIR / "ecg_static_data.txt"
RAW_OUTPUT_DIR = Path("./P2_results/autoxmanual/ecg_cru")
ANALYSIS_FILE = Path("./P2_results/autoxmanual/analise_comparativa.txt")

LEADS_AUTOMATICO = np.array(
    [
        [ 64201, 173961,  35096],  # V1
        [ 45794, 156424,  58708],  # V2
        [ 39480, 139593,  82570],  # V3
        [ 37515, 120931, 103112],  # V4
        [-17016,  63586, 113015],  # V5
        [-41756,  19384,  59628],  # V6
    ],
    dtype=np.float64,
)

LEADS_REAL = np.array(
    [
        [ 85571, 167768,  41211],  # V1
        [ 54208, 145703,  65230],  # V2
        [ 47224, 122221,  90984],  # V3
        [ 42970,  95963, 111566],  # V4
        [-17694, -21164,  90069],  # V5
        [-25994, -52391,  67087],  # V6
    ],
    dtype=np.float64,
)




# Conversion factor µm ➜ m for leads only (centers/discretizations are already SI)
MICROMETER_TO_METER = 1e-6  # (mantido caso você queira usar depois, mas não é usado aqui)

# -----------------------------------------------------------------------------
# 2. I/O helpers
# -----------------------------------------------------------------------------

def load_static_data(path: Path):
    """Load scale factor, cell centers and discretizations from the static file."""
    if not path.is_file():
        raise FileNotFoundError(f"Static file not found: {path}")

    lines = path.read_text().splitlines()
    scale_factor = float(lines[0].split(":")[1].strip())

    start_idx = lines.index("# Cell Data (center_x, center_y, center_z, dx, dy, dz):") + 1
    cell_data = np.array([list(map(float, ln.split())) for ln in lines[start_idx:]], dtype=np.float64)

    centers = cell_data[:, :3]          # já em metros
    discretizations = cell_data[:, 3:]  # half-widths em metros
    return scale_factor, centers, discretizations


def discover_beta_files(directory: Path):
    """Return (time, file_path) tuples sorted by time."""
    pattern = re.compile(r"ecg_beta_im_t=([\d.]+)\.bin")
    matches = []
    for file in directory.iterdir():
        match = pattern.match(file.name)
        if match:
            matches.append((float(match.group(1)), file))
    return sorted(matches)


def save_results(results: List[List[float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for row in results:
            time = f"{row[0]:.6f}"
            values = " ".join(f"{v:.24f}" for v in row[1:])
            f.write(f"{time} {values}\n")

# -----------------------------------------------------------------------------
# 3. Low-level numeric kernel (Numba-JIT)
# -----------------------------------------------------------------------------

@njit
def _compute_ecg_per_lead(
    lead_pos: np.ndarray,
    centers: np.ndarray,
    discretizations: np.ndarray,
    beta_im: np.ndarray,
    scale_factor: float,
) -> np.ndarray:
    """Compute ECG for all leads for a single time frame."""
    n_leads = lead_pos.shape[0]
    n_cells = centers.shape[0]
    ecg = np.zeros(n_leads, dtype=np.float64)

    for l in range(n_leads):
        acc = 0.0
        for i in range(n_cells):
            # cell volume (real volume = (2dx)(2dy)(2dz) = 8*dx*dy*dz)
            vol = discretizations[i, 0] * discretizations[i, 1] * discretizations[i, 2] * 8.0
            dx = centers[i, 0] - lead_pos[l, 0]
            dy = centers[i, 1] - lead_pos[l, 1]
            dz = centers[i, 2] - lead_pos[l, 2]
            dist = (dx * dx + dy * dy + dz * dz) ** 0.5
            if dist > 0.0:
                acc += (beta_im[i] / dist) * vol
        ecg[l] = -scale_factor * acc
    return ecg

# -----------------------------------------------------------------------------
# 4. Statistics helpers
# -----------------------------------------------------------------------------

def pearson_by_channel(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.array([pearsonr(x[:, i], y[:, i])[0] for i in range(x.shape[1])])


def rrmse(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    rms = np.sqrt(np.sum((x - y) ** 2, axis=0))
    denom = np.sqrt(np.sum(y ** 2, axis=0))
    return 100.0 * rms / denom

# -----------------------------------------------------------------------------
# 5. Core processing routines
# -----------------------------------------------------------------------------

def compute_ecg_series(
    beta_files: List[tuple[float, Path]],
    leads_m: np.ndarray,
    scale_factor: float,
    centers: np.ndarray,
    discretizations: np.ndarray,
) -> List[List[float]]:
    results = []
    for idx, (time_val, fpath) in enumerate(beta_files, 1):
        beta_im = np.fromfile(fpath, dtype=np.float64)
        ecg_vals = _compute_ecg_per_lead(leads_m, centers, discretizations, beta_im, scale_factor)
        results.append([round(time_val, 6)] + ecg_vals.tolist())
        print(f"  • Frame {idx}/{len(beta_files)} @ t={time_val:.3f} done")
    return results


def generate_groups() -> Dict[str, np.ndarray]:
    """
    Gera grupos apenas com V1–V6.

    - 'automatico' : V1–V6 automáticos
    - 'manual'     : V1–V6 reais
    - 'automatico_lead{i}manual'   : todos automáticos, exceto lead i (1..6) que é manual
    - 'manual_lead{i}automatico'   : todos manuais, exceto lead i (1..6) que é automático
    """
    auto_6 = LEADS_AUTOMATICO[:6].copy()
    real_6 = LEADS_REAL[:6].copy()

    groups: Dict[str, np.ndarray] = {
        "automatico": auto_6,
        "manual": real_6,
    }

    # Automatico com substituições manuais (apenas V1–V6)
    for i in range(6):
        g = auto_6.copy()
        g[i] = real_6[i]
        groups[f"automatico_lead{i+1}manual"] = g

    # Manual com substituições automáticas (apenas V1–V6)
    for i in range(6):
        g = real_6.copy()
        g[i] = auto_6[i]
        groups[f"manual_lead{i+1}automatico"] = g

    return groups

# -----------------------------------------------------------------------------
# 6. Comparative analysis (somente V1–V6)
# -----------------------------------------------------------------------------

def compare_groups(ecg_data: Dict[str, np.ndarray]) -> None:
    # Agora só temos os precordiais
    names = ["V1", "V2", "V3", "V4", "V5", "V6"]

    ANALYSIS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with ANALYSIS_FILE.open("w") as f:
        def _write_pair(n1: str, n2: str):
            # Dados: coluna 0 = tempo, colunas 1–6 = V1–V6
            e1 = ecg_data[n1][:, 1:]
            e2 = ecg_data[n2][:, 1:]

            p_vals = pearson_by_channel(e1, e2)
            p_global = pearsonr(e1.flatten(), e2.flatten())[0]
            r_vals = rrmse(e1, e2)
            numer = np.sum((e1.flatten() - e2.flatten()) ** 2)
            denom = np.sum(e2.flatten() ** 2)
            r_global = 100 * np.sqrt(numer / denom) if denom else np.nan

            f.write(f"Comparando '{n1}' x '{n2}'\n" + "-" * 60 + "\n")
            f.write("Pearson por canal:\n")
            for nm, v in zip(names, p_vals):
                f.write(f"{nm}: {v:.4f}\n")
            f.write(f"Pearson Global: {p_global:.4f}\n\n")

            f.write("rRMSE por canal (%):\n")
            for nm, v in zip(names, r_vals):
                f.write(f"{nm}: {v:.2f} %\n")
            f.write(f"rRMSE Global: {r_global:.2f} %\n" + "=" * 60 + "\n\n")

        # comparação base
        _write_pair("manual", "automatico")

        # variações de cada lead
        for i in range(1, 7):
            _write_pair(f"automatico_lead{i}manual", "automatico")
            _write_pair(f"manual_lead{i}automatico", "manual")

    print(f"✅ Comparative analysis saved to: {ANALYSIS_FILE}")

# -----------------------------------------------------------------------------
# 7. Main entry point
# -----------------------------------------------------------------------------

def main() -> None:
    # Load static mesh information
    scale, centers, discretizations = load_static_data(STATIC_FILE)

    # Descobrir arquivos de beta (mesmo conjunto para todos os grupos)
    beta_files = discover_beta_files(PROJECT_DIR)

    groups = generate_groups()
    ecg_raw_data: Dict[str, List[List[float]]] = {}

    for name, leads_um in groups.items():
        # Os eletrodos já estão na mesma unidade do mesh (metros)
        leads_m = leads_um.astype(np.float64)
        print(f"\nCalculando ECG para grupo: {name}")
        series = compute_ecg_series(beta_files, leads_m, scale, centers, discretizations)
        ecg_raw_data[name] = series

        out_path = RAW_OUTPUT_DIR / f"{name}.txt"
        save_results(series, out_path)
        print(f"  → Grupo '{name}' salvo em {out_path}")

    print("\n✅ Todos os grupos processados com sucesso.")

    # Converter listas para arrays e rodar análise comparativa diretamente em V1–V6
    ecg_arrays: Dict[str, np.ndarray] = {
        name: np.array(series, dtype=np.float64) for name, series in ecg_raw_data.items()
    }

    compare_groups(ecg_arrays)


if __name__ == "__main__":
    main()
