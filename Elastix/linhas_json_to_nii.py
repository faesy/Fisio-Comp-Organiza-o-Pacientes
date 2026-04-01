import os
import json
import numpy as np
import SimpleITK as sitk

# ==========================
# CONFIGURAÇÕES DO USUÁRIO
# ==========================

# CT no ESPAÇO em que você desenhou as ROIs no Slicer
CT_PATH = "CT.nii.gz"

# Pasta onde estão os .mrk.json das linhas
INPUT_JSON_DIR = "linhas json"

# Pasta de saída dos .nii.gz
OUTPUT_NII_DIR = "linhas_nii"

# Espessura mínima em VOXELS (em cada eixo local da ROI)
# Aumente se ainda estiver sumindo depois do transformix
MIN_THICKNESS_VOXELS = 4

os.makedirs(OUTPUT_NII_DIR, exist_ok=True)


def load_ct_reference(ct_path: str):
    """Carrega o CT de referência e retorna imagem e metadados."""
    img = sitk.ReadImage(ct_path)
    spacing = np.array(img.GetSpacing(), dtype=float)      # (sx, sy, sz)
    origin = np.array(img.GetOrigin(), dtype=float)        # (ox, oy, oz)
    direction = np.array(img.GetDirection(), dtype=float).reshape(3, 3)
    size = np.array(img.GetSize(), dtype=int)              # (nx, ny, nz)
    return img, spacing, origin, direction, size


def build_mask_for_roi(
    size, spacing, origin, direction,
    center, size_box, R, min_thickness_voxels=4
):
    """
    Cria uma máscara binária (z,y,x) para uma ROI Box no espaço do CT.
    - size: (nx, ny, nz)
    - spacing: (sx, sy, sz)
    - origin: (ox, oy, oz)
    - direction: 3x3
    - center: (3,) em mm (LPS do CT)
    - size_box: (3,) tamanho da caixa em mm
    - R: 3x3 orientação da ROI (LPS)
    """
    # half extents originais
    half = np.array(size_box, dtype=float) / 2.0

    # espessura mínima em mm baseada no menor voxel
    min_thickness_mm = float(min_thickness_voxels) * float(spacing.min())
    min_half = min_thickness_mm / 2.0

    # garante espessura mínima em todos os eixos locais
    half = np.maximum(half, min_half)

    print("    Half extents (mm) após engrossar:", half)

    # cria volume vazio (z,y,x)
    nx, ny, nz = size  # cuidado: size vem como (nx,ny,nz)
    mask = np.zeros((nz, ny, nx), dtype=np.uint8)

    # gera grade de índices (k,j,i) = (z,y,x)
    zz, yy, xx = np.meshgrid(
        np.arange(nz),
        np.arange(ny),
        np.arange(nx),
        indexing="ij"
    )

    # empilha como (3, N) na ordem (x,y,z) = (i,j,k)
    ijk = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()])  # (3, N)

    # índice -> mm no sistema de índices
    ijk_mm = ijk * spacing[:, None]  # (3,N)

    # aplica direção + origem -> coordenadas físicas (LPS) do CT
    xyz_world = origin[:, None] + direction @ ijk_mm  # (3,N)

    # coordenadas no referencial local da ROI
    # local = R^T * (world - center)
    local = R.T @ (xyz_world - center[:, None])

    inside = (
        (np.abs(local[0]) <= half[0]) &
        (np.abs(local[1]) <= half[1]) &
        (np.abs(local[2]) <= half[2])
    )

    mask_flat = mask.ravel()
    mask_flat[inside] = 1
    mask = mask_flat.reshape(mask.shape)

    print("    Voxels marcados:", int(mask.sum()))
    return mask


def convert_single_json(ct_img, spacing, origin, direction, size, json_path, output_path):
    """Converte um único .mrk.json (ROI Box) em um .nii.gz binário."""
    print(f"\n==> Convertendo ROI de: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not data.get("markups"):
        print("    ⚠️  JSON sem campo 'markups', pulando.")
        return

    roi = data["markups"][0]

    if roi.get("type") != "ROI" or roi.get("roiType") != "Box":
        print("    ⚠️  Markup não é ROI Box, pulando.")
        return

    center = np.array(roi["center"], dtype=float)          # (3,)
    size_box = np.array(roi["size"], dtype=float)          # (3,)
    R = np.array(roi["orientation"], dtype=float).reshape(3, 3)

    print("    Center (mm):", center)
    print("    Size box (mm):", size_box)

    mask = build_mask_for_roi(
        size=size,
        spacing=spacing,
        origin=origin,
        direction=direction,
        center=center,
        size_box=size_box,
        R=R,
        min_thickness_voxels=MIN_THICKNESS_VOXELS,
    )

    if mask.sum() == 0:
        print("    ⚠️  Máscara ficou vazia (0 voxels marcados). Verificar ROI/CT.")
        return

    # cria imagem ITK (z,y,x) -> (x,y,z)
    mask_img = sitk.GetImageFromArray(mask)
    mask_img.SetSpacing(tuple(spacing))
    mask_img.SetOrigin(tuple(origin))
    mask_img.SetDirection(tuple(direction.ravel()))

    sitk.WriteImage(mask_img, output_path)
    print(f"    ✅ Salvo NIfTI em: {output_path}")


def main():
    # carrega CT de referência
    print(f"Carregando CT de referência: {CT_PATH}")
    ct_img, spacing, origin, direction, size = load_ct_reference(CT_PATH)
    print("  Size (nx,ny,nz):", size)
    print("  Spacing (mm):   ", spacing)
    print("  Origin (mm):    ", origin)

    # percorre todos os .mrk.json da pasta
    json_files = [
        os.path.join(INPUT_JSON_DIR, f)
        for f in os.listdir(INPUT_JSON_DIR)
        if f.lower().endswith(".mrk.json")
    ]

    if not json_files:
        print(f"⚠️  Nenhum .mrk.json encontrado em: {INPUT_JSON_DIR}")
        return

    print(f"\nEncontrados {len(json_files)} arquivos .mrk.json.")

    for json_path in json_files:
        base = os.path.basename(json_path)

        # nome de saída: tira só o ".mrk.json" e adiciona ".nii.gz"
        if base.lower().endswith(".mrk.json"):
            base_name = base[:-len(".mrk.json")]
        else:
            base_name = os.path.splitext(base)[0]

        out_nii = os.path.join(OUTPUT_NII_DIR, base_name + ".nii.gz")

        convert_single_json(
            ct_img=ct_img,
            spacing=spacing,
            origin=origin,
            direction=direction,
            size=size,
            json_path=json_path,
            output_path=out_nii,
        )

    print("\n🚀 Conversão concluída.")


if __name__ == "__main__":
    main()
