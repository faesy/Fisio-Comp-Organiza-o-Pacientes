import os
import SimpleITK as sitk
import numpy as np
from skimage import measure
import pyvista as pv

# ========================
# CONFIGURAÇÕES
# ========================
pacientes = ["p1", "p2", "p7", "p10"]

# Se você estiver comparando com 3D Slicer e quiser RAS, coloque True
CONVERTER_PARA_RAS = False

for p in pacientes:
    print(f"\n========== CONVERTENDO PACIENTE {p} ==========")

    base_folder = os.path.join("output_transformix_linhas", p)
    out_folder = f"vtp_linhas_{p}"
    os.makedirs(out_folder, exist_ok=True)

    if not os.path.isdir(base_folder):
        print(f"⚠️  Pasta não encontrada: {base_folder}")
        continue

    for roi_name in os.listdir(base_folder):
        subdir = os.path.join(base_folder, roi_name)
        mhd_path = os.path.join(subdir, "result.mhd")

        if not os.path.isfile(mhd_path):
            print(f"⚠️  Ignorado: {roi_name} (sem result.mhd)")
            continue

        print(f"➡️ Convertendo {roi_name} ...")

        # ----------------------
        # Leitura da imagem MHD
        # ----------------------
        image = sitk.ReadImage(mhd_path)
        array = sitk.GetArrayFromImage(image)  # (z, y, x)

        # (z, y, x) -> (x, y, z)
        array = np.transpose(array, (2, 1, 0))

        spacing = image.GetSpacing()  # (x, y, z)
        origin = image.GetOrigin()    # (x, y, z)
        direction = np.array(image.GetDirection()).reshape(3, 3)

        print(f"  Dimensão corrigida: {array.shape}, Spacing: {spacing}, Origem: {origin}")
        # Diagnóstico rápido (det=-1 sugere reflexão/flip)
        try:
            print(f"  det(Direction) = {np.linalg.det(direction):.3f}")
        except Exception:
            pass

        # ----------------------
        # Binarização simples
        # ----------------------
        unique_vals = np.unique(array)
        print(f"  Valores únicos: {unique_vals}")

        if len(unique_vals) <= 1:
            print("⚠️  Sem dados binários válidos, pulando.")
            continue

        if np.array_equal(unique_vals, [0, 1]):
            binary = array
            level = 0.5
        else:
            threshold = np.mean(unique_vals)
            binary = (array > threshold).astype(np.uint8)
            level = 0.5

        # ----------------------
        # Marching Cubes
        # ----------------------
        verts, faces, normals, _ = measure.marching_cubes(
            binary, level=level, spacing=spacing
        )

        # ----------------------
        # Criação da malha PyVista
        # ----------------------
        faces_flat = np.hstack([[3, *face] for face in faces])
        mesh = pv.PolyData(verts, faces_flat)

        # APLICA ORIENTAÇÃO (Direction) + ORIGEM
        mesh.points = (mesh.points @ direction.T) + np.array(origin)

        # Opcional: converter LPS -> RAS (para bater com 3D Slicer)
        if CONVERTER_PARA_RAS:
            lps_to_ras = np.diag([-1, -1, 1])
            mesh.points = mesh.points @ lps_to_ras.T

        # ----------------------
        # Salvar como .vtp
        # ----------------------
        out_vtp = os.path.join(out_folder, f"{roi_name}.vtp")
        mesh.save(out_vtp)
        print(f"  ✅ Salvo: {out_vtp}")
