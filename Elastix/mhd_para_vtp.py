import os
import SimpleITK as sitk
import numpy as np
from skimage import measure
import pyvista as pv

# ========================
# CONFIGURAÇÕES
# ========================
pacientes = ["p1", "p2", "p7", "p10"]

for p in pacientes:
    print(f"\n========== CONVERTENDO PACIENTE {p} ==========")

    base_folder = f"output_transformix_desenhos_{p}"
    out_folder = f"vtp_desenhos_{p}"  # pasta final de saída
    os.makedirs(out_folder, exist_ok=True)

    # Percorre todas as subpastas (costelas, esterno, etc.)
    for estrutura in os.listdir(base_folder):
        subdir = os.path.join(base_folder, estrutura)
        mhd_path = os.path.join(subdir, "result.mhd")

        if not os.path.isfile(mhd_path):
            print(f"⚠️  Ignorado: {estrutura} (sem result.mhd)")
            continue

        print(f"➡️ Convertendo {estrutura} ...")

        # ----------------------
        # Leitura da imagem MHD
        # ----------------------
        image = sitk.ReadImage(mhd_path)
        array = sitk.GetArrayFromImage(image)  # (z, y, x) por padrão

        # Corrigir eixos: (z, y, x) -> (x, y, z)
        array = np.transpose(array, (2, 1, 0))

        spacing = image.GetSpacing()           # (x, y, z)
        origin = image.GetOrigin()             # (x, y, z)

        spacing_corrected = (spacing[0], spacing[1], spacing[2])
        origin_corrected = (origin[0], origin[1], origin[2])

        print(f"  Dimensão corrigida: {array.shape}, Spacing: {spacing_corrected}, Origem: {origin_corrected}")

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
            binary, level=level, spacing=spacing_corrected
        )

        # ----------------------
        # Criação da malha PyVista
        # ----------------------
        faces_flat = np.hstack([[3, *face] for face in faces])
        direction = np.array(image.GetDirection()).reshape(3,3)  # aplica orientação ITK
        mesh = pv.PolyData(verts, faces_flat)
        mesh.points = (mesh.points @ direction.T) + np.array(origin_corrected)


        # ----------------------
        # Salvar como .vtp
        # ----------------------
        out_vtp = os.path.join(out_folder, f"{estrutura}.vtp")
        mesh.save(out_vtp)
        print(f"  ✅ Salvo: {out_vtp}")
