#!/bin/bash

# Lista de pacientes
pacientes=("p1" "p2" "p7" "p10")

for p in "${pacientes[@]}"; do
    echo "========== PROCESSANDO PACIENTE $p =========="
    
    input_folder="coste_nii"              # pasta com os .nii.gz de entrada
    output_folder="output_transformix_desenhos_${p}" # nova pasta de saída
    param_folder="output_elastix_${p}"       # pasta com os parametros

    # Cria pasta de saída
    mkdir -p "$output_folder"

    # Loop em todos os .nii.gz do paciente
    for file in ${input_folder}/*.nii.gz; do
        filename=$(basename "$file" .nii.gz)
        echo "➡️ Rodando transformix para $filename ..."
        
        mkdir -p "${output_folder}/${filename}"
        
        transformix -in "$file" \
                    -out "${output_folder}/${filename}" \
                    -tp "${param_folder}/TransformParameters.2.txt"
    done
done
