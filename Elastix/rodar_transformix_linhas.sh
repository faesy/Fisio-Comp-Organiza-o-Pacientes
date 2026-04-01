#!/usr/bin/env bash
# run_transformix_linhas.sh
# Executa transformix para volumes NIfTI de "linhas" (máscaras) em múltiplos pacientes.
# Requer: transformix no PATH (Elastix/Transformix instalado)

set -u  # erro em variável não definida

# -------- Configurações --------
PATIENTES=("p1" "p2" "p7" "p10")

# Sempre resolve caminhos a partir da pasta onde o script está
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

INPUT_FOLDER="${SCRIPT_DIR}/linhas_nii"                 # onde estão os .nii.gz das linhas
OUTPUT_ROOT="${SCRIPT_DIR}/output_transformix_linhas"   # raiz de saída (será organizado por paciente/arquivo)
PARAM_ROOT="${SCRIPT_DIR}"                              # raiz onde ficam os "output_elastix_${paciente}"

# -------- Checagens --------
if ! command -v transformix >/dev/null 2>&1; then
  echo "ERRO: 'transformix' não encontrado no PATH. Instale/adicione ao PATH e tente novamente."
  exit 1
fi

if [[ ! -d "$INPUT_FOLDER" ]]; then
  echo "ERRO: pasta de entrada não existe: $INPUT_FOLDER"
  exit 1
fi

shopt -s nullglob
NIFTIS=( "${INPUT_FOLDER}"/*.nii.gz )
if (( ${#NIFTIS[@]} == 0 )); then
  echo "ERRO: não encontrei nenhum .nii.gz em: $INPUT_FOLDER"
  exit 1
fi

mkdir -p "$OUTPUT_ROOT"

# -------- Loop principal --------
for P in "${PATIENTES[@]}"; do
  echo "========== PROCESSANDO PACIENTE: ${P} =========="

  PARAM_FOLDER="${PARAM_ROOT}/output_elastix_${P}"

  if [[ ! -d "$PARAM_FOLDER" ]]; then
    echo "ERRO: pasta de parâmetros não existe: $PARAM_FOLDER"
    echo "      (você está rodando do lugar certo? confira onde está o output_elastix_${P})"
    continue
  fi

  # Monta lista de TransformParameters.*.txt em ordem (0,1,2,...)
  LAST_TP=""
  i=0
  while [[ -f "${PARAM_FOLDER}/TransformParameters.${i}.txt" ]]; do
    LAST_TP="${PARAM_FOLDER}/TransformParameters.${i}.txt"
    ((i++))
  done

  if [[ -z "$LAST_TP" ]]; then
    echo "ERRO: não encontrei TransformParameters.0.txt (ou sequência) em: ${PARAM_FOLDER}"
    echo "      Conteúdo da pasta:"
    ls -la "$PARAM_FOLDER"
    continue
  fi

  echo "Usando TP final: $LAST_TP"

  # Loop em todos os NIfTIs
  for FILE in "${NIFTIS[@]}"; do
    BASENAME="$(basename "$FILE" .nii.gz)"
    OUT_DIR="${OUTPUT_ROOT}/${P}/${BASENAME}"
    mkdir -p "$OUT_DIR"

    echo "➡️  ${P} :: transformix em '${BASENAME}.nii.gz' ..."
    LOGFILE="${OUT_DIR}/transformix.log"

    # Executa transformix (CORRETO: -tp)
    if transformix -in "$FILE" -out "$OUT_DIR" -tp "$LAST_TP" > "$LOGFILE" 2>&1; then
      echo "✅ Concluído: ${OUT_DIR}"
    else
      echo "⚠️  Falha no transformix para ${P}/${BASENAME}. Veja o log: ${LOGFILE}"
      tail -n 30 "$LOGFILE" 2>/dev/null || true
    fi
  done
done

echo "🚀 Fim."
