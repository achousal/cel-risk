#!/usr/bin/env bash
# submit_experiment.sh — Unified LSF orchestrator for V0 gate and factorial experiments
#
# Generates per-cell runner scripts with deterministic --run-id and --outdir,
# writes provenance metadata, then submits an LSF array job.
#
# Usage:
#   # V0 gate (120 cells)
#   bash submit_experiment.sh \
#       --experiment v0_gate \
#       --manifest analysis/configs/recipes/v0/v0_cell_manifest.csv \
#       --results-root results/v0_gate \
#       --seeds 100-119
#
#   # Full factorial (1,566 cells)
#   bash submit_experiment.sh \
#       --experiment factorial \
#       --manifest analysis/configs/recipes/cell_manifest.csv \
#       --results-root results/factorial \
#       --seeds 100-119
#
#   # Options
#       --wall 48:00         LSF wall time (default: 48:00)
#       --cores 12           Cores per cell (default: 12)
#       --mem 8000           MB per core (default: 8000)
#       --queue premium      LSF queue (default: premium)
#       --project acc_vascbrain  LSF project (default: acc_vascbrain)
#       --cells 1-10         Submit only cell_ids 1-10 (default: all)
#       --dry-run            Generate scripts + README but do not bsub

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
EXPERIMENT=""
MANIFEST=""
RESULTS_ROOT=""
SEEDS_START=100
SEEDS_END=119
WALL="48:00"
CORES=12
MEM=8000
QUEUE="premium"
PROJECT="acc_vascbrain"
CELL_RANGE=""
DRY_RUN=false

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --experiment)   EXPERIMENT="$2";    shift 2;;
        --manifest)     MANIFEST="$2";      shift 2;;
        --results-root) RESULTS_ROOT="$2";  shift 2;;
        --seeds)
            SEEDS_START="${2%%-*}"
            SEEDS_END="${2##*-}"
            shift 2;;
        --wall)         WALL="$2";          shift 2;;
        --cores)        CORES="$2";         shift 2;;
        --mem)          MEM="$2";           shift 2;;
        --queue)        QUEUE="$2";         shift 2;;
        --project)      PROJECT="$2";       shift 2;;
        --cells)        CELL_RANGE="$2";    shift 2;;
        --dry-run)      DRY_RUN=true;       shift;;
        *)
            echo "Unknown option: $1" >&2
            exit 1;;
    esac
done

# Validate required args
if [[ -z "$EXPERIMENT" || -z "$MANIFEST" || -z "$RESULTS_ROOT" ]]; then
    echo "ERROR: --experiment, --manifest, and --results-root are required" >&2
    exit 1
fi

if [[ ! -f "$MANIFEST" ]]; then
    echo "ERROR: Manifest not found: $MANIFEST" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Parse manifest
# ---------------------------------------------------------------------------
HEADER=$(head -1 "$MANIFEST")

col_idx() {
    echo "$HEADER" | awk -F',' -v col="$1" '{for(i=1;i<=NF;i++) if($i==col) print i}'
}

COL_CELL_ID=$(col_idx "cell_id")
COL_RECIPE=$(col_idx "recipe_id")
COL_MODEL=$(col_idx "model")
COL_CELL_NAME=$(col_idx "cell_name")
COL_PIPELINE=$(col_idx "pipeline_config")

# Optional columns (V0-specific)
COL_STRATEGY=$(col_idx "strategy")
COL_CTRL_RATIO=$(col_idx "control_ratio")
COL_CALIBRATION=$(col_idx "calibration")
COL_WEIGHTING=$(col_idx "weighting")
COL_DOWNSAMPLING=$(col_idx "downsampling")

if [[ -z "$COL_CELL_ID" || -z "$COL_RECIPE" || -z "$COL_MODEL" || -z "$COL_CELL_NAME" || -z "$COL_PIPELINE" ]]; then
    echo "ERROR: Manifest missing required columns (cell_id, recipe_id, model, cell_name, pipeline_config)" >&2
    echo "Header: $HEADER" >&2
    exit 1
fi

N_CELLS=$(( $(wc -l < "$MANIFEST") - 1 ))
N_SEEDS=$(( SEEDS_END - SEEDS_START + 1 ))

# Determine cell ID range
if [[ -n "$CELL_RANGE" ]]; then
    RANGE_START="${CELL_RANGE%%-*}"
    RANGE_END="${CELL_RANGE##*-}"
else
    RANGE_START=1
    RANGE_END="$N_CELLS"
fi

echo "============================================================"
echo "Experiment:    $EXPERIMENT"
echo "Manifest:      $MANIFEST ($N_CELLS cells)"
echo "Results root:  $RESULTS_ROOT"
echo "Seeds:         $SEEDS_START-$SEEDS_END ($N_SEEDS per cell)"
echo "Cell range:    $RANGE_START-$RANGE_END"
echo "Resources:     $CORES cores, ${MEM}MB/core, wall=$WALL, queue=$QUEUE"
echo "Dry run:       $DRY_RUN"
echo "============================================================"

# ---------------------------------------------------------------------------
# Create directories
# ---------------------------------------------------------------------------
RUNNERS_DIR="logs/${EXPERIMENT}_runners"
mkdir -p "$RUNNERS_DIR"
mkdir -p "$RESULTS_ROOT"

# ---------------------------------------------------------------------------
# Write experiment README.md
# ---------------------------------------------------------------------------
SUBMIT_TS=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
README_PATH="${RESULTS_ROOT}/README.md"

cat > "$README_PATH" << READMEEOF
# ${EXPERIMENT} Results

experiment: ${EXPERIMENT}
manifest: ${MANIFEST}
submitted: ${SUBMIT_TS}
seeds: ${SEEDS_START}-${SEEDS_END} (${N_SEEDS} seeds per cell)
cells: ${N_CELLS}
cell_range_submitted: ${RANGE_START}-${RANGE_END}
layout: {recipe_id}/{cell_name}/run_{recipe_id}__{cell_name}/

## Manifest Columns
$(echo "$HEADER" | tr ',' '\n' | sed 's/^/- /')

## Provenance
- Per-cell metadata: cell_manifest_entry.json (inside each run dir)
- Pipeline metadata: run_metadata.json (pipeline-written, per run dir)
- Per-split metadata: config_metadata.json (pipeline-written, per split)

## Resources
- Cores per cell: ${CORES}
- Memory per core: ${MEM} MB
- Wall time: ${WALL}
- Queue: ${QUEUE}
- Project: ${PROJECT}
READMEEOF

echo "Wrote $README_PATH"

# ---------------------------------------------------------------------------
# Generate per-cell runner scripts
# ---------------------------------------------------------------------------
echo "Generating runner scripts..."

for CELL_ID in $(seq "$RANGE_START" "$RANGE_END"); do
    ROW=$(awk -F',' -v id="$CELL_ID" -v col="$COL_CELL_ID" 'NR>1 && $col==id' "$MANIFEST")
    if [[ -z "$ROW" ]]; then
        echo "WARNING: Cell ID $CELL_ID not found in manifest, skipping" >&2
        continue
    fi

    RECIPE_ID=$(echo "$ROW" | awk -F',' -v col="$COL_RECIPE" '{print $col}')
    MODEL=$(echo "$ROW" | awk -F',' -v col="$COL_MODEL" '{print $col}')
    CELL_NAME=$(echo "$ROW" | awk -F',' -v col="$COL_CELL_NAME" '{print $col}')
    PIPELINE_CONFIG=$(echo "$ROW" | awk -F',' -v col="$COL_PIPELINE" '{print $col}')

    # Build run_id: recipe__cell_name (unique across experiment)
    RUN_ID="${RECIPE_ID}__${CELL_NAME}"
    OUTDIR="${RESULTS_ROOT}/${RECIPE_ID}/${CELL_NAME}"

    # Build cell_manifest_entry.json content
    # Start with required fields
    MANIFEST_JSON="{\"cell_id\": ${CELL_ID}, \"experiment\": \"${EXPERIMENT}\", \"recipe_id\": \"${RECIPE_ID}\", \"model\": \"${MODEL}\", \"cell_name\": \"${CELL_NAME}\""

    # Add optional fields if columns exist
    if [[ -n "$COL_STRATEGY" ]]; then
        STRATEGY=$(echo "$ROW" | awk -F',' -v col="$COL_STRATEGY" '{print $col}')
        MANIFEST_JSON="${MANIFEST_JSON}, \"strategy\": \"${STRATEGY}\""
    fi
    if [[ -n "$COL_CTRL_RATIO" ]]; then
        CTRL_RATIO=$(echo "$ROW" | awk -F',' -v col="$COL_CTRL_RATIO" '{print $col}')
        MANIFEST_JSON="${MANIFEST_JSON}, \"control_ratio\": ${CTRL_RATIO}"
    fi
    if [[ -n "$COL_CALIBRATION" ]]; then
        CALIBRATION=$(echo "$ROW" | awk -F',' -v col="$COL_CALIBRATION" '{print $col}')
        MANIFEST_JSON="${MANIFEST_JSON}, \"calibration\": \"${CALIBRATION}\""
    fi
    if [[ -n "$COL_WEIGHTING" ]]; then
        WEIGHTING=$(echo "$ROW" | awk -F',' -v col="$COL_WEIGHTING" '{print $col}')
        MANIFEST_JSON="${MANIFEST_JSON}, \"weighting\": \"${WEIGHTING}\""
    fi
    if [[ -n "$COL_DOWNSAMPLING" ]]; then
        DOWNSAMPLING=$(echo "$ROW" | awk -F',' -v col="$COL_DOWNSAMPLING" '{print $col}')
        MANIFEST_JSON="${MANIFEST_JSON}, \"downsampling\": ${DOWNSAMPLING}"
    fi

    MANIFEST_JSON="${MANIFEST_JSON}, \"seeds\": [${SEEDS_START}, ${SEEDS_END}], \"pipeline_config\": \"${PIPELINE_CONFIG}\", \"submit_timestamp\": \"${SUBMIT_TS}\"}"

    # Write runner script
    RUNNER="${RUNNERS_DIR}/cell_${CELL_ID}.sh"
    cat > "$RUNNER" << RUNNEREOF
#!/usr/bin/env bash
set -euo pipefail
cd /sc/arion/projects/vascbrain/andres/cel-risk
source analysis/.venv/bin/activate
export OPENBLAS_NUM_THREADS=\${LSB_DJOB_NUMPROC:-${CORES}}

RECIPE_ID="${RECIPE_ID}"
CELL_NAME="${CELL_NAME}"
RUN_ID="${RUN_ID}"
OUTDIR="${OUTDIR}"
PIPELINE_CONFIG="${PIPELINE_CONFIG}"

echo "============================================================"
echo "Cell ${CELL_ID}: \${CELL_NAME}"
echo "Recipe: \${RECIPE_ID} | Run ID: \${RUN_ID}"
echo "Output: \${OUTDIR}/run_\${RUN_ID}/"
echo "Seeds: ${SEEDS_START}-${SEEDS_END}"
echo "============================================================"

# Write provenance metadata before first seed
mkdir -p "\${OUTDIR}/run_\${RUN_ID}"
cat > "\${OUTDIR}/run_\${RUN_ID}/cell_manifest_entry.json" << 'MANIFESTEOF'
${MANIFEST_JSON}
MANIFESTEOF

for SEED in \$(seq ${SEEDS_START} ${SEEDS_END}); do
    echo "--- Cell ${CELL_ID} (\${CELL_NAME}) Seed \$SEED ---"
    ced run-pipeline \\
        --pipeline-config "\${PIPELINE_CONFIG}" \\
        --split-seeds "\$SEED" \\
        --run-id "\${RUN_ID}" \\
        --outdir "\${OUTDIR}"
done

echo "Cell ${CELL_ID} complete."
RUNNEREOF
    chmod +x "$RUNNER"
done

echo "Generated $((RANGE_END - RANGE_START + 1)) runner scripts in $RUNNERS_DIR/"

# ---------------------------------------------------------------------------
# Submit LSF array
# ---------------------------------------------------------------------------
if [[ "$DRY_RUN" == "true" ]]; then
    echo ""
    echo "DRY RUN — not submitting. To submit manually:"
    echo "  bsub -J \"${EXPERIMENT}[${RANGE_START}-${RANGE_END}]\" \\"
    echo "       -P ${PROJECT} -q ${QUEUE} -n ${CORES} -W ${WALL} \\"
    echo "       -R \"rusage[mem=${MEM}] span[hosts=1]\" \\"
    echo "       -o logs/${EXPERIMENT}_%J_%I.out \\"
    echo "       -e logs/${EXPERIMENT}_%J_%I.err \\"
    echo "       ${RUNNERS_DIR}/cell_\\\$LSB_JOBINDEX.sh"
    echo ""
    echo "Sample runner (cell ${RANGE_START}):"
    cat "${RUNNERS_DIR}/cell_${RANGE_START}.sh"
    exit 0
fi

echo ""
echo "Submitting LSF array [${RANGE_START}-${RANGE_END}]..."

bsub -J "${EXPERIMENT}[${RANGE_START}-${RANGE_END}]" \
     -P "$PROJECT" \
     -q "$QUEUE" \
     -n "$CORES" \
     -W "$WALL" \
     -R "rusage[mem=${MEM}] span[hosts=1]" \
     -o "logs/${EXPERIMENT}_%J_%I.out" \
     -e "logs/${EXPERIMENT}_%J_%I.err" \
     "${RUNNERS_DIR}/cell_\$LSB_JOBINDEX.sh"

echo "Submitted. Monitor with: bjobs -w -u \$USER | grep ${EXPERIMENT}"
