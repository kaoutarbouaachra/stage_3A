#!/usr/bin/env bash
# ============================================================================
# run_campaign_tcg_cg.sh
# Column-generation comparison campaign: STELA vs CARLA, on setAP-001..125.
#
#   PHASE 1  TCG-Diag           stela (TCG_DIAG)        carla (TCG_DIAG_CARLA)
#   PHASE 2  TCG-Exact-Warm     stela  <- seed TCG_DIAG
#                               carla  <- seed TCG_DIAG_CARLA
#
# Diag runs FIRST for both encodings on every instance, so that each exact-warm
# run finds its own-encoding diagonal seed. STELA uses the TCG_diag/TCG_Exact
# packages (encoding defaults to stela, no flag needed); CARLA uses the
# hardcoded TCG_carla package.
# ============================================================================

set -u

############################# CONFIG  ###############################
# Absolute path to the package root (where TCG_Exact/, TCG_diag/, TCG_carla/,
# core/, stela/ live) and to the setAP dataset.
CODES_DIR="${CODES_DIR:-$HOME/stage_3A/codes/Algorithm_codes}"
DATA_DIR="${DATA_DIR:-../setAP}"          # relative to CODES_DIR, as in the -m usage

FIRST="${FIRST:-1}"                       # first instance index  (setAP-001)
LAST="${LAST:-125}"                       # last  instance index  (setAP-125)

RESUME="${RESUME:-1}"                     # 1 = skip runs whose summary.json exists
DRYRUN="${DRYRUN:-0}"                     # 1 = print commands, run nothing
KILL_AFTER="${KILL_AFTER:-60}"            # SIGKILL this many s after the SIGINT fallback

# ---- TCG-Diag budgets (per instance, per encoding) ----
DIAG_K="${DIAG_K:-8}"
DIAG_TIME_LIMIT="${DIAG_TIME_LIMIT:-600}" # soft global cap INSIDE the driver
DIAG_LP_TIME="${DIAG_LP_TIME:-120}"
DIAG_MIP_TIME="${DIAG_MIP_TIME:-180}"
DIAG_PRICING_TIME="${DIAG_PRICING_TIME:-180}"
DIAG_TOPK="${DIAG_TOPK:-5}"
DIAG_MAX_CG_ITER="${DIAG_MAX_CG_ITER:-20}"

# ---- TCG-Exact-Warm budgets (per instance, per encoding) ----
EW_WALL="${EW_WALL:-1800}"                # per-instance wall clock (stops between ranks)
EW_MIP="${EW_MIP:-180}"                   # cap for ONE rank MIP
EW_MIPGAP="${EW_MIPGAP:-1e-9}"
EW_MAXRANK="${EW_MAXRANK:-}"              # empty = all ranks
DIAG_EXT="${DIAG_EXT:-$(( DIAG_TIME_LIMIT + DIAG_MIP_TIME + 300 ))}"
EW_EXT="${EW_EXT:-$(( EW_WALL + 600 ))}"
#############################################################################

CAMP_LOG_ROOT="${CAMP_LOG_ROOT:-campaign_logs}"
MASTER_LOG="$CAMP_LOG_ROOT/campaign.log"

export PYTHONUNBUFFERED=1

cd "$CODES_DIR" || { echo "FATAL: cannot cd to CODES_DIR=$CODES_DIR"; exit 1; }
mkdir -p "$CAMP_LOG_ROOT"

ts()        { date '+%Y-%m-%d %H:%M:%S'; }
inst_name() { printf "setAP-%03d" "$1"; }
log_master(){ echo "[$(ts)] $*" | tee -a "$MASTER_LOG"; }

# run_one FOLDER INSTANCE EXT_TIMEOUT CMD...
run_one() {
  local folder="$1"; local inst="$2"; local ext="$3"; shift 3
  local logdir="$CAMP_LOG_ROOT/$folder"
  mkdir -p "$logdir"
  local logf="$logdir/$inst.log"
  local summary="results/$inst/$folder/$inst-summary.json"

  if [ "$RESUME" = "1" ] && [ -f "$summary" ]; then
    log_master "SKIP  $folder  $inst  (summary exists)"
    return 0
  fi
  if [ "$DRYRUN" = "1" ]; then
    echo "DRYRUN [$folder $inst]  timeout -s SIGINT -k $KILL_AFTER $ext $*"
    return 0
  fi

  log_master "START $folder  $inst"
  local t0=$SECONDS
  timeout -s SIGINT -k "$KILL_AFTER" "$ext" "$@" > "$logf" 2>&1
  local rc=$?
  local dt=$(( SECONDS - t0 ))
  if [ -f "$summary" ]; then
    log_master "DONE  $folder  $inst  rc=$rc  ${dt}s"
  else
    log_master "FAIL  $folder  $inst  rc=$rc  ${dt}s  (no summary -> see $logf)"
  fi
  return 0
}

on_int() { log_master "!! campaign received SIGINT/SIGTERM -- stopping after the current run"; exit 130; }
trap on_int INT TERM

# -------- worst-case time estimate (every run hitting its external cap) ------
N=$(( LAST - FIRST + 1 ))
WORST=$(( N*2*DIAG_EXT + N*2*EW_EXT ))
log_master "================ CAMPAIGN START ================"
log_master "instances    : $(inst_name "$FIRST") .. $(inst_name "$LAST")  (N=$N)"
log_master "codes_dir    : $CODES_DIR"
log_master "data_dir     : $DATA_DIR"
log_master "resume=$RESUME  dryrun=$DRYRUN"
log_master "diag  : K=$DIAG_K time_limit=${DIAG_TIME_LIMIT}s ext=${DIAG_EXT}s (x $((N*2)) runs)"
log_master "exact : wall=${EW_WALL}s mip=${EW_MIP}s ext=${EW_EXT}s (x $((N*2)) runs)"
log_master "worst-case wall time if EVERY run times out: ~$(( WORST/3600 )) h $(( (WORST%3600)/60 )) min"
log_master "(typical runs finish well before their cap; this is an upper bound)"

MAXRANK_ARG=()
[ -n "$EW_MAXRANK" ] && MAXRANK_ARG=(--max-rank "$EW_MAXRANK")

# ============================ PHASE 1: TCG-Diag ============================
log_master "---------------- PHASE 1: TCG-Diag (stela + carla) ----------------"
for i in $(seq "$FIRST" "$LAST"); do
  INST=$(inst_name "$i")

  # --- STELA diagonal  -> results/<inst>/TCG_DIAG/
  run_one TCG_DIAG "$INST" "$DIAG_EXT" \
    python3 -m TCG_diag.run_TCG_Diag "$INST" --data-dir "$DATA_DIR" \
      --K "$DIAG_K" --time-limit "$DIAG_TIME_LIMIT" \
      --lp-time "$DIAG_LP_TIME" --mip-time "$DIAG_MIP_TIME" \
      --pricing-time "$DIAG_PRICING_TIME" \
      --topk-cols "$DIAG_TOPK" --max-cg-iter "$DIAG_MAX_CG_ITER"

  # --- CARLA diagonal  -> results/<inst>/TCG_DIAG_CARLA/
  run_one TCG_DIAG_CARLA "$INST" "$DIAG_EXT" \
    python3 -m TCG_carla.run_diag "$INST" --data-dir "$DATA_DIR" \
      --K "$DIAG_K" --time-limit "$DIAG_TIME_LIMIT" \
      --lp-time "$DIAG_LP_TIME" --mip-time "$DIAG_MIP_TIME" \
      --pricing-time "$DIAG_PRICING_TIME" \
      --topk-cols "$DIAG_TOPK" --max-cg-iter "$DIAG_MAX_CG_ITER"
done

# ====================== PHASE 2: TCG-Exact-Warm ======================
log_master "------------- PHASE 2: TCG-Exact-Warm (warm from diag) -------------"
for i in $(seq "$FIRST" "$LAST"); do
  INST=$(inst_name "$i")

  # --- STELA exact, warm-started from the STELA diagonal seed ---
  run_one TCG_EXACT_WARM "$INST" "$EW_EXT" \
    python3 -m TCG_Exact.run_TCG_exact_warm "$INST" --data-dir "$DATA_DIR" \
      --wall-time "$EW_WALL" --time-limit "$EW_MIP" --mip-gap "$EW_MIPGAP" \
      --warm-algo TCG_DIAG "${MAXRANK_ARG[@]}"

  # --- CARLA exact, warm-started from the CARLA diagonal seed ---
  run_one TCG_EXACT_WARM_CARLA "$INST" "$EW_EXT" \
    python3 -m TCG_carla.run_exact_warm "$INST" --data-dir "$DATA_DIR" \
      --wall-time "$EW_WALL" --time-limit "$EW_MIP" --mip-gap "$EW_MIPGAP" \
      --warm-algo TCG_DIAG_CARLA "${MAXRANK_ARG[@]}"
done

log_master "================ CAMPAIGN DONE ================"
