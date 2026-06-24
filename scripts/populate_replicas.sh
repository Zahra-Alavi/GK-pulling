#!/usr/bin/env bash
# Create new pre-populated replica directories from an existing input template.
# Copies only inputs needed by eq_start.sh; completed simulation outputs are not copied.
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

START_REP="${1:-21}"
END_REP="${2:-50}"
TEMPLATE_REP="${3:-$BASE_DIR/rep_01}"

[[ "$START_REP" =~ ^[0-9]+$ ]] || { echo "ERROR: START_REP must be an integer"; exit 1; }
[[ "$END_REP" =~ ^[0-9]+$ ]] || { echo "ERROR: END_REP must be an integer"; exit 1; }
(( START_REP >= 1 && START_REP <= END_REP && END_REP <= 99 )) || {
  echo "ERROR: require 1 <= START_REP <= END_REP <= 99"
  exit 1
}

[[ -d "$TEMPLATE_REP" ]] || { echo "ERROR: template replica not found: $TEMPLATE_REP"; exit 1; }

required_files=(npt.gro topol.top index.ndx eq_start_k1000.mdp)
for name in "${required_files[@]}"; do
  [[ -f "$TEMPLATE_REP/$name" ]] || {
    echo "ERROR: template is missing required input: $TEMPLATE_REP/$name"
    exit 1
  }
done

shopt -s nullglob
topology_inputs=("$TEMPLATE_REP"/*.itp)
force_fields=("$TEMPLATE_REP"/*.ff)
shopt -u nullglob

(( ${#topology_inputs[@]} > 0 )) || {
  echo "ERROR: template has no .itp topology inputs: $TEMPLATE_REP"
  exit 1
}
(( ${#force_fields[@]} > 0 )) || {
  echo "ERROR: template has no .ff force-field directory: $TEMPLATE_REP"
  exit 1
}

for ((i = START_REP; i <= END_REP; i++)); do
  printf -v rep_name "rep_%02d" "$i"
  rep="$BASE_DIR/$rep_name"

  if [[ -e "$rep" ]]; then
    echo "ERROR: $rep already exists; refusing to overwrite it"
    exit 1
  fi

  echo "Creating $rep_name from inputs in $(basename "$TEMPLATE_REP")"
  mkdir "$rep"
  cp -p "$TEMPLATE_REP/npt.gro" "$rep/"
  cp -p "$TEMPLATE_REP/topol.top" "$rep/"
  cp -p "$TEMPLATE_REP/index.ndx" "$rep/"
  cp -p "$TEMPLATE_REP/eq_start_k1000.mdp" "$rep/"
  cp -p "${topology_inputs[@]}" "$rep/"

  for force_field in "${force_fields[@]}"; do
    cp -aL "$force_field" "$rep/"
  done
done

echo "Created replicas $(printf 'rep_%02d' "$START_REP") through $(printf 'rep_%02d' "$END_REP")."
echo "Next: $BASE_DIR/eq_start.sh $START_REP $END_REP"
