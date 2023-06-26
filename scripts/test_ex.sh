set -o errexit
set -o pipefail
set -o nounset
set -o xtrace

pushd "$(dirname "$0")/.." >/dev/null

python3 worker.py --trace config/test_ex.csv --log_path results/test_ex_results.csv

popd >/dev/null