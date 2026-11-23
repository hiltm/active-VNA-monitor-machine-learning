#!/usr/bin/env bash
#
# package_for_pi.sh - bundle the minimal RF inference package for a Raspberry Pi 5
#
# produces a single tarball containing only what is needed to classify .s2p
# measurements on the Pi: the inference script, the Random Forest model, the
# scaler, metadata, and pinned dependencies
#
# the 2.0 GB KNN model and 51 MB SVM model are intentionally EXCLUDED — the
# Random Forest model (19 MB) has the best accuracy and lightest footprint.
#
# usage:
#   ./package_for_pi.sh
#   scp chiptamper_pi_bundle.tar.gz pi@raspberrypi.local:~
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"   # scripts live in deploy/; models/ is at the repo root

BUNDLE_NAME="chiptamper_pi_bundle"
STAGING="$(mktemp -d)/${BUNDLE_NAME}"
mkdir -p "${STAGING}/models"

echo "Staging inference files..."
cp "${SCRIPT_DIR}/classify_measurement.py"           "${STAGING}/"
cp "${SCRIPT_DIR}/requirements-inference.txt"         "${STAGING}/"
cp "${REPO_ROOT}/models/rf_model.joblib"             "${STAGING}/models/"
cp "${REPO_ROOT}/models/scaler.joblib"               "${STAGING}/models/"
cp "${REPO_ROOT}/models/training_metadata.joblib"    "${STAGING}/models/"

# minimal README for the Pi
cat > "${STAGING}/README_PI.md" <<'EOF'
# chiptamper RF classifier — Raspberry Pi inference

## one-time setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-inference.txt
```

## run
```bash
source .venv/bin/activate
# classify a single measurement (Random Forest is the only bundled model)
python classify_measurement.py path/to/measurement.s2p --model rf --verbose

# classify a whole directory
python classify_measurement.py path/to/folder/ --model rf --batch-summary
```

if you see an InconsistentVersionWarning from scikit-learn, the installed
version does not match the one the model was pickled with. Verify with:
```bash
python -c "import sklearn; print(sklearn.__version__)"   # expect 1.5.1
```
EOF

echo "Creating tarball..."
tar -czf "${REPO_ROOT}/${BUNDLE_NAME}.tar.gz" -C "$(dirname "$STAGING")" "${BUNDLE_NAME}"
rm -rf "$(dirname "$STAGING")"

echo ""
echo "Created: ${REPO_ROOT}/${BUNDLE_NAME}.tar.gz"
du -h "${REPO_ROOT}/${BUNDLE_NAME}.tar.gz"
echo ""
echo "Transfer to the Pi with:"
echo "  scp ${BUNDLE_NAME}.tar.gz pi@raspberrypi.local:~"
echo "Then on the Pi:"
echo "  tar -xzf ${BUNDLE_NAME}.tar.gz && cd ${BUNDLE_NAME}"
echo "  cat README_PI.md"
