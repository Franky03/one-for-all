#!/usr/bin/env bash
# Downloads the trained student from the Modal volume and pushes to HF Hub.
# Usage: ./push_model.sh [checkpoint]   (default: final)
#
# The README is NOT pulled from the volume (the volume has the default HF
# template). It is read from ofa/space/../ofa_student_final_dir/README.md
# or from README_MODEL.md next to this script, whichever exists.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CKPT="${1:-final}"
VOLUME="ofa-student"
REMOTE_DIR="ofa_student/${CKPT}"
LOCAL_DIR="/home/kai/ofa_student_push"
HF_REPO="build-small-hackathon/deku"

# README to use — prefer README_MODEL.md next to this script, fall back to
# the copy that was previously downloaded from the volume.
README_SRC="${SCRIPT_DIR}/README_MODEL.md"
if [ ! -f "${README_SRC}" ]; then
    README_SRC="/home/kai/ofa_student_final_dir/README.md"
fi

echo "==> Downloading checkpoint '${CKPT}' from Modal volume '${VOLUME}'..."
rm -rf "${LOCAL_DIR}"
mkdir -p "${LOCAL_DIR}"

# modal volume get can't download a directory as a directory — fetch file by file.
FILES=$(modal volume ls "${VOLUME}" "${REMOTE_DIR}/" 2>&1 | grep -v '^$' | tail -n +1)
while IFS= read -r remote_path; do
    [ -z "${remote_path}" ] && continue
    filename=$(basename "${remote_path}")
    # Skip the template README that PEFT saves — we supply our own below.
    [ "${filename}" = "README.md" ] && continue
    echo "    ${filename}"
    modal volume get "${VOLUME}" "${remote_path}" "${LOCAL_DIR}/${filename}" --force
done <<< "${FILES}"

echo "==> Using README: ${README_SRC}"
cp "${README_SRC}" "${LOCAL_DIR}/README.md"

echo "==> Files in ${LOCAL_DIR}:"
ls -lh "${LOCAL_DIR}"

echo "==> Uploading to ${HF_REPO}..."
hf upload "${HF_REPO}" "${LOCAL_DIR}" --repo-type model

echo "==> Done. https://huggingface.co/${HF_REPO}"
