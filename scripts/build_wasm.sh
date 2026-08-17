#!/usr/bin/env bash
# Reproduce src/googleaisearch2api/vendor/{quickjs.wasm,crypto.so,encoding.so}
# from the immutable npm tarball of quickjs-wasi (vercel-labs/quickjs-wasi).
# Deterministic: exact tarball URL + sha256 pin; the tarball version is never
# re-published, so the extracted artifacts are byte-identical every run.
set -euo pipefail

PACKAGE="quickjs-wasi"
VERSION="3.4.0"
TARBALL_SHA256="b3ced715004545529995551489b1cb449650b410af81d1044beb4ba1cfb52095"
TARBALL_URL="https://registry.npmjs.org/${PACKAGE}/-/${PACKAGE}-${VERSION}.tgz"

VENDOR_DIR="$(cd "$(dirname "$0")/../src/googleaisearch2api/vendor" && pwd)"
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT

echo "== fetching ${TARBALL_URL}"
curl -fsSL "${TARBALL_URL}" -o "${WORK_DIR}/${PACKAGE}.tgz"

echo "== verifying tarball sha256"
echo "${TARBALL_SHA256}  ${WORK_DIR}/${PACKAGE}.tgz" | sha256sum -c -

echo "== extracting"
tar -xzf "${WORK_DIR}/${PACKAGE}.tgz" -C "${WORK_DIR}" --strip-components=1 \
    package/quickjs.wasm \
    package/extensions/crypto/crypto.so \
    package/extensions/encoding/encoding.so

echo "== installing into ${VENDOR_DIR}"
install -m 0644 "${WORK_DIR}/quickjs.wasm" "${VENDOR_DIR}/quickjs.wasm"
install -m 0644 "${WORK_DIR}/extensions/crypto/crypto.so" "${VENDOR_DIR}/crypto.so"
install -m 0644 "${WORK_DIR}/extensions/encoding/encoding.so" "${VENDOR_DIR}/encoding.so"

echo "== writing checksums"
( cd "${VENDOR_DIR}" && sha256sum quickjs.wasm crypto.so encoding.so > SHA256SUMS )

echo "== done; vendored artifacts:"
( cd "${VENDOR_DIR}" && ls -la quickjs.wasm crypto.so encoding.so SHA256SUMS )