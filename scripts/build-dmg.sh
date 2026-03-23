#!/bin/zsh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DERIVED_DATA_PATH="${DERIVED_DATA_PATH:-/tmp/LocalWisprReleaseDerived}"
APP_NAME="${APP_NAME:-LocalWisprHost.app}"
SCHEME="${SCHEME:-LocalWisprHost}"
PROJECT_PATH="${PROJECT_PATH:-$ROOT_DIR/AppHost/LocalWisprHost.xcodeproj}"
BUILD_PRODUCTS_DIR="$DERIVED_DATA_PATH/Build/Products/Release"
STAGING_DIR="$ROOT_DIR/dist/dmg-root"
DMG_BASENAME="${DMG_BASENAME:-LocalWispr}"
DMG_PATH="$ROOT_DIR/dist/${DMG_BASENAME}.dmg"
CHECKSUM_PATH="$ROOT_DIR/dist/${DMG_BASENAME}.dmg.sha256"
VOLUME_NAME="${VOLUME_NAME:-LocalWispr}"

echo "Building Release app..."
xcodebuild \
  -project "$PROJECT_PATH" \
  -scheme "$SCHEME" \
  -configuration Release \
  -derivedDataPath "$DERIVED_DATA_PATH" \
  build

APP_PATH="$BUILD_PRODUCTS_DIR/$APP_NAME"
if [[ ! -d "$APP_PATH" ]]; then
  echo "Expected app not found at $APP_PATH" >&2
  exit 1
fi

echo "Preparing DMG staging directory..."
rm -rf "$STAGING_DIR" "$DMG_PATH" "$CHECKSUM_PATH"
mkdir -p "$STAGING_DIR"
ditto "$APP_PATH" "$STAGING_DIR/$APP_NAME"
ln -s /Applications "$STAGING_DIR/Applications"

echo "Creating DMG..."
hdiutil create \
  -volname "$VOLUME_NAME" \
  -srcfolder "$STAGING_DIR" \
  -ov \
  -format UDZO \
  "$DMG_PATH"

(
  cd "$ROOT_DIR/dist"
  shasum -a 256 "${DMG_BASENAME}.dmg" >"${DMG_BASENAME}.dmg.sha256"
)

echo "DMG created at $DMG_PATH"
echo "SHA256: $(cut -d' ' -f1 < "$CHECKSUM_PATH")"
echo "Checksum file: $CHECKSUM_PATH"
