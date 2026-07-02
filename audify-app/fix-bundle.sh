#!/bin/zsh
# Post-build fix for the bundled engine + DMG creation.
#
# Tauri's resource bundler dereferences symlinks. PyInstaller's dist relies
# on _internal/libmlx.dylib -> mlx/lib/libmlx.dylib (and friends); with a
# dereferenced copy, mlx loads the duplicate dylib and then can't find
# mlx.metallib next to it ("Failed to load the default metallib").
# This script restores the symlinks, re-signs ad hoc, and builds the DMG.
#
# Usage: ./fix-bundle.sh   (after `npm run tauri build`)

set -e
APP="$(dirname "$0")/src-tauri/target/release/bundle/macos/Audify.app"
INTERNAL="$APP/Contents/Resources/engine/_internal"

for pair in \
  "libjaccl.dylib:mlx/lib/libjaccl.dylib" \
  "libmlx.dylib:mlx/lib/libmlx.dylib" \
  "libmupdfcpp.so:pymupdf/libmupdfcpp.so" \
  "libmupdf.dylib:pymupdf/libmupdf.dylib"
do
  link="${pair%%:*}"; target="${pair#*:}"
  rm -f "$INTERNAL/$link"
  ln -s "$target" "$INTERNAL/$link"
done

codesign --force --deep -s - "$APP"

DMG="$(dirname "$0")/src-tauri/target/release/bundle/dmg/Audify_standalone.dmg"
mkdir -p "$(dirname "$DMG")"
rm -f "$DMG"
hdiutil create -volname Audify -srcfolder "$APP" -ov -format UDZO "$DMG"

echo "Fixed app: $APP"
echo "DMG:       $DMG"
