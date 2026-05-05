#!/usr/bin/env bash
#
# Build CLiteRTLM.xcframework from Google's LiteRT-LM source.
#
# Prerequisites:
#   - Bazel 7.6.1 (install via Bazelisk: brew install bazelisk)
#   - Xcode 16+ with iOS SDK
#   - ~20 GB disk space for Bazel build cache
#
# Usage:
#   ./scripts/build-xcframework.sh [/path/to/LiteRT-LM]
#
# If no path is provided, clones the repo to a temp directory.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="$PROJECT_DIR/Frameworks/LiteRTLM.xcframework"
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 1. Locate or clone LiteRT-LM source
# ---------------------------------------------------------------------------

LITERT_LM_DIR="${1:-}"

if [ -z "$LITERT_LM_DIR" ]; then
    LITERT_LM_DIR="$WORK_DIR/LiteRT-LM"
    info "Cloning LiteRT-LM source..."
    git clone --depth 1 https://github.com/google-ai-edge/LiteRT-LM.git "$LITERT_LM_DIR"
fi

if [ ! -f "$LITERT_LM_DIR/c/BUILD" ]; then
    error "Invalid LiteRT-LM source directory: $LITERT_LM_DIR (missing c/BUILD)"
fi

# Resolve to absolute path (relative paths break after `cd` into the source dir)
LITERT_LM_DIR="$(cd "$LITERT_LM_DIR" && pwd)"

info "Using LiteRT-LM source at: $LITERT_LM_DIR"

# ---------------------------------------------------------------------------
# 1b. Patch upstream BUILD if needed
# ---------------------------------------------------------------------------
# Two patches may be needed depending on the upstream version:
#
# 1. ios_engine.bzl stub — HEAD's c/BUILD loads `:ios_engine.bzl` which isn't
#    shipped yet. Without the stub Bazel can't parse the BUILD file at all.
#
# 2. cc_binary dylib target — releases up to v0.10.2 only define cc_library
#    targets (:engine, :engine_cpu). The cc_binary that produces the shared
#    library was added later. We append it if missing.

if [ ! -f "$LITERT_LM_DIR/c/ios_engine.bzl" ] && grep -q 'ios_engine\.bzl' "$LITERT_LM_DIR/c/BUILD"; then
    info "Creating stub ios_engine.bzl (missing from upstream)..."
    cat > "$LITERT_LM_DIR/c/ios_engine.bzl" << 'STUB'
"""Stub for ios_shared_engine macro (not yet published upstream)."""

def ios_shared_engine(**kwargs):
    pass
STUB
fi

if ! grep -q 'libLiteRTLMEngine\.dylib' "$LITERT_LM_DIR/c/BUILD"; then
    info "Adding libLiteRTLMEngine.dylib target (not present in this version)..."
    cat >> "$LITERT_LM_DIR/c/BUILD" << 'BUILD_PATCH'

cc_binary(
    name = "libLiteRTLMEngine.dylib",
    srcs = [
        "engine.cc",
        "engine.h",
        "litert_lm_logging.cc",
        "litert_lm_logging.h",
    ],
    linkopts = [
        "-Wl,-exported_symbol,_litert_lm_*",
    ],
    linkshared = True,
    linkstatic = True,
    visibility = ["//visibility:public"],
    deps = ENGINE_COMMON_DEPS + [
        "//runtime/core:engine_impl",
    ],
)
BUILD_PATCH
fi

# ---------------------------------------------------------------------------
# 2. Check prerequisites
# ---------------------------------------------------------------------------

if ! command -v bazel &>/dev/null && ! command -v bazelisk &>/dev/null; then
    error "Bazel not found. Install via: brew install bazelisk"
fi

BAZEL_CMD="bazel"
if command -v bazelisk &>/dev/null; then
    BAZEL_CMD="bazelisk"
fi

if ! xcode-select -p &>/dev/null; then
    error "Xcode command line tools not found. Run: xcode-select --install"
fi

info "Using $($BAZEL_CMD --version | head -1)"
info "Using $(xcodebuild -version | head -1)"

# ---------------------------------------------------------------------------
# 3. Build for iOS device (arm64)
# ---------------------------------------------------------------------------

info "Building for iOS device (arm64)..."
cd "$LITERT_LM_DIR"

$BAZEL_CMD build --config=ios_arm64 //c:libLiteRTLMEngine.dylib 2>&1 | tail -5

DEVICE_DYLIB_SRC="$LITERT_LM_DIR/bazel-bin/c/libLiteRTLMEngine.dylib"
if [ ! -f "$DEVICE_DYLIB_SRC" ]; then
    error "Device build failed: $DEVICE_DYLIB_SRC not found"
fi
info "Device build OK: $(du -h "$DEVICE_DYLIB_SRC" | cut -f1)"

# Copy device dylib aside before sim build overwrites bazel-bin
DEVICE_DYLIB="$WORK_DIR/libLiteRTLMEngine-device.dylib"
cp "$DEVICE_DYLIB_SRC" "$DEVICE_DYLIB"

# Also grab the GemmaModelConstraintProvider dylib if present
CONSTRAINT_DYLIB=""
if [ -f "$LITERT_LM_DIR/bazel-bin/c/libGemmaModelConstraintProvider.dylib" ]; then
    CONSTRAINT_DYLIB="$WORK_DIR/libGemmaModelConstraintProvider.dylib"
    cp "$LITERT_LM_DIR/bazel-bin/c/libGemmaModelConstraintProvider.dylib" "$CONSTRAINT_DYLIB"
    info "Found libGemmaModelConstraintProvider.dylib"
fi

# ---------------------------------------------------------------------------
# 3b. Collect prebuilt GPU dylibs from upstream
# ---------------------------------------------------------------------------
# The LiteRT-LM repo ships prebuilt Metal GPU dylibs that are required for
# GPU backend support on iOS. These are dynamically loaded at runtime.

PREBUILT_DEVICE_DIR="$LITERT_LM_DIR/prebuilt/ios_arm64"
PREBUILT_SIM_DIR="$LITERT_LM_DIR/prebuilt/ios_sim_arm64"

# Device GPU dylibs
DEVICE_GPU_DYLIBS=()
for DYLIB_NAME in libLiteRtTopKMetalSampler.dylib libLiteRtMetalAccelerator.dylib libLiteRt.dylib; do
    if [ -f "$PREBUILT_DEVICE_DIR/$DYLIB_NAME" ]; then
        cp "$PREBUILT_DEVICE_DIR/$DYLIB_NAME" "$WORK_DIR/${DYLIB_NAME%.dylib}-device.dylib"
        DEVICE_GPU_DYLIBS+=("$WORK_DIR/${DYLIB_NAME%.dylib}-device.dylib")
        info "Found prebuilt device GPU dylib: $DYLIB_NAME"
    else
        warn "Prebuilt device GPU dylib not found: $DYLIB_NAME (GPU may not work)"
    fi
done

# Also grab prebuilt constraint provider if we didn't get one from the build
if [ -z "$CONSTRAINT_DYLIB" ] && [ -f "$PREBUILT_DEVICE_DIR/libGemmaModelConstraintProvider.dylib" ]; then
    CONSTRAINT_DYLIB="$WORK_DIR/libGemmaModelConstraintProvider.dylib"
    cp "$PREBUILT_DEVICE_DIR/libGemmaModelConstraintProvider.dylib" "$CONSTRAINT_DYLIB"
    info "Using prebuilt libGemmaModelConstraintProvider.dylib"
fi

# Simulator GPU dylibs (no Metal sampler on sim — no real GPU)
SIM_GPU_DYLIBS=()
for DYLIB_NAME in libLiteRtMetalAccelerator.dylib libLiteRt.dylib; do
    if [ -f "$PREBUILT_SIM_DIR/$DYLIB_NAME" ]; then
        cp "$PREBUILT_SIM_DIR/$DYLIB_NAME" "$WORK_DIR/${DYLIB_NAME%.dylib}-sim.dylib"
        SIM_GPU_DYLIBS+=("$WORK_DIR/${DYLIB_NAME%.dylib}-sim.dylib")
        info "Found prebuilt simulator GPU dylib: $DYLIB_NAME"
    else
        warn "Prebuilt simulator GPU dylib not found: $DYLIB_NAME"
    fi
done

SIM_CONSTRAINT_DYLIB=""
if [ -f "$PREBUILT_SIM_DIR/libGemmaModelConstraintProvider.dylib" ]; then
    SIM_CONSTRAINT_DYLIB="$WORK_DIR/libGemmaModelConstraintProvider-sim.dylib"
    cp "$PREBUILT_SIM_DIR/libGemmaModelConstraintProvider.dylib" "$SIM_CONSTRAINT_DYLIB"
    info "Using prebuilt simulator libGemmaModelConstraintProvider.dylib"
fi

# ---------------------------------------------------------------------------
# 4. Build for iOS simulator (arm64)
# ---------------------------------------------------------------------------

info "Building for iOS simulator (arm64)..."

$BAZEL_CMD build --config=ios_sim_arm64 //c:libLiteRTLMEngine.dylib 2>&1 | tail -5

SIM_DYLIB_SRC="$LITERT_LM_DIR/bazel-bin/c/libLiteRTLMEngine.dylib"
if [ ! -f "$SIM_DYLIB_SRC" ]; then
    error "Simulator build failed: $SIM_DYLIB_SRC not found"
fi
info "Simulator build OK: $(du -h "$SIM_DYLIB_SRC" | cut -f1)"

SIM_DYLIB="$WORK_DIR/libLiteRTLMEngine-sim.dylib"
cp "$SIM_DYLIB_SRC" "$SIM_DYLIB"

# ---------------------------------------------------------------------------
# 5. Package as .framework bundles
# ---------------------------------------------------------------------------

HEADERS_DIR="$LITERT_LM_DIR/c"
BUNDLE_ID="com.google.CLiteRTLM"
FRAMEWORK_NAME="CLiteRTLM"
MIN_IOS="13.0"

package_framework() {
    local ARCH_NAME="$1"  # e.g. "ios-arm64"
    local DYLIB_PATH="$2"
    shift 2
    local EXTRA_DYLIBS=("$@")
    local FW_DIR="$WORK_DIR/$ARCH_NAME/$FRAMEWORK_NAME.framework"

    mkdir -p "$FW_DIR/Headers" "$FW_DIR/Modules"

    # Copy binary (rename to framework name)
    cp "$DYLIB_PATH" "$FW_DIR/$FRAMEWORK_NAME"

    # Fix install name
    install_name_tool -id "@rpath/$FRAMEWORK_NAME.framework/$FRAMEWORK_NAME" "$FW_DIR/$FRAMEWORK_NAME"

    # Copy extra dylibs (GPU samplers, accelerators, etc.)
    for EXTRA in "${EXTRA_DYLIBS[@]}"; do
        if [ -n "$EXTRA" ] && [ -f "$EXTRA" ]; then
            # Strip the -device/-sim suffix to recover the original dylib name.
            # Work files are named like: libFoo-device.dylib or libFoo-sim.dylib
            # but some (e.g. constraint provider) have no suffix at all.
            local BASENAME
            BASENAME=$(basename "$EXTRA")
            # Remove -device or -sim tag if present (before .dylib)
            BASENAME=$(echo "$BASENAME" | sed -E 's/-(device|sim)\.dylib$/.dylib/')
            cp "$EXTRA" "$FW_DIR/$BASENAME"
            # Use codesign to clear existing signature first, then set install name.
            # Some prebuilt dylibs lack header padding for longer install names,
            # so we re-link with padding if install_name_tool fails.
            if ! install_name_tool -id "@rpath/$FRAMEWORK_NAME.framework/$BASENAME" "$FW_DIR/$BASENAME" 2>/dev/null; then
                warn "install_name_tool failed for $BASENAME (insufficient header padding), using codesign workaround"
                # Strip the code signature to free header space, then retry
                codesign --remove-signature "$FW_DIR/$BASENAME" 2>/dev/null || true
                install_name_tool -id "@rpath/$FRAMEWORK_NAME.framework/$BASENAME" "$FW_DIR/$BASENAME"
            fi
            codesign --force --sign - "$FW_DIR/$BASENAME"
            info "  Bundled $BASENAME"
        fi
    done

    # Copy headers
    cp "$HEADERS_DIR/engine.h" "$FW_DIR/Headers/"
    cp "$HEADERS_DIR/litert_lm_logging.h" "$FW_DIR/Headers/"

    # Create module map
    cat > "$FW_DIR/Modules/module.modulemap" << 'MODULEMAP'
framework module CLiteRTLM {
    header "engine.h"
    export *
}
MODULEMAP

    # Create Info.plist
    cat > "$FW_DIR/Info.plist" << PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>$FRAMEWORK_NAME</string>
    <key>CFBundleIdentifier</key>
    <string>$BUNDLE_ID</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>$FRAMEWORK_NAME</string>
    <key>CFBundlePackageType</key>
    <string>FMWK</string>
    <key>CFBundleVersion</key>
    <string>1.0</string>
    <key>MinimumOSVersion</key>
    <string>$MIN_IOS</string>
</dict>
</plist>
PLIST

    # Ad-hoc code sign the main binary
    codesign --force --sign - "$FW_DIR/$FRAMEWORK_NAME"

    info "Packaged $ARCH_NAME framework at $FW_DIR"
}

info "Packaging device framework..."
package_framework "ios-arm64" "$DEVICE_DYLIB" "$CONSTRAINT_DYLIB" "${DEVICE_GPU_DYLIBS[@]}"

info "Packaging simulator framework..."
package_framework "ios-arm64-simulator" "$SIM_DYLIB" "$SIM_CONSTRAINT_DYLIB" "${SIM_GPU_DYLIBS[@]}"

# ---------------------------------------------------------------------------
# 6. Create xcframework
# ---------------------------------------------------------------------------

info "Creating xcframework..."

# Remove existing
rm -rf "$OUTPUT_DIR"

xcodebuild -create-xcframework \
    -framework "$WORK_DIR/ios-arm64/$FRAMEWORK_NAME.framework" \
    -framework "$WORK_DIR/ios-arm64-simulator/$FRAMEWORK_NAME.framework" \
    -output "$OUTPUT_DIR"

info "XCFramework created at: $OUTPUT_DIR"

# ---------------------------------------------------------------------------
# 7. Verify
# ---------------------------------------------------------------------------

info "Verifying xcframework..."

for ARCH_DIR in "$OUTPUT_DIR"/ios-*/; do
    BINARY="$ARCH_DIR$FRAMEWORK_NAME.framework/$FRAMEWORK_NAME"
    if [ -f "$BINARY" ]; then
        ARCH_INFO=$(file "$BINARY" | grep -oE 'arm64|x86_64' | head -1)
        SIZE=$(du -h "$BINARY" | cut -f1)
        info "  $(basename "$ARCH_DIR"): $ARCH_INFO ($SIZE)"
    fi
done

TOTAL_SIZE=$(du -sh "$OUTPUT_DIR" | cut -f1)
info "Total xcframework size: $TOTAL_SIZE"

info "Done! xcframework is ready at Frameworks/LiteRTLM.xcframework"
# WORK_DIR is cleaned up automatically by the EXIT trap
