// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "LiteRTLMSwift",
    platforms: [
        .iOS(.v17),
        .macOS(.v14)
    ],
    products: [
        .library(name: "LiteRTLMSwift", targets: ["LiteRTLMSwift"])
    ],
    targets: [
        // Top-level engine framework (Swift API + C wrapper).
        .binaryTarget(
            name: "CLiteRTLM",
            path: "Frameworks/LiteRTLM.xcframework"
        ),
        // Gemma model-constraint provider (kept as separate framework since
        // asc.7 — keeps CLiteRTLM.framework lean and ITMS-90171 clean).
        .binaryTarget(
            name: "GemmaConstraints",
            path: "Frameworks/GemmaConstraints.xcframework"
        ),
        // asc.10: Each ML-Drift Metal sidecar is wrapped as its own xcframework
        // so SPM/Xcode treat them as proper "Embed Frameworks" entries (real
        // Mach-O FMWK bundles with Info.plist + CFBundleExecutable +
        // _CodeSignature). Previously we shipped these as raw .dylib payloads
        // inside CLiteRTLM.framework, which tripped Apple's ITMS-90171
        // ("Invalid bundle structure: a dynamic library is included in a
        // bundle that is not a framework") at exportArchive validation. Each
        // framework's binary is patched with LC_ID_DYLIB
        // `@rpath/<Name>.framework/<Name>` so the consumer app embeds them
        // into its `Frameworks/` dir and `dlopen` from the host's
        // privateFrameworksPath resolves cleanly.
        .binaryTarget(
            name: "LiteRtRuntime",
            path: "Frameworks/LiteRtRuntime.xcframework"
        ),
        .binaryTarget(
            name: "LiteRtMetalAccelerator",
            path: "Frameworks/LiteRtMetalAccelerator.xcframework"
        ),
        .binaryTarget(
            name: "LiteRtTopKMetalSampler",
            path: "Frameworks/LiteRtTopKMetalSampler.xcframework"
        ),
        .binaryTarget(
            name: "LiteRtWebGpuAccelerator",
            path: "Frameworks/LiteRtWebGpuAccelerator.xcframework"
        ),
        .binaryTarget(
            name: "LiteRtTopKWebGpuSampler",
            path: "Frameworks/LiteRtTopKWebGpuSampler.xcframework"
        ),
        .target(
            name: "LiteRTLMSwift",
            dependencies: [
                "CLiteRTLM",
                "GemmaConstraints",
                "LiteRtRuntime",
                "LiteRtMetalAccelerator",
                "LiteRtTopKMetalSampler",
                .target(name: "LiteRtWebGpuAccelerator", condition: .when(platforms: [.macOS])),
                .target(name: "LiteRtTopKWebGpuSampler", condition: .when(platforms: [.macOS]))
            ],
            path: "Sources/LiteRTLMSwift"
        ),
    ]
)
