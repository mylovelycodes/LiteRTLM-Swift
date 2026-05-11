//
//  B4LiteRTGPUSymbolProbe.swift
//
//  Path C diagnostic for litert_lm_engine_create returning NULL on gpu_metal.
//  See tasks/litert_gpu_path_c_probe_spec_2026_05_10.md for full provenance.
//
//  Symbol provenance:
//    GPU registry symbols ("LiteRtAcceleratorImpl", "LiteRtRegisterGpuAccelerator"):
//      https://github.com/google-ai-edge/LiteRT/blob/main/litert/runtime/accelerators/gpu_registry.cc
//      lines ~67 and ~77 — passed to RegisterSharedObjectAcceleratorViaAcceleratorDef and
//      RegisterSharedObjectAcceleratorViaFunctionPointer respectively. The actual dlsym
//      lives in registration_helper.cc lines ~119/~127.
//
//    TopK Metal sampler symbols ("LiteRtTopKMetalSampler_*"):
//      https://github.com/google-ai-edge/LiteRT-LM/blob/main/runtime/components/sampler_factory.cc
//      lines ~510-521 — passed verbatim to GetSamplerCApi → SharedLibrary::Load → LookupSymbol
//      (lines 306-369). The factory dlopen's "libLiteRtTopKMetalSampler.dylib" by bare basename
//      with RtldFlags::Lazy().Local() — there is NO RTLD_DEFAULT fallback in this path.
//
//  Run AFTER preloadAcceleratorSidecarsOnce() (which dlopen's all three sidecars at absolute
//  path with RTLD_NOW | RTLD_GLOBAL) and BEFORE/AFTER litert_lm_engine_create() so the probe
//  sees the same symbol-table state the runtime saw at NULL-return time.
//

import Foundation
import Darwin
import os

public enum B4LiteRTGPUSymbolProbe {

    /// Symbols the GPU registry tries to resolve when registering the Metal
    /// accelerator. Either one resolving is sufficient — registry tries
    /// LiteRtAcceleratorImpl first, then falls back to
    /// LiteRtRegisterGpuAccelerator.
    public static let gpuRegistrySymbols: [String] = [
        "LiteRtAcceleratorImpl",
        "LiteRtRegisterGpuAccelerator",
    ]

    /// Symbols the sampler factory tries to resolve from
    /// libLiteRtTopKMetalSampler.dylib when a session uses TopK/TopP sampling.
    /// The first four are REQUIRED — failing any one aborts sampler creation.
    /// The last three are passed by sampler_factory.cc but tolerated as
    /// nullable in current API; we probe them so we can distinguish "Google
    /// hasn't shipped them" from "we haven't exported them".
    public static let metalSamplerSymbols: [String] = [
        // required
        "LiteRtTopKMetalSampler_Create",
        "LiteRtTopKMetalSampler_Destroy",
        "LiteRtTopKMetalSampler_SampleToIdAndScoreBuffer",
        "LiteRtTopKMetalSampler_UpdateConfig",
        // optional but always probed by factory
        "LiteRtTopKMetalSampler_CanHandleInput",
        "LiteRtTopKMetalSampler_HandlesInput",
        "LiteRtTopKMetalSampler_SetInputTensorsAndInferenceFunc",
    ]

    /// Combined symbol list — the union of every symbol any code path inside
    /// litert_lm_engine_create + downstream sampler init will dlsym.
    public static var allSymbols: [String] {
        gpuRegistrySymbols + metalSamplerSymbols
    }

    /// Diagnostic logger pinned to the b4.litert.gpu.symbol-probe subsystem so
    /// `log show --predicate 'subsystem == "b4.litert.gpu.symbol-probe"'`
    /// gives a clean stream on device.
    private static let log = Logger(
        subsystem: "b4.litert.gpu.symbol-probe",
        category: "Probe"
    )

    /// Run the probe. For each symbol in `allSymbols`, calls
    /// `dlsym(RTLD_DEFAULT, sym)` and records whether the lookup returned a
    /// non-NULL pointer. Logs every NULL via os_log at the .error level so
    /// they show up in Console.app filtered by the subsystem above. Returns
    /// the full result map so callers can also push it into DiagnosticLogger
    /// for the diagnostics ZIP that ships with feedback reports.
    @discardableResult
    public static func run() -> [String: Bool] {
        var results: [String: Bool] = [:]
        var missing: [String] = []

        for sym in allSymbols {
            // Darwin defines RTLD_DEFAULT as (void*)-2; Swift's Darwin overlay
            // doesn't expose the macro, so we use the canonical bit-pattern.
            let ptr = dlsym(UnsafeMutableRawPointer(bitPattern: -2), sym)
            let found = (ptr != nil)
            results[sym] = found
            if !found {
                missing.append(sym)
            }
        }

        let foundCount = results.values.filter { $0 }.count
        log.notice("symbol probe complete — \(foundCount, privacy: .public)/\(self.allSymbols.count, privacy: .public) symbols visible via RTLD_DEFAULT")

        if missing.isEmpty {
            log.notice("ALL SYMBOLS RESOLVED — registry-side dlsym is not the failure point; investigate Metal device selection or kernel compilation")
        } else {
            for sym in missing {
                log.error("MISSING symbol via RTLD_DEFAULT: \(sym, privacy: .public)")
            }
            // Categorize so the report is actionable without re-reading the spec.
            let missingRegistry = missing.filter { gpuRegistrySymbols.contains($0) }
            let missingSampler  = missing.filter { metalSamplerSymbols.contains($0) }
            if missingRegistry.count == gpuRegistrySymbols.count {
                log.error("BOTH gpu_registry symbols missing — Metal accelerator registration cannot succeed; either libLiteRtMetalAccelerator preload failed silently or RTLD_DEFAULT cannot see RTLD_GLOBAL-loaded sidecars on this iOS build")
            } else if !missingSampler.isEmpty && missingRegistry.isEmpty {
                log.error("Sampler symbols missing but registry symbols present — sampler_factory.cc will fail; switch to GPU-with-CPU-sampling fallback per Issue #1990 or accept upstream gap")
            }
        }

        return results
    }
}
