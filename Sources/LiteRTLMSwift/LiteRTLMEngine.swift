import Foundation
import CoreGraphics
import ImageIO
import os
import CLiteRTLM

/// Swift wrapper for Google's LiteRT-LM on-device inference engine.
///
/// Supports text generation (Session API) and multimodal inference — vision
/// and audio — (Conversation API) with `.litertlm` model files (e.g. Gemma 4 E2B).
///
/// Thread safety: all C API calls are serialized on an internal dispatch queue.
/// The class is `@unchecked Sendable` because OpaquePointers are only accessed
/// on that queue.
///
/// ## Quick Start
/// ```swift
/// let engine = LiteRTLMEngine(modelPath: modelURL)
/// try await engine.load()
///
/// // Text
/// let response = try await engine.generate(prompt: "Hello!", temperature: 0.7, maxTokens: 256)
///
/// // Vision
/// let caption = try await engine.vision(imageData: jpegData, prompt: "Describe this photo.")
///
/// // Audio
/// let transcript = try await engine.audio(audioData: wavData, prompt: "Transcribe this audio.")
/// ```
@Observable
public final class LiteRTLMEngine: @unchecked Sendable {

    // MARK: - Types

    public enum Status: Sendable, Equatable {
        case notLoaded
        case loading
        case ready
        case error(String)
    }

    // MARK: - Properties

    public private(set) var status: Status = .notLoaded

    /// Whether the engine is ready for inference (text, vision, and audio).
    public var isReady: Bool { status == .ready }

    private let modelPath: URL
    private let backend: String

    private var engine: OpaquePointer?  // LiteRtLmEngine*
    private let inferenceQueue = DispatchQueue(label: "com.litertlm.inference", qos: .userInitiated)

    /// Background queue used to invoke long-running, blocking C streaming calls.
    /// Concurrent + userInitiated so multiple streams can be in flight on different
    /// engines, while still being serialized against `inferenceQueue` for handle access.
    private static let streamingExecQueue = DispatchQueue(
        label: "com.litertlm.streaming.exec", qos: .userInitiated, attributes: .concurrent
    )

    /// Tracks the currently active conversation pointer (if any) so `cancel()` can
    /// dispatch `litert_lm_conversation_cancel_process` from any thread without
    /// blocking on `inferenceQueue`. Mutated only under `inferenceQueue` sync.
    /// Reads from `cancel()` use an os_unfair_lock for atomicity.
    private var activeConversation: OpaquePointer?

    /// Set true when `cancel()` is called; checked by streaming callbacks and
    /// inference loops to bail out early. Reset at the start of each new
    /// inference call.
    private var cancellationRequested: Bool = false

    /// Lock guarding `activeConversation` and `cancellationRequested` so that
    /// `cancel()` (callable from any thread) can safely read/write them without
    /// hopping onto `inferenceQueue` (which may itself be blocked running a
    /// streaming generate call).
    private let cancelStateLock = NSLock()

    private static let log = Logger(subsystem: "LiteRTLMSwift", category: "Engine")

    // MARK: - Init

    /// Create an engine instance.
    /// - Parameters:
    ///   - modelPath: Path to the `.litertlm` model file on disk.
    ///   - backend: Compute backend — `"cpu"` or `"gpu"` (GPU uses Metal on iOS).
    public init(modelPath: URL, backend: String = "cpu") {
        self.modelPath = modelPath
        self.backend = backend
    }

    /// Releases all C handles synchronously.
    ///
    /// **Lifecycle contract:** consumers SHOULD call `unload()` before releasing
    /// the last strong reference. If the engine is still loaded at deinit, this
    /// runs a `inferenceQueue.sync` cleanup so it serializes against any
    /// in-flight inference. This is safe because `inferenceQueue` blocks
    /// retain `self` via `[self]`, so deinit cannot fire while a queue block is
    /// executing — there is no path on which deinit would re-enter
    /// `inferenceQueue` and deadlock.
    ///
    /// In DEBUG builds we log a warning if `unload()` was not called explicitly,
    /// because that's almost always a leak of the multi-GB model weights up to
    /// the next ARC cycle.
    deinit {
        let eng = engine
        let ses = chatSession
        let sesCfg = chatSessionConfig
        let conv = multimodalConversation
        let convCfg = multimodalConvConfig
        let convSesCfg = multimodalSessionConfig
        guard eng != nil || ses != nil || conv != nil else { return }

        #if DEBUG
        Self.log.warning("LiteRTLMEngine deinit while still loaded — call unload() first to release model memory deterministically.")
        #endif

        // Serialize against any in-flight inference. This is safe: queue blocks
        // capture `[self]`, so deinit cannot run while a block is executing.
        inferenceQueue.sync {
            if let s = ses { litert_lm_session_delete(s) }
            if let c = sesCfg { litert_lm_session_config_delete(c) }
            if let c = conv { litert_lm_conversation_delete(c) }
            if let c = convCfg { litert_lm_conversation_config_delete(c) }
            if let c = convSesCfg { litert_lm_session_config_delete(c) }
            if let e = eng { litert_lm_engine_delete(e) }
        }
    }

    // MARK: - Lifecycle

    /// Load the `.litertlm` model. Call once, reuse for multiple inferences.
    /// Vision and audio encoders are embedded in the model file — no separate load step needed.
    @MainActor
    public func load() async throws {
        guard status != .ready && status != .loading else { return }

        status = .loading
        Self.log.info("Loading model: \(self.modelPath.lastPathComponent), backend: \(self.backend)")

        let path = modelPath.path
        let backendStr = self.backend
        let startTime = CFAbsoluteTimeGetCurrent()

        guard FileManager.default.fileExists(atPath: path) else {
            let msg = "Model file not found at \(path)"
            Self.log.error("\(msg)")
            status = .error(msg)
            throw LiteRTLMError.modelNotFound
        }

        do {
            let createdEngine = try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<OpaquePointer, any Error>) in
                self.inferenceQueue.async {
                    do {
                        litert_lm_set_min_log_level(1)

                        guard let settings = litert_lm_engine_settings_create(
                            path, backendStr, backendStr, backendStr
                        ) else {
                            throw LiteRTLMError.engineCreationFailed("Failed to create engine settings")
                        }

                        litert_lm_engine_settings_set_max_num_tokens(settings, 4096)

                        let cacheBase: URL = FileManager.default
                            .urls(for: .cachesDirectory, in: .userDomainMask)
                            .first
                            ?? URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
                        let cacheDir = cacheBase.appendingPathComponent("litertlm_cache").path
                        try? FileManager.default.createDirectory(atPath: cacheDir, withIntermediateDirectories: true)
                        litert_lm_engine_settings_set_cache_dir(settings, cacheDir)

                        litert_lm_engine_settings_enable_benchmark(settings)

                        guard let createdEngine = litert_lm_engine_create(settings) else {
                            litert_lm_engine_settings_delete(settings)
                            throw LiteRTLMError.engineCreationFailed("litert_lm_engine_create returned NULL")
                        }
                        litert_lm_engine_settings_delete(settings)

                        continuation.resume(returning: createdEngine)
                    } catch {
                        continuation.resume(throwing: error)
                    }
                }
            }

            inferenceQueue.sync { self.engine = createdEngine }

            let elapsed = CFAbsoluteTimeGetCurrent() - startTime
            Self.log.info("Model loaded in \(String(format: "%.1f", elapsed))s")
            status = .ready
        } catch {
            let msg = "Load failed: \(error.localizedDescription)"
            Self.log.error("\(msg)")
            status = .error(msg)
            throw error
        }
    }

    /// Unload the model to free memory.
    @MainActor
    public func unload() {
        inferenceQueue.sync {
            if let s = chatSession {
                litert_lm_session_delete(s)
                chatSession = nil
            }
            if let c = chatSessionConfig {
                litert_lm_session_config_delete(c)
                chatSessionConfig = nil
            }
            if let c = multimodalConversation {
                litert_lm_conversation_delete(c)
                multimodalConversation = nil
            }
            if let c = multimodalConvConfig {
                litert_lm_conversation_config_delete(c)
                multimodalConvConfig = nil
            }
            if let c = multimodalSessionConfig {
                litert_lm_session_config_delete(c)
                multimodalSessionConfig = nil
            }
            if let eng = engine { litert_lm_engine_delete(eng) }
            engine = nil
        }
        status = .notLoaded
        Self.log.info("Model unloaded")
    }

    // MARK: - Cancellation

    /// Cancel the in-flight inference, if any.
    ///
    /// Cancellation contract:
    /// - Safe to call from any thread, including from inside an `await` site
    ///   on the same task that started the inference.
    /// - For multimodal/Conversation inferences, this calls
    ///   `litert_lm_conversation_cancel_process` which signals the C runtime to
    ///   stop generation early. The pending `await` will then unwind via the
    ///   normal `isFinal` callback path.
    /// - For text-only Session streaming, the C runtime does not expose a
    ///   cancel symbol. We mark the in-flight stream cancelled so the
    ///   `AsyncThrowingStream` finishes immediately on the consumer side; the
    ///   underlying C generate loop continues to completion in the background
    ///   but its tokens are dropped. Tokens already buffered before cancel
    ///   may still be delivered.
    /// - Calling `cancel()` when no inference is active is a no-op.
    /// - This method does NOT throw; it always succeeds.
    public func cancel() {
        cancelStateLock.lock()
        cancellationRequested = true
        let conv = activeConversation
        cancelStateLock.unlock()

        if let conv {
            // The C cancel call is itself thread-safe per the engine.h contract.
            litert_lm_conversation_cancel_process(conv)
            Self.log.info("Cancellation requested (conversation)")
        } else {
            Self.log.info("Cancellation requested (no active conversation; session stream will be dropped on consumer side)")
        }
    }

    /// Snapshot the cancellation flag in a thread-safe way.
    fileprivate func isCancelled() -> Bool {
        cancelStateLock.lock()
        defer { cancelStateLock.unlock() }
        return cancellationRequested
    }

    /// Reset cancellation state at the start of a new inference.
    fileprivate func resetCancellationState() {
        cancelStateLock.lock()
        cancellationRequested = false
        cancelStateLock.unlock()
    }

    /// Register the currently active conversation pointer so `cancel()` can
    /// reach it from any thread. Pass `nil` to unregister.
    fileprivate func setActiveConversation(_ conv: OpaquePointer?) {
        cancelStateLock.lock()
        activeConversation = conv
        cancelStateLock.unlock()
    }

    // MARK: - Text Generation (Session API)

    /// Generate text from a prompt. Creates a one-shot session per call.
    ///
    /// - Parameters:
    ///   - prompt: The input text. For Gemma 4, use `<|turn>user\n...<turn|>\n<|turn>model\n` format.
    ///   - temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative). Default 0.7.
    ///   - maxTokens: Maximum tokens to generate. Default 512.
    /// - Returns: Generated text.
    public func generate(
        prompt: String,
        temperature: Float = 0.7,
        maxTokens: Int = 512
    ) async throws -> String {
        try ensureReady()
        return try await runSessionInference(
            prompt: prompt, temperature: temperature, maxTokens: Int32(maxTokens)
        )
    }

    /// Stream text generation token by token.
    ///
    /// Creates a one-shot session per call. For multi-turn conversations with
    /// KV cache reuse, use the persistent session API instead.
    ///
    /// - Parameters:
    ///   - prompt: The input text.
    ///   - temperature: Sampling temperature. Default 0.7.
    ///   - maxTokens: Maximum tokens to generate. Default 512.
    /// - Returns: An `AsyncThrowingStream` yielding text chunks.
    public func generateStreaming(
        prompt: String,
        temperature: Float = 0.7,
        maxTokens: Int = 512
    ) -> AsyncThrowingStream<String, Error> {
        runSessionInferenceStreaming(
            prompt: prompt, temperature: temperature, maxTokens: Int32(maxTokens)
        )
    }

    // MARK: - Vision (Conversation API)

    /// Run vision inference on a single image.
    ///
    /// Uses the Conversation API, which handles image decoding, resizing, and
    /// patchification internally. Input images are auto-converted to JPEG and
    /// resized to fit within `maxImageDimension`.
    ///
    /// - Parameters:
    ///   - imageData: Raw image bytes (JPEG, PNG, HEIC, etc.).
    ///   - prompt: Text prompt for the vision model (e.g., "Describe this photo.").
    ///   - temperature: Sampling temperature. Default 0.7.
    ///   - maxTokens: Maximum tokens to generate. Default 512.
    ///   - maxImageDimension: Resize long edge to this value. Default 1024.
    /// - Returns: Generated text response.
    public func vision(
        imageData: Data,
        prompt: String,
        temperature: Float = 0.7,
        maxTokens: Int = 512,
        maxImageDimension: Int = 1024
    ) async throws -> String {
        try ensureReady()

        guard let jpegData = Self.prepareImageForVision(imageData, maxDimension: maxImageDimension) else {
            throw LiteRTLMError.inferenceFailure("Failed to convert image to JPEG")
        }

        let tempURL = Self.makeTempURL(extension: "jpg")
        try jpegData.write(to: tempURL)

        let messageJSON = Self.buildMultimodalMessageJSON(
            audioPaths: [], imagePaths: [tempURL.path], text: prompt
        )
        return try await runConversationInference(
            messageJSON: messageJSON,
            tempURLs: [tempURL],
            temperature: temperature,
            maxTokens: maxTokens
        )
    }

    /// Run vision inference on multiple images.
    ///
    /// - Parameters:
    ///   - imagesData: Array of raw image bytes.
    ///   - prompt: Text prompt about the images.
    ///   - temperature: Sampling temperature. Default 0.7.
    ///   - maxTokens: Maximum tokens to generate. Default 1024.
    ///   - maxImageDimension: Resize long edge to this value. Default 1024.
    /// - Returns: Generated text response.
    public func visionMultiImage(
        imagesData: [Data],
        prompt: String,
        temperature: Float = 0.7,
        maxTokens: Int = 1024,
        maxImageDimension: Int = 1024
    ) async throws -> String {
        try ensureReady()
        guard !imagesData.isEmpty else {
            throw LiteRTLMError.inferenceFailure("No images provided")
        }

        var tempURLs: [URL] = []
        do {
            for (i, data) in imagesData.enumerated() {
                guard let jpegData = Self.prepareImageForVision(data, maxDimension: maxImageDimension) else {
                    throw LiteRTLMError.inferenceFailure("Failed to convert image \(i + 1) to JPEG")
                }
                let url = Self.makeTempURL(extension: "jpg")
                try jpegData.write(to: url)
                tempURLs.append(url)
            }
        } catch {
            Self.cleanupTempFiles(tempURLs)
            throw error
        }

        let messageJSON = Self.buildMultimodalMessageJSON(
            audioPaths: [], imagePaths: tempURLs.map(\.path), text: prompt
        )
        return try await runConversationInference(
            messageJSON: messageJSON,
            tempURLs: tempURLs,
            temperature: temperature,
            maxTokens: maxTokens
        )
    }

    // MARK: - Audio (Conversation API)

    /// Supported audio formats for the `audio()` and `multimodal()` methods.
    public enum AudioFormat: String, Sendable {
        case wav, flac, mp3
    }

    /// Run audio inference on a single audio file.
    ///
    /// Uses the Conversation API, which handles audio decoding and preprocessing
    /// (resample to 16 kHz, convert to mel spectrogram) internally.
    ///
    /// - Parameters:
    ///   - audioData: Raw audio bytes (WAV, FLAC, or MP3).
    ///   - prompt: Text prompt (e.g., "Transcribe this audio.", "Summarize what is being said.").
    ///   - format: Audio container format. Default `.wav`.
    ///   - temperature: Sampling temperature. Default 0.7.
    ///   - maxTokens: Maximum tokens to generate. Default 512.
    /// - Returns: Generated text response.
    public func audio(
        audioData: Data,
        prompt: String,
        format: AudioFormat = .wav,
        temperature: Float = 0.7,
        maxTokens: Int = 512
    ) async throws -> String {
        try ensureReady()
        guard !audioData.isEmpty else {
            throw LiteRTLMError.inferenceFailure("No audio data provided")
        }

        let tempURL = Self.makeTempURL(extension: format.rawValue)
        try audioData.write(to: tempURL)

        let messageJSON = Self.buildMultimodalMessageJSON(
            audioPaths: [tempURL.path], imagePaths: [], text: prompt
        )
        return try await runConversationInference(
            messageJSON: messageJSON,
            tempURLs: [tempURL],
            temperature: temperature,
            maxTokens: maxTokens
        )
    }

    /// Run multimodal inference combining audio, images, and text in a single query.
    ///
    /// Useful for tasks like "describe what's happening in this video" where you have
    /// both the audio track and keyframes, or "does this photo match what the speaker describes?".
    ///
    /// - Parameters:
    ///   - audioData: Array of raw audio bytes (WAV, FLAC, or MP3). Pass empty array to skip.
    ///   - imagesData: Array of raw image bytes (JPEG, PNG, HEIC). Pass empty array to skip.
    ///   - prompt: Text prompt about the audio and/or images.
    ///   - temperature: Sampling temperature. Default 0.7.
    ///   - maxTokens: Maximum tokens to generate. Default 1024.
    ///   - maxImageDimension: Resize image long edge to this value. Default 1024.
    /// - Returns: Generated text response.
    public func multimodal(
        audioData: [Data] = [],
        audioFormat: AudioFormat = .wav,
        imagesData: [Data] = [],
        prompt: String,
        temperature: Float = 0.7,
        maxTokens: Int = 1024,
        maxImageDimension: Int = 1024
    ) async throws -> String {
        try ensureReady()
        guard !audioData.isEmpty || !imagesData.isEmpty else {
            throw LiteRTLMError.inferenceFailure("No audio or image data provided")
        }

        var tempURLs: [URL] = []
        var audioPaths: [String] = []
        var imagePaths: [String] = []

        do {
            // Write audio files
            for (i, data) in audioData.enumerated() {
                guard !data.isEmpty else {
                    throw LiteRTLMError.inferenceFailure("Audio data \(i + 1) is empty")
                }
                let url = Self.makeTempURL(extension: audioFormat.rawValue)
                try data.write(to: url)
                tempURLs.append(url)
                audioPaths.append(url.path)
            }

            // Write image files
            for (i, data) in imagesData.enumerated() {
                guard let jpegData = Self.prepareImageForVision(data, maxDimension: maxImageDimension) else {
                    throw LiteRTLMError.inferenceFailure("Failed to convert image \(i + 1) to JPEG")
                }
                let url = Self.makeTempURL(extension: "jpg")
                try jpegData.write(to: url)
                tempURLs.append(url)
                imagePaths.append(url.path)
            }
        } catch {
            Self.cleanupTempFiles(tempURLs)
            throw error
        }

        let messageJSON = Self.buildMultimodalMessageJSON(
            audioPaths: audioPaths, imagePaths: imagePaths, text: prompt
        )
        return try await runConversationInference(
            messageJSON: messageJSON,
            tempURLs: tempURLs,
            temperature: temperature,
            maxTokens: maxTokens
        )
    }

    // MARK: - Persistent Session (KV Cache Reuse)
    //
    // LiteRT-LM's Session maintains a KV cache across multiple generate_content
    // calls. By keeping the session alive across turns, subsequent messages only
    // need to prefill NEW tokens instead of the entire conversation history.
    // This reduces TTFT from ~20s (full prefill) to ~1-2s (incremental).

    private var chatSession: OpaquePointer?
    private var chatSessionConfig: OpaquePointer?

    /// Open a persistent session for multi-turn generation with KV cache reuse.
    ///
    /// Call once when a conversation begins. Subsequent calls to
    /// `sessionGenerateStreaming(input:)` reuse this session's KV cache.
    ///
    /// - Parameters:
    ///   - temperature: Sampling temperature. Default 0.3.
    ///   - maxTokens: Maximum tokens per generation. Default 512.
    public func openSession(temperature: Float = 0.3, maxTokens: Int = 512) async throws {
        try ensureReady()
        try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
            inferenceQueue.async { [self] in
                do {
                    if let s = chatSession {
                        litert_lm_session_delete(s)
                        chatSession = nil
                    }
                    if let c = chatSessionConfig {
                        litert_lm_session_config_delete(c)
                        chatSessionConfig = nil
                    }

                    guard let eng = engine else { throw LiteRTLMError.modelNotLoaded }
                    let (session, config) = try createSession(
                        engine: eng, temperature: temperature, maxTokens: Int32(maxTokens)
                    )
                    chatSession = session
                    chatSessionConfig = config
                    Self.log.info("Persistent session opened")
                    cont.resume()
                } catch {
                    cont.resume(throwing: error)
                }
            }
        }
    }

    /// Close the persistent session, freeing KV cache memory.
    public func closeSession() {
        inferenceQueue.async { [self] in
            guard chatSession != nil else { return }
            if let s = chatSession {
                logSessionBenchmark(s)
                litert_lm_session_delete(s)
                chatSession = nil
            }
            if let c = chatSessionConfig {
                litert_lm_session_config_delete(c)
                chatSessionConfig = nil
            }
            Self.log.info("Persistent session closed")
        }
    }

    // MARK: - Persistent Conversation (Multimodal KV Cache Reuse)
    //
    // Like the text-only persistent session above, but uses the Conversation
    // API — supporting images, audio, and text. The conversation's KV cache
    // persists across turns, so follow-up messages only prefill new tokens.

    private var multimodalConversation: OpaquePointer?
    private var multimodalConvConfig: OpaquePointer?
    private var multimodalSessionConfig: OpaquePointer?

    /// Open a persistent multimodal conversation with KV cache reuse.
    ///
    /// Call once when a conversation begins. Subsequent calls to
    /// `conversationSend(...)` reuse this conversation's KV cache,
    /// reducing TTFT from ~20s to ~1-2s for follow-up turns.
    ///
    /// - Parameters:
    ///   - temperature: Sampling temperature. Default 0.7.
    ///   - maxTokens: Maximum tokens per generation. Default 1024.
    public func openConversation(temperature: Float = 0.7, maxTokens: Int = 1024) async throws {
        try ensureReady()
        try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
            inferenceQueue.async { [self] in
                do {
                    // Close existing conversation if any
                    if let c = multimodalConversation {
                        litert_lm_conversation_delete(c)
                        multimodalConversation = nil
                    }
                    if let c = multimodalConvConfig {
                        litert_lm_conversation_config_delete(c)
                        multimodalConvConfig = nil
                    }
                    if let c = multimodalSessionConfig {
                        litert_lm_session_config_delete(c)
                        multimodalSessionConfig = nil
                    }

                    guard let eng = engine else { throw LiteRTLMError.modelNotLoaded }

                    guard let sessionConfig = litert_lm_session_config_create() else {
                        throw LiteRTLMError.inferenceFailure("Failed to create session config")
                    }
                    litert_lm_session_config_set_max_output_tokens(sessionConfig, Int32(maxTokens))
                    var samplerParams = LiteRtLmSamplerParams(
                        type: kTopP, top_k: 40, top_p: 0.95,
                        temperature: temperature, seed: 0
                    )
                    litert_lm_session_config_set_sampler_params(sessionConfig, &samplerParams)

                    guard let convConfig = litert_lm_conversation_config_create(
                        eng, sessionConfig, nil, nil, nil, false
                    ) else {
                        litert_lm_session_config_delete(sessionConfig)
                        throw LiteRTLMError.inferenceFailure("Failed to create conversation config")
                    }

                    guard let conversation = litert_lm_conversation_create(eng, convConfig) else {
                        litert_lm_conversation_config_delete(convConfig)
                        litert_lm_session_config_delete(sessionConfig)
                        throw LiteRTLMError.inferenceFailure("Failed to create conversation")
                    }

                    multimodalConversation = conversation
                    multimodalConvConfig = convConfig
                    multimodalSessionConfig = sessionConfig
                    Self.log.info("Persistent multimodal conversation opened")
                    cont.resume()
                } catch {
                    cont.resume(throwing: error)
                }
            }
        }
    }

    /// Send a message in the persistent multimodal conversation.
    ///
    /// Each call reuses the conversation's KV cache. Pass any combination of
    /// audio, images, and text — or just text for a follow-up question.
    ///
    /// - Parameters:
    ///   - audioData: Array of raw audio bytes. Pass empty array (default) for non-audio turns.
    ///   - audioFormat: Audio container format. Default `.wav`.
    ///   - imagesData: Array of raw image bytes. Pass empty array (default) for non-image turns.
    ///   - prompt: Text prompt for this turn.
    ///   - maxImageDimension: Resize image long edge to this value. Default 1024.
    /// - Returns: Generated text response.
    public func conversationSend(
        audioData: [Data] = [],
        audioFormat: AudioFormat = .wav,
        imagesData: [Data] = [],
        prompt: String,
        maxImageDimension: Int = 1024
    ) async throws -> String {
        try ensureReady()

        // Prepare media files
        var tempURLs: [URL] = []
        var audioPaths: [String] = []
        var imagePaths: [String] = []

        do {
            for (i, data) in audioData.enumerated() {
                guard !data.isEmpty else {
                    throw LiteRTLMError.inferenceFailure("Audio data \(i + 1) is empty")
                }
                let url = Self.makeTempURL(extension: audioFormat.rawValue)
                try data.write(to: url)
                tempURLs.append(url)
                audioPaths.append(url.path)
            }
            for (i, data) in imagesData.enumerated() {
                guard let jpegData = Self.prepareImageForVision(data, maxDimension: maxImageDimension) else {
                    throw LiteRTLMError.inferenceFailure("Failed to convert image \(i + 1) to JPEG")
                }
                let url = Self.makeTempURL(extension: "jpg")
                try jpegData.write(to: url)
                tempURLs.append(url)
                imagePaths.append(url.path)
            }
        } catch {
            Self.cleanupTempFiles(tempURLs)
            throw error
        }

        let messageJSON = Self.buildMultimodalMessageJSON(
            audioPaths: audioPaths, imagePaths: imagePaths, text: prompt
        )

        let urlsToCleanup = tempURLs
        return try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<String, any Error>) in
            self.inferenceQueue.async { [self, urlsToCleanup] in
                defer { Self.cleanupTempFiles(urlsToCleanup) }
                do {
                    guard let conversation = self.multimodalConversation else {
                        throw LiteRTLMError.inferenceFailure(
                            "No persistent conversation open — call openConversation() first"
                        )
                    }
                    self.resetCancellationState()
                    self.setActiveConversation(conversation)
                    defer { self.setActiveConversation(nil) }

                    guard let response = messageJSON.withCString({ msgPtr in
                        litert_lm_conversation_send_message(conversation, msgPtr, nil)
                    }) else {
                        if self.isCancelled() {
                            throw LiteRTLMError.cancelled
                        }
                        throw LiteRTLMError.inferenceFailure("Conversation returned no response")
                    }
                    defer { litert_lm_json_response_delete(response) }

                    guard let responsePtr = litert_lm_json_response_get_string(response) else {
                        throw LiteRTLMError.inferenceFailure("Response string is NULL")
                    }

                    let result = Self.extractTextFromConversationResponse(String(cString: responsePtr))
                    continuation.resume(returning: result)
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    /// Close the persistent multimodal conversation, freeing KV cache memory.
    public func closeConversation() {
        inferenceQueue.async { [self] in
            guard multimodalConversation != nil else { return }
            if let c = multimodalConversation {
                litert_lm_conversation_delete(c)
                multimodalConversation = nil
            }
            if let c = multimodalConvConfig {
                litert_lm_conversation_config_delete(c)
                multimodalConvConfig = nil
            }
            if let c = multimodalSessionConfig {
                litert_lm_session_config_delete(c)
                multimodalSessionConfig = nil
            }
            Self.log.info("Persistent multimodal conversation closed")
        }
    }

    /// Stream text using the persistent session.
    ///
    /// `input` should be ONLY the new turn content — the session's KV cache
    /// already holds all previous context.
    ///
    /// - Parameter input: New input text for this turn.
    /// - Returns: An `AsyncThrowingStream` yielding text chunks.
    public func sessionGenerateStreaming(input: String) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            // Snapshot the session pointer under the inference queue (handle guard),
            // then run the blocking C streaming call on a dedicated background queue
            // so we do NOT hold `inferenceQueue` while waiting on the semaphore. This
            // avoids deadlocking if the C runtime ever dispatches its callback onto
            // the same queue.
            let session: OpaquePointer? = self.inferenceQueue.sync { self.chatSession }
            guard let session else {
                continuation.finish(throwing: LiteRTLMError.inferenceFailure("No persistent session open — call openSession() first"))
                return
            }

            self.resetCancellationState()

            // Wire up Task cancellation: when the consumer cancels the parent Task
            // (or the AsyncThrowingStream is dropped), forward to engine.cancel().
            continuation.onTermination = { [weak self] _ in
                self?.cancel()
            }

            Self.streamingExecQueue.async { [self] in
                let streamDone = DispatchSemaphore(value: 0)
                let state = StreamCallbackState(
                    continuation: continuation,
                    doneSemaphore: streamDone,
                    isCancelled: { [weak self] in self?.isCancelled() ?? false }
                )
                let statePtr = Unmanaged.passRetained(state).toOpaque()

                let result = input.withCString { textPtr -> Int32 in
                    var inputData = InputData(
                        type: kInputText,
                        data: UnsafeRawPointer(textPtr),
                        size: strlen(textPtr)
                    )
                    return litert_lm_session_generate_content_stream(
                        session, &inputData, 1,
                        Self.streamCallback,
                        statePtr
                    )
                }

                if result != 0 {
                    Unmanaged<StreamCallbackState>.fromOpaque(statePtr).release()
                    continuation.finish(throwing: LiteRTLMError.inferenceFailure("Failed to start stream"))
                    return
                }

                streamDone.wait()
                // Re-enter inferenceQueue ONLY for handle-touching post-work
                // (benchmark read). This keeps the streaming queue free.
                self.inferenceQueue.sync { self.logSessionBenchmark(session) }
            }
        }
    }

    // MARK: - Private: Session-based Inference

    private func runSessionInference(
        prompt: String,
        temperature: Float,
        maxTokens: Int32
    ) async throws -> String {
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<String, any Error>) in
            self.inferenceQueue.async { [self] in
                do {
                    guard let eng = self.engine else { throw LiteRTLMError.modelNotLoaded }

                    let (session, sessionConfig) = try self.createSession(
                        engine: eng, temperature: temperature, maxTokens: maxTokens
                    )
                    defer {
                        litert_lm_session_delete(session)
                        litert_lm_session_config_delete(sessionConfig)
                    }

                    let output = prompt.withCString { textPtr -> String? in
                        var input = InputData(
                            type: kInputText,
                            data: UnsafeRawPointer(textPtr),
                            size: strlen(textPtr)
                        )
                        guard let responses = litert_lm_session_generate_content(session, &input, 1) else {
                            return nil
                        }
                        defer { litert_lm_responses_delete(responses) }
                        return self.extractResponseText(responses)
                    }

                    guard let result = output else {
                        throw LiteRTLMError.inferenceFailure("generate_content returned no output")
                    }

                    self.logSessionBenchmark(session)
                    continuation.resume(returning: result)
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    private func runSessionInferenceStreaming(
        prompt: String,
        temperature: Float,
        maxTokens: Int32
    ) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            // Phase 1: handle-guarded session creation on inferenceQueue.
            let creation: Result<(OpaquePointer, OpaquePointer), Error> = self.inferenceQueue.sync {
                do {
                    try self.ensureReady()
                    guard let eng = self.engine else {
                        return .failure(LiteRTLMError.modelNotLoaded)
                    }
                    let pair = try self.createSession(
                        engine: eng, temperature: temperature, maxTokens: maxTokens
                    )
                    return .success(pair)
                } catch {
                    return .failure(error)
                }
            }

            let session: OpaquePointer
            let sessionConfig: OpaquePointer
            switch creation {
            case .failure(let err):
                continuation.finish(throwing: err)
                return
            case .success(let pair):
                session = pair.0
                sessionConfig = pair.1
            }

            self.resetCancellationState()

            continuation.onTermination = { [weak self] _ in
                self?.cancel()
            }

            // Phase 2: blocking generate on a dedicated background queue. Do NOT
            // hold `inferenceQueue` here — the C streaming callback may post on
            // the same queue, which would deadlock.
            Self.streamingExecQueue.async { [self] in
                let streamDone = DispatchSemaphore(value: 0)
                let state = StreamCallbackState(
                    continuation: continuation,
                    doneSemaphore: streamDone,
                    isCancelled: { [weak self] in self?.isCancelled() ?? false }
                )
                let statePtr = Unmanaged.passRetained(state).toOpaque()

                let result = prompt.withCString { textPtr -> Int32 in
                    var input = InputData(
                        type: kInputText,
                        data: UnsafeRawPointer(textPtr),
                        size: strlen(textPtr)
                    )
                    return litert_lm_session_generate_content_stream(
                        session, &input, 1,
                        Self.streamCallback,
                        statePtr
                    )
                }

                if result != 0 {
                    Unmanaged<StreamCallbackState>.fromOpaque(statePtr).release()
                    self.inferenceQueue.sync {
                        litert_lm_session_delete(session)
                        litert_lm_session_config_delete(sessionConfig)
                    }
                    continuation.finish(throwing: LiteRTLMError.inferenceFailure("Failed to start stream"))
                    return
                }

                streamDone.wait()
                // Phase 3: handle teardown back on inferenceQueue.
                self.inferenceQueue.sync {
                    self.logSessionBenchmark(session)
                    litert_lm_session_delete(session)
                    litert_lm_session_config_delete(sessionConfig)
                }
            }
        }
    }

    // MARK: - Private Helpers

    private func ensureReady() throws {
        guard status == .ready else { throw LiteRTLMError.modelNotLoaded }
    }

    private func createSession(
        engine eng: OpaquePointer,
        temperature: Float,
        maxTokens: Int32
    ) throws -> (session: OpaquePointer, config: OpaquePointer) {
        guard let sessionConfig = litert_lm_session_config_create() else {
            throw LiteRTLMError.inferenceFailure("Failed to create session config")
        }

        litert_lm_session_config_set_max_output_tokens(sessionConfig, maxTokens)
        var samplerParams = LiteRtLmSamplerParams(
            type: kTopP, top_k: 40, top_p: 0.95,
            temperature: temperature, seed: 0
        )
        litert_lm_session_config_set_sampler_params(sessionConfig, &samplerParams)

        guard let session = litert_lm_engine_create_session(eng, sessionConfig) else {
            litert_lm_session_config_delete(sessionConfig)
            throw LiteRTLMError.inferenceFailure("Failed to create session")
        }

        return (session, sessionConfig)
    }

    private func extractResponseText(_ responses: OpaquePointer) -> String? {
        let numCandidates = litert_lm_responses_get_num_candidates(responses)
        guard numCandidates > 0,
              let resultPtr = litert_lm_responses_get_response_text_at(responses, 0) else {
            return nil
        }
        return String(cString: resultPtr)
    }

    private func logSessionBenchmark(_ session: OpaquePointer) {
        guard let info = litert_lm_session_get_benchmark_info(session) else { return }
        defer { litert_lm_benchmark_info_delete(info) }

        let initTime = litert_lm_benchmark_info_get_total_init_time_in_second(info)
        let ttft = litert_lm_benchmark_info_get_time_to_first_token(info)
        let numDecode = litert_lm_benchmark_info_get_num_decode_turns(info)
        let numPrefill = litert_lm_benchmark_info_get_num_prefill_turns(info)

        Self.log.info("Benchmark: init=\(String(format: "%.2f", initTime))s, TTFT=\(String(format: "%.2f", ttft))s")

        for i in 0..<numPrefill {
            let tps = litert_lm_benchmark_info_get_prefill_tokens_per_sec_at(info, Int32(i))
            let count = litert_lm_benchmark_info_get_prefill_token_count_at(info, Int32(i))
            Self.log.info("  Prefill[\(i)]: \(count) tokens @ \(String(format: "%.1f", tps)) tok/s")
        }
        for i in 0..<numDecode {
            let tps = litert_lm_benchmark_info_get_decode_tokens_per_sec_at(info, Int32(i))
            let count = litert_lm_benchmark_info_get_decode_token_count_at(info, Int32(i))
            Self.log.info("  Decode[\(i)]: \(count) tokens @ \(String(format: "%.1f", tps)) tok/s")
        }
    }

    // MARK: - Private: Conversation-based Inference (Vision / Audio / Multimodal)

    /// Shared helper for all Conversation API calls (vision, audio, multimodal).
    /// Handles session/conversation lifecycle and temp file cleanup.
    private func runConversationInference(
        messageJSON: String,
        tempURLs: [URL],
        temperature: Float,
        maxTokens: Int
    ) async throws -> String {
        let urlsToCleanup = tempURLs
        return try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<String, any Error>) in
            self.inferenceQueue.async { [self, urlsToCleanup] in
                defer {
                    for url in urlsToCleanup {
                        try? FileManager.default.removeItem(at: url)
                    }
                }
                do {
                    guard let eng = self.engine else { throw LiteRTLMError.modelNotLoaded }

                    guard let sessionConfig = litert_lm_session_config_create() else {
                        throw LiteRTLMError.inferenceFailure("Failed to create session config")
                    }
                    litert_lm_session_config_set_max_output_tokens(sessionConfig, Int32(maxTokens))
                    var samplerParams = LiteRtLmSamplerParams(
                        type: kTopP, top_k: 40, top_p: 0.95,
                        temperature: temperature, seed: 0
                    )
                    litert_lm_session_config_set_sampler_params(sessionConfig, &samplerParams)

                    guard let convConfig = litert_lm_conversation_config_create(
                        eng, sessionConfig, nil, nil, nil, false
                    ) else {
                        litert_lm_session_config_delete(sessionConfig)
                        throw LiteRTLMError.inferenceFailure("Failed to create conversation config")
                    }

                    guard let conversation = litert_lm_conversation_create(eng, convConfig) else {
                        litert_lm_conversation_config_delete(convConfig)
                        litert_lm_session_config_delete(sessionConfig)
                        throw LiteRTLMError.inferenceFailure("Failed to create conversation")
                    }
                    self.resetCancellationState()
                    self.setActiveConversation(conversation)
                    defer {
                        self.setActiveConversation(nil)
                        litert_lm_conversation_delete(conversation)
                        litert_lm_conversation_config_delete(convConfig)
                        litert_lm_session_config_delete(sessionConfig)
                    }

                    guard let response = messageJSON.withCString({ msgPtr in
                        litert_lm_conversation_send_message(conversation, msgPtr, nil)
                    }) else {
                        if self.isCancelled() {
                            throw LiteRTLMError.cancelled
                        }
                        throw LiteRTLMError.inferenceFailure("Conversation returned no response")
                    }
                    defer { litert_lm_json_response_delete(response) }

                    guard let responsePtr = litert_lm_json_response_get_string(response) else {
                        throw LiteRTLMError.inferenceFailure("Response string is NULL")
                    }

                    let result = Self.extractTextFromConversationResponse(String(cString: responsePtr))
                    continuation.resume(returning: result)
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    // MARK: - Media Helpers

    /// Create a uniquely-named temp file URL.
    nonisolated static func makeTempURL(extension ext: String) -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString + "." + ext)
    }

    /// Remove temp files, ignoring errors (best-effort cleanup).
    nonisolated static func cleanupTempFiles(_ urls: [URL]) {
        for url in urls {
            try? FileManager.default.removeItem(at: url)
        }
    }

    /// Convert any image format to JPEG and resize for vision inference.
    nonisolated static func prepareImageForVision(_ data: Data, maxDimension: Int = 1024) -> Data? {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil),
              let cgImage = CGImageSourceCreateImageAtIndex(source, 0, nil) else { return nil }

        let width = cgImage.width
        let height = cgImage.height

        let maxDim = maxDimension
        let scale: Double
        if width > height {
            scale = width > maxDim ? Double(maxDim) / Double(width) : 1.0
        } else {
            scale = height > maxDim ? Double(maxDim) / Double(height) : 1.0
        }

        let targetWidth = Int(Double(width) * scale)
        let targetHeight = Int(Double(height) * scale)

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(
            data: nil,
            width: targetWidth,
            height: targetHeight,
            bitsPerComponent: 8,
            bytesPerRow: targetWidth * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
        ) else { return nil }

        context.interpolationQuality = .high
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: targetWidth, height: targetHeight))

        guard let resizedImage = context.makeImage() else { return nil }

        let mutableData = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(
            mutableData, "public.jpeg" as CFString, 1, nil
        ) else { return nil }

        let options: [CFString: Any] = [kCGImageDestinationLossyCompressionQuality: 0.85]
        CGImageDestinationAddImage(destination, resizedImage, options as CFDictionary)

        guard CGImageDestinationFinalize(destination) else { return nil }
        return mutableData as Data
    }

    /// Build a Conversation API JSON message with any combination of audio, images, and text.
    nonisolated static func buildMultimodalMessageJSON(
        audioPaths: [String],
        imagePaths: [String],
        text: String
    ) -> String {
        var contentItems: [[String: Any]] = []
        for path in audioPaths {
            contentItems.append(["type": "audio", "path": path])
        }
        for path in imagePaths {
            contentItems.append(["type": "image", "path": path])
        }
        contentItems.append(["type": "text", "text": text])
        let message: [String: Any] = ["role": "user", "content": contentItems]
        guard let jsonData = try? JSONSerialization.data(withJSONObject: message),
              let jsonString = String(data: jsonData, encoding: .utf8) else {
            // Fallback: text-only, properly escaped via JSONSerialization
            let fallback: [String: Any] = ["role": "user", "content": [["type": "text", "text": text]]]
            let fallbackData = (try? JSONSerialization.data(withJSONObject: fallback)) ?? Data()
            return String(data: fallbackData, encoding: .utf8) ?? "{}"
        }
        return jsonString
    }

    nonisolated static func extractTextFromConversationResponse(_ json: String) -> String {
        guard let data = json.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return json.trimmingCharacters(in: .whitespacesAndNewlines)
        }

        if let content = obj["content"] as? [[String: Any]] {
            let texts = content.compactMap { $0["text"] as? String }
            if !texts.isEmpty { return texts.joined(separator: " ") }
        }

        if let candidates = obj["candidates"] as? [[String: Any]],
           let first = candidates.first,
           let content = first["content"] as? [String: Any],
           let parts = content["parts"] as? [[String: Any]] {
            let texts = parts.compactMap { $0["text"] as? String }
            if !texts.isEmpty { return texts.joined(separator: " ") }
        }

        if let text = obj["text"] as? String { return text }

        return json.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

// MARK: - Stream Callback State

private final class StreamCallbackState: @unchecked Sendable {
    let continuation: AsyncThrowingStream<String, Error>.Continuation
    let doneSemaphore: DispatchSemaphore
    let isCancelled: () -> Bool
    /// Set once the consumer has been finished (cancelled or completed) so the
    /// callback drops further chunks instead of yielding into a dead continuation.
    var finished: Bool = false

    init(continuation: AsyncThrowingStream<String, Error>.Continuation,
         doneSemaphore: DispatchSemaphore,
         isCancelled: @escaping () -> Bool = { false }) {
        self.continuation = continuation
        self.doneSemaphore = doneSemaphore
        self.isCancelled = isCancelled
    }
}

extension LiteRTLMEngine {
    /// Shared C-callable streaming callback.
    ///
    /// Honors cancellation:
    /// - If `state.isCancelled()` returns true, finish the continuation with
    ///   `LiteRTLMError.cancelled` immediately and signal done so the caller
    ///   unblocks. Subsequent invocations are no-ops because `state.finished`
    ///   is set.
    static let streamCallback: @convention(c) (
        UnsafeMutableRawPointer?, UnsafePointer<CChar>?, Bool, UnsafePointer<CChar>?
    ) -> Void = { callbackData, chunk, isFinal, errorMsg in
        guard let cbData = callbackData else { return }
        let st = Unmanaged<StreamCallbackState>.fromOpaque(cbData).takeUnretainedValue()

        if st.finished { return }

        let errorMessage: String? = {
            guard let errorMsg else { return nil }
            let msg = String(cString: errorMsg)
            return msg.isEmpty ? nil : msg
        }()

        // Cancellation short-circuit: bail out as soon as the consumer asks.
        if st.isCancelled() && !isFinal && errorMessage == nil {
            st.finished = true
            st.continuation.finish(throwing: LiteRTLMError.cancelled)
            let semaphore = st.doneSemaphore
            Unmanaged<StreamCallbackState>.fromOpaque(cbData).release()
            semaphore.signal()
            return
        }

        if let chunk, errorMessage == nil {
            let text = String(cString: chunk)
            if !text.isEmpty { st.continuation.yield(text) }
        }

        if isFinal || errorMessage != nil {
            st.finished = true
            if let error = errorMessage {
                st.continuation.finish(throwing: LiteRTLMError.inferenceFailure(error))
            } else if st.isCancelled() {
                st.continuation.finish(throwing: LiteRTLMError.cancelled)
            } else {
                st.continuation.finish()
            }
            let semaphore = st.doneSemaphore
            Unmanaged<StreamCallbackState>.fromOpaque(cbData).release()
            semaphore.signal()
        }
    }
}

// MARK: - Errors

public enum LiteRTLMError: LocalizedError {
    case modelNotFound
    case modelNotLoaded
    case engineCreationFailed(String)
    case inferenceFailure(String)
    /// Inference was cancelled by the caller via `LiteRTLMEngine.cancel()` or
    /// by Task cancellation propagating to the streaming continuation.
    case cancelled

    public var errorDescription: String? {
        switch self {
        case .modelNotFound:
            "LiteRT-LM model file not found"
        case .modelNotLoaded:
            "LiteRT-LM model is not loaded — call load() first"
        case .engineCreationFailed(let detail):
            "Failed to create LiteRT-LM engine: \(detail)"
        case .inferenceFailure(let detail):
            "LiteRT-LM inference failed: \(detail)"
        case .cancelled:
            "LiteRT-LM inference was cancelled"
        }
    }
}
