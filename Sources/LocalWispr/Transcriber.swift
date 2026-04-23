@preconcurrency import AVFAudio
import Foundation
@preconcurrency import Speech

public enum TranscriberError: LocalizedError {
    case emptyAudio
    case sessionInitializationFailed
    case unsupportedLocale(String)
    case localeAllocationFailed(String)
    case speechAnalysisFailed(String)
    case legacyRecognizerUnavailable(String)
    case legacyRecognitionFailed(String)
    case cloudRecognitionFailed(String)

    public var errorDescription: String? {
        switch self {
        case .emptyAudio:
            return "No audio buffers were captured."
        case .sessionInitializationFailed:
            return "Unable to initialize live transcription session."
        case .unsupportedLocale(let localeIdentifier):
            return "Speech transcription is not available for locale \(localeIdentifier)."
        case .localeAllocationFailed(let localeIdentifier):
            return "Unable to allocate speech assets for locale \(localeIdentifier)."
        case .speechAnalysisFailed(let details):
            return "Speech analysis failed: \(details)"
        case .legacyRecognizerUnavailable(let localeIdentifier):
            return "Legacy speech recognition is not available for locale \(localeIdentifier)."
        case .legacyRecognitionFailed(let details):
            return "Legacy speech recognition failed: \(details)"
        case .cloudRecognitionFailed(let details):
            return "Cloud speech recognition failed: \(details)"
        }
    }
}

private struct GroqVerboseTranscriptionResponse: Decodable {
    struct Metrics: Decodable, Sendable {
        let decodeMs: Int?
        let requestReadMs: Int?
        let audioPrepareMs: Int?
        let totalMs: Int?
        let audioSeconds: Double?

        private enum CodingKeys: String, CodingKey {
            case decodeMs = "decode_ms"
            case requestReadMs = "request_read_ms"
            case audioPrepareMs = "audio_prepare_ms"
            case totalMs = "total_ms"
            case audioSeconds = "audio_seconds"
        }
    }

    struct Segment: Decodable, Sendable {
        let id: Int?
        let start: Double
        let end: Double
        let text: String
        let avgLogprob: Double?
        let compressionRatio: Double?
        let noSpeechProb: Double?

        private enum CodingKeys: String, CodingKey {
            case id
            case start
            case end
            case text
            case avgLogprob = "avg_logprob"
            case compressionRatio = "compression_ratio"
            case noSpeechProb = "no_speech_prob"
        }
    }

    let text: String
    let segments: [Segment]?
    let decodeMs: Int?
    let metrics: Metrics?

    private enum CodingKeys: String, CodingKey {
        case text
        case segments
        case decodeMs = "decode_ms"
        case metrics
    }

    var serverDecodeMilliseconds: Int? {
        metrics?.decodeMs ?? decodeMs
    }

    var serverRequestReadMilliseconds: Int? {
        metrics?.requestReadMs
    }

    var serverAudioPrepareMilliseconds: Int? {
        metrics?.audioPrepareMs
    }

    var serverTotalMilliseconds: Int? {
        metrics?.totalMs ?? serverDecodeMilliseconds
    }

    var reportedAudioSeconds: Double? {
        metrics?.audioSeconds
    }
}

struct GroqStreamingChunkPlanner {
    let chunkDurationSeconds: Double
    let overlapSeconds: Double

    private(set) var nextRegularStartSeconds: Double = 0
    private(set) var lastDispatchedWindow: ClosedRange<Double>?

    init(
        chunkDurationSeconds: Double = 10,
        overlapSeconds: Double = 2
    ) {
        self.chunkDurationSeconds = max(1, chunkDurationSeconds)
        self.overlapSeconds = max(0, min(overlapSeconds, self.chunkDurationSeconds / 2))
    }

    mutating func nextRegularWindow(bufferedDurationSeconds: Double) -> ClosedRange<Double>? {
        let end = nextRegularStartSeconds + chunkDurationSeconds
        guard bufferedDurationSeconds + 0.0001 >= end else { return nil }

        let window = nextRegularStartSeconds...end
        lastDispatchedWindow = window
        nextRegularStartSeconds += max(0.25, chunkDurationSeconds - overlapSeconds)
        return window
    }

    mutating func finalWindow(totalDurationSeconds: Double) -> ClosedRange<Double>? {
        guard totalDurationSeconds > 0 else { return nil }

        let start = max(0, totalDurationSeconds - chunkDurationSeconds)
        let window = start...totalDurationSeconds

        if let lastDispatchedWindow {
            let sameStart = abs(lastDispatchedWindow.lowerBound - window.lowerBound) < 0.01
            let sameEnd = abs(lastDispatchedWindow.upperBound - window.upperBound) < 0.01
            if sameStart && sameEnd {
                return nil
            }
        }

        lastDispatchedWindow = window
        return window
    }
}

private extension Data {
    mutating func appendUTF8(_ string: String) {
        append(contentsOf: string.utf8)
    }

    mutating func appendLittleEndian<T: FixedWidthInteger>(_ value: T) {
        var littleEndianValue = value.littleEndian
        Swift.withUnsafeBytes(of: &littleEndianValue) { buffer in
            append(buffer.bindMemory(to: UInt8.self))
        }
    }
}

private func describeBufferLevel(_ buffer: AVAudioPCMBuffer) -> String {
    switch buffer.format.commonFormat {
    case .pcmFormatFloat32:
        guard let channels = buffer.floatChannelData else { return "unavailable" }
        let frameCount = Int(buffer.frameLength)
        guard frameCount > 0 else { return "empty" }

        let samples = UnsafeBufferPointer(start: channels[0], count: frameCount)
        let rms = sqrt(samples.reduce(0.0) { $0 + Double($1 * $1) } / Double(frameCount))
        let peak = samples.reduce(0.0) { max($0, Double(abs($1))) }
        return String(format: "rms=%.5f peak=%.5f", rms, peak)

    case .pcmFormatInt16:
        guard let channels = buffer.int16ChannelData else { return "unavailable" }
        let frameCount = Int(buffer.frameLength)
        guard frameCount > 0 else { return "empty" }

        let samples = UnsafeBufferPointer(start: channels[0], count: frameCount)
        let scale = Double(Int16.max)
        let rms = sqrt(samples.reduce(0.0) { partial, sample in
            let normalized = Double(sample) / scale
            return partial + (normalized * normalized)
        } / Double(frameCount))
        let peak = samples.reduce(0.0) { partial, sample in
            max(partial, abs(Double(sample) / scale))
        }
        return String(format: "rms=%.5f peak=%.5f", rms, peak)

    default:
        return "unsupported-format"
    }
}

private final class LegacyRecognitionBridge: @unchecked Sendable {
    private let lock = NSLock()
    private var continuation: CheckedContinuation<String, Error>?
    private var task: SFSpeechRecognitionTask?

    func install(_ continuation: CheckedContinuation<String, Error>) {
        lock.lock()
        self.continuation = continuation
        lock.unlock()
    }

    func setTask(_ task: SFSpeechRecognitionTask?) {
        lock.lock()
        self.task = task
        lock.unlock()
    }

    func succeed(_ transcript: String) {
        resume(with: .success(transcript))
    }

    func fail(_ error: Error) {
        resume(with: .failure(error))
    }

    func cancel() {
        resume(with: .failure(CancellationError()))
    }

    private func resume(with result: Result<String, Error>) {
        lock.lock()
        let continuation = self.continuation
        self.continuation = nil
        let task = self.task
        self.task = nil
        lock.unlock()

        task?.cancel()

        guard let continuation else { return }
        switch result {
        case .success(let transcript):
            continuation.resume(returning: transcript)
        case .failure(let error):
            continuation.resume(throwing: error)
        }
    }
}

private final class BufferedCloudTranscriptionSession: @unchecked Sendable, LiveTranscriptionSession {
    private let transcribe: @Sendable ([Float]) async throws -> String
    private let stateLock = NSLock()
    private var samples: [Float] = []
    private var isFinished = false
    private var isCancelled = false

    init(transcribe: @escaping @Sendable ([Float]) async throws -> String) {
        self.transcribe = transcribe
    }

    func append(_ buffer: AVAudioPCMBuffer) {
        let converted = Self.convertToFloatSamples(buffer)
        guard !converted.isEmpty else { return }

        stateLock.lock()
        defer { stateLock.unlock() }
        guard !isFinished, !isCancelled else { return }
        samples.append(contentsOf: converted)
    }

    func finish() async throws -> String {
        let snapshot: [Float] = try withStateLock {
            if isCancelled {
                throw CancellationError()
            }
            isFinished = true
            return samples
        }

        guard !snapshot.isEmpty else {
            throw TranscriberError.emptyAudio
        }
        return try await transcribe(snapshot)
    }

    func cancel() async {
        withStateLock {
            isCancelled = true
            samples.removeAll(keepingCapacity: false)
        }
    }

    private static func convertToFloatSamples(_ buffer: AVAudioPCMBuffer) -> [Float] {
        let frameCount = Int(buffer.frameLength)
        guard frameCount > 0 else { return [] }

        switch buffer.format.commonFormat {
        case .pcmFormatFloat32:
            guard let channels = buffer.floatChannelData else { return [] }
            return Array(UnsafeBufferPointer(start: channels[0], count: frameCount))
        case .pcmFormatInt16:
            guard let channels = buffer.int16ChannelData else { return [] }
            let source = UnsafeBufferPointer(start: channels[0], count: frameCount)
            let scale = Float(Int16.max)
            return source.map { Float($0) / scale }
        default:
            return []
        }
    }

    private func withStateLock<T>(_ body: () throws -> T) rethrows -> T {
        stateLock.lock()
        defer { stateLock.unlock() }
        return try body()
    }
}

private final class GroqLiveTranscriptionSession: @unchecked Sendable, LiveTranscriptionSession {
    private let sampleRate: Double
    private let transcribeChunk: @Sendable ([Float], ClosedRange<Double>) async throws -> GroqVerboseTranscriptionResponse
    private let transcribeFull: @Sendable ([Float]) async throws -> String
    private let transcriptUpdateHandler: (@Sendable (String) -> Void)?
    private let streamingQueue = ParakeetStreamingQueue()
    private let stateLock = NSLock()

    private var planner = GroqStreamingChunkPlanner()
    private var samples: [Float] = []
    private var assembler = TranscriptAssembler()
    private var latestAssembledTranscript: String = ""
    private var chunkFailure: Error?
    private var isFinished = false
    private var chunkRequestCount = 0
    private var bufferCount = 0

    init(
        sampleRate: Double,
        transcriptUpdateHandler: (@Sendable (String) -> Void)? = nil,
        transcribeChunk: @escaping @Sendable ([Float], ClosedRange<Double>) async throws -> GroqVerboseTranscriptionResponse,
        transcribeFull: @escaping @Sendable ([Float]) async throws -> String
    ) {
        self.sampleRate = sampleRate
        self.transcriptUpdateHandler = transcriptUpdateHandler
        self.transcribeChunk = transcribeChunk
        self.transcribeFull = transcribeFull
    }

    func append(_ buffer: AVAudioPCMBuffer) {
        let converted = Self.convertToFloatSamples(buffer)
        guard !converted.isEmpty else { return }

        let dispatches: [(window: ClosedRange<Double>, chunkSamples: [Float])] = withStateLock {
            guard !isFinished else { return [] }

            samples.append(contentsOf: converted)
            bufferCount += 1
            if bufferCount == 1 || bufferCount % 50 == 0 {
                DebugLog.write("[GroqLive] buffer #\(bufferCount) frames=\(buffer.frameLength) samples=\(samples.count)")
            }

            let bufferedDurationSeconds = Double(samples.count) / sampleRate
            var windows: [(window: ClosedRange<Double>, chunkSamples: [Float])] = []
            while let window = planner.nextRegularWindow(bufferedDurationSeconds: bufferedDurationSeconds) {
                windows.append((window: window, chunkSamples: sliceSamples(in: window, from: samples)))
            }
            return windows
        }

        for dispatch in dispatches {
            enqueueChunk(dispatch.chunkSamples, window: dispatch.window, isFinalWindow: false)
        }
    }

    func finish() async throws -> String {
        let finishState = withStateLock { () -> (snapshot: [Float], finalWindow: ClosedRange<Double>?) in
            if !isFinished {
                isFinished = true
            }

            let snapshot = samples
            let totalDurationSeconds = Double(snapshot.count) / sampleRate
            let finalWindow = planner.finalWindow(totalDurationSeconds: totalDurationSeconds)
            return (snapshot, finalWindow)
        }

        guard !finishState.snapshot.isEmpty else { return "" }

        if let finalWindow = finishState.finalWindow {
            let tailSamples = sliceSamples(in: finalWindow, from: finishState.snapshot)
            enqueueChunk(tailSamples, window: finalWindow, isFinalWindow: true)
        }

        await streamingQueue.waitForIdle()

        let finalizedState = withStateLock { () -> (assembledTranscript: String, chunkCount: Int, chunkFailure: Error?) in
            (
                assembledTranscript: latestAssembledTranscript.trimmingCharacters(in: .whitespacesAndNewlines),
                chunkCount: chunkRequestCount,
                chunkFailure: chunkFailure
            )
        }

        if finalizedState.chunkCount > 0, finalizedState.chunkFailure == nil, !finalizedState.assembledTranscript.isEmpty {
            DebugLog.write("[GroqLive] assembled final transcript from \(finalizedState.chunkCount) background chunk(s)")
            return finalizedState.assembledTranscript
        }

        if let failure = finalizedState.chunkFailure {
            DebugLog.write("[GroqLive] falling back to single-request finalization after chunk failure: \(failure.localizedDescription)")
        } else {
            DebugLog.write("[GroqLive] no background chunks available, using single-request finalization")
        }

        return try await transcribeFull(finishState.snapshot).trimmingCharacters(in: .whitespacesAndNewlines)
    }

    func cancel() async {
        withStateLock {
            isFinished = true
            samples.removeAll(keepingCapacity: false)
        }
        await streamingQueue.waitForIdle()
    }

    private func enqueueChunk(_ chunkSamples: [Float], window: ClosedRange<Double>, isFinalWindow: Bool) {
        guard !chunkSamples.isEmpty else { return }
        withStateLock {
            chunkRequestCount += 1
        }
        let logPrefix = isFinalWindow ? "[GroqLive] final chunk" : "[GroqLive] chunk"
        DebugLog.write("\(logPrefix) queued start=\(Self.formatSeconds(window.lowerBound)) end=\(Self.formatSeconds(window.upperBound)) duration=\(Self.formatSeconds(window.upperBound - window.lowerBound))")

        streamingQueue.enqueue { [weak self, chunkSamples, window, isFinalWindow] in
            guard let self else { return }

            do {
                let response = try await self.transcribeChunk(chunkSamples, window)
                let updatedTranscript = self.withStateLock { () -> String in
                    Self.consume(response: response, absoluteWindow: window, into: &self.assembler)
                    let transcript = self.assembler.transcript.trimmingCharacters(in: .whitespacesAndNewlines)
                    self.latestAssembledTranscript = transcript
                    return transcript
                }

                DebugLog.write("[GroqLive] \(isFinalWindow ? "final " : "")chunk success start=\(Self.formatSeconds(window.lowerBound)) end=\(Self.formatSeconds(window.upperBound)) transcript=\(updatedTranscript.prefix(120))")
                if !updatedTranscript.isEmpty {
                    self.transcriptUpdateHandler?(updatedTranscript)
                }
            } catch {
                self.withStateLock {
                    if self.chunkFailure == nil {
                        self.chunkFailure = error
                    }
                }
                DebugLog.write("[GroqLive] \(isFinalWindow ? "final " : "")chunk failed start=\(Self.formatSeconds(window.lowerBound)) end=\(Self.formatSeconds(window.upperBound)) error=\(error.localizedDescription)")
            }
        }
    }

    private func withStateLock<T>(_ body: () -> T) -> T {
        stateLock.lock()
        defer { stateLock.unlock() }
        return body()
    }

    private func sliceSamples(in window: ClosedRange<Double>, from source: [Float]) -> [Float] {
        let startIndex = max(0, Int((window.lowerBound * sampleRate).rounded(.down)))
        let endIndex = min(source.count, Int((window.upperBound * sampleRate).rounded(.up)))
        guard startIndex < endIndex else { return [] }
        return Array(source[startIndex..<endIndex])
    }

    private static func consume(
        response: GroqVerboseTranscriptionResponse,
        absoluteWindow: ClosedRange<Double>,
        into assembler: inout TranscriptAssembler
    ) {
        if let segments = response.segments, !segments.isEmpty {
            for segment in segments {
                let start = max(absoluteWindow.lowerBound, absoluteWindow.lowerBound + segment.start)
                let end = min(absoluteWindow.upperBound, absoluteWindow.lowerBound + segment.end)
                guard end > start else { continue }

                assembler.consume(
                    text: segment.text,
                    range: CMTimeRange(
                        start: CMTime(seconds: start, preferredTimescale: 600),
                        duration: CMTime(seconds: end - start, preferredTimescale: 600)
                    ),
                    isFinal: true
                )
            }
            return
        }

        let normalized = response.text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !normalized.isEmpty else { return }
        assembler.consume(
            text: normalized,
            range: CMTimeRange(
                start: CMTime(seconds: absoluteWindow.lowerBound, preferredTimescale: 600),
                duration: CMTime(seconds: absoluteWindow.upperBound - absoluteWindow.lowerBound, preferredTimescale: 600)
            ),
            isFinal: true
        )
    }

    private static func convertToFloatSamples(_ buffer: AVAudioPCMBuffer) -> [Float] {
        let frameCount = Int(buffer.frameLength)
        guard frameCount > 0 else { return [] }

        switch buffer.format.commonFormat {
        case .pcmFormatFloat32:
            guard let channels = buffer.floatChannelData else { return [] }
            return Array(UnsafeBufferPointer(start: channels[0], count: frameCount))
        case .pcmFormatInt16:
            guard let channels = buffer.int16ChannelData else { return [] }
            let source = UnsafeBufferPointer(start: channels[0], count: frameCount)
            let scale = Float(Int16.max)
            return source.map { Float($0) / scale }
        default:
            return []
        }
    }

    private static func formatSeconds(_ value: Double) -> String {
        String(format: "%.2f", value)
    }
}

private final class LegacyLiveTranscriptionSession: @unchecked Sendable, LiveTranscriptionSession {
    private let request = SFSpeechAudioBufferRecognitionRequest()
    private let stateLock = NSLock()

    private var recognitionTask: SFSpeechRecognitionTask?
    private var waitingContinuations: [CheckedContinuation<String, Error>] = []
    private var latestTranscript: String = ""
    private var hasEndedAudio = false
    private var completedResult: Result<String, Error>?
    private var appendedBufferCount = 0

    init(
        recognizer: SFSpeechRecognizer,
        mode: TranscriberMode,
        contextualStrings: [String]
    ) {
        request.shouldReportPartialResults = true
        request.requiresOnDeviceRecognition = true
        request.taskHint = mode == .dictationLong ? .dictation : .unspecified
        if !contextualStrings.isEmpty {
            request.contextualStrings = contextualStrings
        }

        recognitionTask = recognizer.recognitionTask(with: request) { [weak self] result, error in
            self?.handle(result: result, error: error)
        }
    }

    func append(_ buffer: AVAudioPCMBuffer) {
        let shouldAppend: Bool = withLock {
            completedResult == nil && !hasEndedAudio
        }
        guard shouldAppend else { return }

        let logLine: String? = withLock {
            appendedBufferCount += 1
            guard appendedBufferCount == 1 || appendedBufferCount % 50 == 0 else { return nil }
            return "[LegacyLive] buffer #\(appendedBufferCount) frames=\(buffer.frameLength) format=\(buffer.format) level=\(describeBufferLevel(buffer))"
        }
        if let logLine {
            DebugLog.write(logLine)
        }

        request.append(buffer)
    }

    func finish() async throws -> String {
        if let completedResult = withLock({ self.completedResult }) {
            return try completedResult.get()
        }

        return try await withCheckedThrowingContinuation { continuation in
            let immediateResult: Result<String, Error>? = withLock {
                if let completedResult = self.completedResult {
                    return completedResult
                }

                waitingContinuations.append(continuation)
                if !hasEndedAudio {
                    hasEndedAudio = true
                    request.endAudio()
                }
                return nil
            }

            if let immediateResult {
                switch immediateResult {
                case .success(let transcript):
                    continuation.resume(returning: transcript)
                case .failure(let error):
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    func cancel() async {
        let continuations: [CheckedContinuation<String, Error>] = withLock {
            recognitionTask?.cancel()
            recognitionTask = nil
            request.endAudio()
            let continuations = waitingContinuations
            waitingContinuations = []
            completedResult = .failure(CancellationError())
            return continuations
        }

        for continuation in continuations {
            continuation.resume(throwing: CancellationError())
        }
    }

    private func handle(result: SFSpeechRecognitionResult?, error: Error?) {
        if let result {
            let transcript = result.bestTranscription.formattedString.trimmingCharacters(in: .whitespacesAndNewlines)
            if !transcript.isEmpty {
                withLock {
                    latestTranscript = transcript
                }
            }

            DebugLog.write("[LegacyLive] result isFinal=\(result.isFinal) text=\(transcript.prefix(80))")
            if result.isFinal {
                if let resolvedTranscript = TranscriptResolutionPolicy.preferredTranscript(
                    primary: transcript,
                    alternative: withLock { latestTranscript }
                ) {
                    resolve(.success(resolvedTranscript))
                } else {
                    resolve(.failure(TranscriberError.legacyRecognitionFailed("Empty transcript")))
                }
                return
            }
        }

        if let error {
            DebugLog.write("[LegacyLive] failed: \(error.localizedDescription)")
            let fallbackTranscript = withLock { latestTranscript }
            if !fallbackTranscript.isEmpty {
                resolve(.success(fallbackTranscript))
            } else {
                resolve(.failure(TranscriberError.legacyRecognitionFailed(error.localizedDescription)))
            }
        }
    }

    private func resolve(_ result: Result<String, Error>) {
        let continuations: [CheckedContinuation<String, Error>] = withLock {
            if completedResult != nil {
                return []
            }

            completedResult = result
            recognitionTask = nil
            let continuations = waitingContinuations
            waitingContinuations = []
            return continuations
        }

        for continuation in continuations {
            switch result {
            case .success(let transcript):
                continuation.resume(returning: transcript)
            case .failure(let error):
                continuation.resume(throwing: error)
            }
        }
    }

    private func withLock<T>(_ body: () -> T) -> T {
        stateLock.lock()
        defer { stateLock.unlock() }
        return body()
    }
}

private final class InternalLiveTranscriptionSession: @unchecked Sendable, LiveTranscriptionSession {
    private let analyzer: SpeechAnalyzer
    private let startTask: Task<Void, Error>
    private let collectorTask: Task<String, Error>

    private let stateLock = NSLock()
    private var continuation: AsyncStream<AnalyzerInput>.Continuation?
    private var finished = false

    init(
        analyzer: SpeechAnalyzer,
        continuation: AsyncStream<AnalyzerInput>.Continuation,
        startTask: Task<Void, Error>,
        collectorTask: Task<String, Error>
    ) {
        self.analyzer = analyzer
        self.continuation = continuation
        self.startTask = startTask
        self.collectorTask = collectorTask
    }

    private var bufferCount = 0

    func append(_ buffer: AVAudioPCMBuffer) {
        let state = withStateLock {
            (continuation: continuation, isFinished: finished)
        }

        guard !state.isFinished else { return }
        bufferCount += 1
        if bufferCount == 1 || bufferCount % 50 == 0 {
            DebugLog.write("[Session] buffer #\(bufferCount) frames=\(buffer.frameLength) format=\(buffer.format) level=\(describeBufferLevel(buffer))")
        }
        state.continuation?.yield(AnalyzerInput(buffer: buffer))
    }

    func finish() async throws -> String {
        let finishState = withStateLock { () -> (alreadyFinished: Bool, continuation: AsyncStream<AnalyzerInput>.Continuation?) in
            if finished {
                return (alreadyFinished: true, continuation: nil)
            }

            finished = true
            let continuation = self.continuation
            self.continuation = nil
            return (alreadyFinished: false, continuation: continuation)
        }

        if finishState.alreadyFinished {
            return try await collectorTask.value.trimmingCharacters(in: .whitespacesAndNewlines)
        }

        finishState.continuation?.finish()

        do {
            try await analyzer.finalizeAndFinishThroughEndOfInput()
            try await startTask.value
            return try await collectorTask.value.trimmingCharacters(in: .whitespacesAndNewlines)
        } catch {
            startTask.cancel()
            collectorTask.cancel()
            await analyzer.cancelAndFinishNow()
            throw error
        }
    }

    func cancel() async {
        let cancelState = withStateLock { () -> (alreadyFinished: Bool, continuation: AsyncStream<AnalyzerInput>.Continuation?) in
            if finished {
                return (alreadyFinished: true, continuation: nil)
            }

            finished = true
            let continuation = self.continuation
            self.continuation = nil
            return (alreadyFinished: false, continuation: continuation)
        }

        if cancelState.alreadyFinished {
            return
        }

        cancelState.continuation?.finish()
        startTask.cancel()
        collectorTask.cancel()
        await analyzer.cancelAndFinishNow()
    }

    private func withStateLock<T>(_ body: () -> T) -> T {
        stateLock.lock()
        defer { stateLock.unlock() }
        return body()
    }
}

private final class ParakeetStreamingQueue: @unchecked Sendable {
    private let lock = NSLock()
    private var tailTask: Task<Void, Never>?

    func enqueue(_ operation: @escaping @Sendable () async -> Void) {
        lock.lock()
        let previous = tailTask
        let task = Task {
            _ = await previous?.result
            await operation()
        }
        tailTask = task
        lock.unlock()
    }

    func waitForIdle() async {
        let task = takeTailTask()
        _ = await task?.result
    }

    private func takeTailTask() -> Task<Void, Never>? {
        lock.lock()
        defer { lock.unlock() }
        let task = tailTask
        tailTask = nil
        return task
    }
}

private final class ParakeetLiveTranscriptionSession: @unchecked Sendable, LiveTranscriptionSession {
    private let provider: ParakeetRealtimeProvider
    private let streamingQueue = ParakeetStreamingQueue()
    private let stateLock = NSLock()

    private var samples: [Float] = []
    private var isFinished = false
    private var bufferCount = 0
    private var lastStreamingDispatchUptime: TimeInterval = 0

    private let minimumStreamingSamples = 3_200 // 200ms @ 16kHz
    private let streamingDispatchInterval: TimeInterval = 0.2

    init(provider: ParakeetRealtimeProvider) {
        self.provider = provider
    }

    func append(_ buffer: AVAudioPCMBuffer) {
        let converted = Self.convertToFloatSamples(buffer)
        guard !converted.isEmpty else { return }

        let state = withStateLock { () -> (finished: Bool, shouldDispatch: Bool, snapshot: [Float]) in
            if isFinished {
                return (finished: true, shouldDispatch: false, snapshot: [])
            }

            samples.append(contentsOf: converted)
            bufferCount += 1
            if bufferCount == 1 || bufferCount % 50 == 0 {
                DebugLog.write("[ParakeetLive] buffer #\(bufferCount) frames=\(buffer.frameLength) samples=\(samples.count)")
            }

            let now = ProcessInfo.processInfo.systemUptime
            let hasEnoughAudio = samples.count >= minimumStreamingSamples
            let canDispatch = (now - lastStreamingDispatchUptime) >= streamingDispatchInterval

            if hasEnoughAudio && canDispatch {
                lastStreamingDispatchUptime = now
                return (finished: false, shouldDispatch: true, snapshot: samples)
            }

            return (finished: false, shouldDispatch: false, snapshot: [])
        }

        guard !state.finished, state.shouldDispatch else { return }
        streamingQueue.enqueue { [provider, snapshot = state.snapshot] in
            _ = try? await provider.transcribeStreaming(snapshot)
        }
    }

    func finish() async throws -> String {
        let snapshot = withStateLock { () -> [Float] in
            guard !isFinished else { return samples }
            isFinished = true
            return samples
        }

        guard !snapshot.isEmpty else { return "" }

        await streamingQueue.waitForIdle()
        let transcript = try await provider.transcribeFinal(snapshot)
        return transcript.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    func cancel() async {
        withStateLock {
            isFinished = true
        }
        await streamingQueue.waitForIdle()
        await provider.resetSession()
    }

    private func withStateLock<T>(_ body: () -> T) -> T {
        stateLock.lock()
        defer { stateLock.unlock() }
        return body()
    }

    private static func convertToFloatSamples(_ buffer: AVAudioPCMBuffer) -> [Float] {
        let frameCount = Int(buffer.frameLength)
        guard frameCount > 0 else { return [] }

        switch buffer.format.commonFormat {
        case .pcmFormatFloat32:
            guard let channels = buffer.floatChannelData else { return [] }
            return Array(UnsafeBufferPointer(start: channels[0], count: frameCount))
        case .pcmFormatInt16:
            guard let channels = buffer.int16ChannelData else { return [] }
            let source = UnsafeBufferPointer(start: channels[0], count: frameCount)
            let scale = Float(Int16.max)
            return source.map { Float($0) / scale }
        default:
            return []
        }
    }
}

private final class IdentifierCorrectingLiveSession: @unchecked Sendable, LiveTranscriptionSession {
    private let base: any LiveTranscriptionSession
    private let identifiers: [String]
    private let label: String

    init(base: any LiveTranscriptionSession, identifiers: [String], label: String) {
        self.base = base
        self.identifiers = identifiers
        self.label = label
    }

    func append(_ buffer: AVAudioPCMBuffer) {
        base.append(buffer)
    }

    func finish() async throws -> String {
        let transcript = try await base.finish()
        return Self.applyContextualCorrections(transcript, identifiers: identifiers, label: label)
    }

    func cancel() async {
        await base.cancel()
    }

    private static func applyContextualCorrections(_ transcript: String, identifiers: [String], label: String) -> String {
        let trimmed = transcript.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, !identifiers.isEmpty else { return trimmed }

        let corrected = FuzzyIdentifierMatcher.postProcess(trimmed, identifiers: identifiers)
        if corrected != trimmed {
            DebugLog.write("[ContextBias:\(label)] \(trimmed.prefix(120)) -> \(corrected.prefix(120))")
        }
        return corrected
    }
}

public final class Transcriber: @unchecked Sendable, Transcribing, LiveTranscribing, SpeechModelWarming {
    public typealias ModelProgressHandler = @Sendable (Progress) -> Void

    public var onModelPreparationProgress: ModelProgressHandler?
    public var onLiveTranscriptUpdate: (@Sendable (String) -> Void)?

    private let localeCoordinator = SpeechLocaleCoordinator()
    private var parakeetRealtimeProvider: ParakeetRealtimeProvider?

    public init() {}

    public func prewarm(mode: TranscriberMode, locale: Locale, audioFormat: AVAudioFormat) async {
        _ = mode
        _ = locale
        _ = audioFormat
        // Modal-hosted STT has no local warmup step.
    }

    public func startSession(
        mode: TranscriberMode,
        locale: Locale,
        audioFormat: AVAudioFormat,
        contextualStrings: [String] = []
    ) async throws -> any LiveTranscriptionSession {
        let env = DotEnv.merged()
        let config = try Self.modalSTTConfig(from: env)
        let prompt = Self.transcriptionPrompt(from: contextualStrings)
        DebugLog.write("[Transcriber] using Modal STT buffered session locale=\(locale.identifier) mode=\(mode) inputFormat=\(audioFormat)")

        return BufferedCloudTranscriptionSession { [weak self] samples in
            guard let self else {
                throw CancellationError()
            }
            let response = try await self.transcribeSamplesWithModal(
                samples: samples,
                locale: locale,
                config: config,
                prompt: prompt
            )
            return response.text
        }
    }

    public func transcribe(
        buffers: [AVAudioPCMBuffer],
        mode: TranscriberMode,
        locale: Locale,
        contextualStrings: [String]
    ) async throws -> String {
        guard !buffers.isEmpty else {
            throw TranscriberError.emptyAudio
        }

        let env = DotEnv.merged()
        _ = try Self.modalSTTConfig(from: env)
        let prompt = Self.transcriptionPrompt(from: contextualStrings)
        DebugLog.write("[Transcriber] using Modal STT batch locale=\(locale.identifier) mode=\(mode) buffers=\(buffers.count)")
        return try await transcribeWithModal(buffers: buffers, locale: locale, env: env, prompt: prompt)
    }

    private struct ModalSTTConfig {
        let endpoint: URL
        let apiKey: String
        let model: String
        let timeoutSeconds: TimeInterval
    }

    private static func modalSTTConfig(from env: [String: String]) throws -> ModalSTTConfig {
        guard let endpointValue = env["LOCALWISPR_MODAL_STT_ENDPOINT"]?.trimmingCharacters(in: .whitespacesAndNewlines),
              !endpointValue.isEmpty,
              let endpoint = URL(string: endpointValue) else {
            throw TranscriberError.cloudRecognitionFailed("Missing LOCALWISPR_MODAL_STT_ENDPOINT")
        }

        guard let apiKey = env["LOCALWISPR_MODAL_STT_API_KEY"]?.trimmingCharacters(in: .whitespacesAndNewlines),
              !apiKey.isEmpty else {
            throw TranscriberError.cloudRecognitionFailed("Missing LOCALWISPR_MODAL_STT_API_KEY")
        }

        let model = env["LOCALWISPR_MODAL_STT_MODEL"]?.trimmingCharacters(in: .whitespacesAndNewlines)
        let resolvedModel = (model?.isEmpty == false) ? model! : "openai/whisper-large-v3-turbo"

        let timeoutValue = env["LOCALWISPR_MODAL_STT_TIMEOUT_SECONDS"]?.trimmingCharacters(in: .whitespacesAndNewlines)
        let timeoutSeconds = timeoutValue.flatMap(TimeInterval.init) ?? 60

        return ModalSTTConfig(
            endpoint: endpoint,
            apiKey: apiKey,
            model: resolvedModel,
            timeoutSeconds: max(5, timeoutSeconds)
        )
    }

    private static func transcriptionPrompt(from contextualStrings: [String]) -> String? {
        let hints = Array(
            NSOrderedSet(
                array: contextualStrings
                    .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                    .filter { !$0.isEmpty }
            )
        ) as? [String] ?? contextualStrings

        let topHints = Array(hints.prefix(12))
        guard !topHints.isEmpty else { return nil }

        return "Use these preferred spellings when they fit the audio: \(topHints.joined(separator: ", "))."
    }

    private func shouldUseParakeet(locale: Locale, environment: [String: String]) -> Bool {
        #if arch(arm64)
        if let rawBackend = explicitBackend(environment: environment) {
            if rawBackend == "apple" || rawBackend == "speech" {
                return false
            }
            if rawBackend == "parakeet" {
                return true
            }
        }

        // Parakeet Flash is currently English-focused in upstream integration.
        let languageCode = locale.language.languageCode?.identifier.lowercased() ?? locale.identifier.lowercased()
        return languageCode.hasPrefix("en")
        #else
        return false
        #endif
    }

    private func explicitBackend(environment: [String: String]) -> String? {
        environment["LOCALWISPR_ASR_BACKEND"]?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
    }

    private func isParakeetExplicitlyRequested(environment: [String: String]) -> Bool {
        explicitBackend(environment: environment) == "parakeet"
    }

    private func ensureParakeetReady() async throws -> ParakeetRealtimeProvider {
        let provider = getParakeetProvider()
        if provider.isReady {
            return provider
        }

        try await provider.prepare(progressHandler: { [weak self] fraction in
            self?.emitModelPreparationProgress(fractionCompleted: fraction)
        })
        return provider
    }

    private func getParakeetProvider() -> ParakeetRealtimeProvider {
        if let parakeetRealtimeProvider {
            return parakeetRealtimeProvider
        }

        let provider = ParakeetRealtimeProvider()
        self.parakeetRealtimeProvider = provider
        return provider
    }

    private func emitModelPreparationProgress(fractionCompleted: Double) {
        guard let onModelPreparationProgress else { return }

        let bounded = max(0.0, min(1.0, fractionCompleted))
        let progress = Progress(totalUnitCount: 1_000)
        progress.completedUnitCount = Int64((bounded * 1_000.0).rounded())
        onModelPreparationProgress(progress)
    }

    private func transcribeWithParakeet(buffers: [AVAudioPCMBuffer], contextualStrings: [String]) async throws -> String {
        let provider = try await ensureParakeetReady()
        let samples = Self.convertBuffersToFloatSamples(buffers)
        guard !samples.isEmpty else {
            throw TranscriberError.emptyAudio
        }

        let transcript = try await provider.transcribeFinal(samples)
        if let normalized = TranscriptResolutionPolicy.normalizedTranscript(transcript) {
            let corrected = applyContextualCorrections(normalized, identifiers: contextualStrings, label: "batch-parakeet")
            DebugLog.write("[ParakeetBatch] success text=\(corrected.prefix(120))")
            return corrected
        }

        throw TranscriberError.speechAnalysisFailed("Parakeet returned empty transcript")
    }

    private static func convertBuffersToFloatSamples(_ buffers: [AVAudioPCMBuffer]) -> [Float] {
        var output: [Float] = []
        output.reserveCapacity(buffers.reduce(0) { $0 + Int($1.frameLength) })

        for buffer in buffers {
            let frameCount = Int(buffer.frameLength)
            guard frameCount > 0 else { continue }

            switch buffer.format.commonFormat {
            case .pcmFormatFloat32:
                guard let channels = buffer.floatChannelData else { continue }
                output.append(contentsOf: UnsafeBufferPointer(start: channels[0], count: frameCount))
            case .pcmFormatInt16:
                guard let channels = buffer.int16ChannelData else { continue }
                let source = UnsafeBufferPointer(start: channels[0], count: frameCount)
                let scale = Float(Int16.max)
                output.append(contentsOf: source.map { Float($0) / scale })
            default:
                continue
            }
        }

        return output
    }

    private func wrapWithContextCorrectionIfNeeded(
        _ session: any LiveTranscriptionSession,
        identifiers: [String],
        label: String
    ) -> any LiveTranscriptionSession {
        guard !identifiers.isEmpty else { return session }
        return IdentifierCorrectingLiveSession(base: session, identifiers: identifiers, label: label)
    }

    private func applyContextualCorrections(_ transcript: String, identifiers: [String], label: String) -> String {
        let trimmed = transcript.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, !identifiers.isEmpty else { return trimmed }

        let corrected = FuzzyIdentifierMatcher.postProcess(trimmed, identifiers: identifiers)
        if corrected != trimmed {
            DebugLog.write("[ContextBias:\(label)] \(trimmed.prefix(120)) -> \(corrected.prefix(120))")
        }
        return corrected
    }

    private func transcribeWithSpeechAnalyzer(
        buffers: [AVAudioPCMBuffer],
        mode: TranscriberMode,
        locale: Locale
    ) async throws -> String {
        let resolvedLocale = try await localeCoordinator.resolveAndReserveLocale(for: mode, requestedLocale: locale)
        let preparation = makePreparation(mode: mode, locale: resolvedLocale)
        let analyzer = SpeechAnalyzer(
            modules: [preparation.module],
            options: .init(priority: .userInitiated, modelRetention: .lingering)
        )

        let targetFormat = await SpeechAnalyzer.bestAvailableAudioFormat(
            compatibleWith: [preparation.module],
            considering: buffers[0].format
        ) ?? buffers[0].format

        DebugLog.write("[SpeechAnalyzerBatch] starting mode=\(mode) locale=\(resolvedLocale.identifier) buffers=\(buffers.count) format=\(buffers[0].format) target=\(targetFormat)")

        try await analyzer.prepareToAnalyze(in: targetFormat) { [weak self] progress in
            self?.onModelPreparationProgress?(progress)
        }

        var continuationRef: AsyncStream<AnalyzerInput>.Continuation?
        let inputStream = AsyncStream<AnalyzerInput> { continuation in
            continuationRef = continuation
        }

        guard let continuationRef else {
            throw TranscriberError.sessionInitializationFailed
        }

        let startTask = Task<Void, Error> {
            try await analyzer.start(inputSequence: inputStream)
        }
        let collectorTask = preparation.collectorFactory()

        for buffer in buffers {
            continuationRef.yield(AnalyzerInput(buffer: buffer))
        }
        continuationRef.finish()

        do {
            try await analyzer.finalizeAndFinishThroughEndOfInput()
            try await startTask.value
            let transcript = try await collectorTask.value
            if let resolvedTranscript = TranscriptResolutionPolicy.normalizedTranscript(transcript) {
                DebugLog.write("[SpeechAnalyzerBatch] success text=\(resolvedTranscript.prefix(120))")
                return resolvedTranscript
            }
            throw TranscriberError.speechAnalysisFailed("Empty transcript")
        } catch let error as TranscriberError {
            collectorTask.cancel()
            await analyzer.cancelAndFinishNow()
            throw error
        } catch {
            collectorTask.cancel()
            startTask.cancel()
            await analyzer.cancelAndFinishNow()
            throw TranscriberError.speechAnalysisFailed(error.localizedDescription)
        }
    }

    private func makePreparation(mode: TranscriberMode, locale: Locale) -> SpeechPreparation {
        switch mode {
        case .dictationLong:
            let module = DictationTranscriber(
                locale: locale,
                contentHints: [],
                transcriptionOptions: [.punctuation, .etiquetteReplacements],
                reportingOptions: [.volatileResults, .frequentFinalization],
                attributeOptions: []
            )

            return SpeechPreparation(
                module: module,
                collectorFactory: {
                    Task<String, Error> {
                        var assembler = TranscriptAssembler()
                        var resultCount = 0
                        DebugLog.write("[Collector] waiting for dictation results...")
                        for try await result in module.results {
                            resultCount += 1
                            let text = String(result.text.characters)
                            DebugLog.write("[Collector] result #\(resultCount): isFinal=\(result.isFinal) text=\(text.prefix(80))")
                            assembler.consume(
                                text: text,
                                range: result.range,
                                isFinal: result.isFinal
                            )
                        }
                        DebugLog.write("[Collector] done, totalResults=\(resultCount) transcript=\(assembler.transcript.prefix(100))")
                        return assembler.transcript
                    }
                }
            )

        case .speechTranscription:
            let module = SpeechTranscriber(
                locale: locale,
                transcriptionOptions: [.etiquetteReplacements],
                reportingOptions: [.volatileResults, .fastResults],
                attributeOptions: []
            )

            return SpeechPreparation(
                module: module,
                collectorFactory: {
                    Task<String, Error> {
                        var assembler = TranscriptAssembler()
                        for try await result in module.results {
                            assembler.consume(
                                text: String(result.text.characters),
                                range: result.range,
                                isFinal: result.isFinal
                            )
                        }
                        return assembler.transcript
                    }
                }
            )
        }
    }

    private func transcribeWithLegacyRecognizer(
        buffers: [AVAudioPCMBuffer],
        mode: TranscriberMode,
        locale: Locale
    ) async throws -> String {
        guard let recognizer = SFSpeechRecognizer(locale: locale) else {
            throw TranscriberError.legacyRecognizerUnavailable(locale.identifier)
        }

        let request = SFSpeechAudioBufferRecognitionRequest()
        request.shouldReportPartialResults = true
        request.requiresOnDeviceRecognition = true
        request.taskHint = mode == .dictationLong ? .dictation : .unspecified

        let bridge = LegacyRecognitionBridge()
        let transcriptLock = NSLock()
        var latestTranscript = ""
        DebugLog.write("[LegacyRecognizer] starting fallback mode=\(mode) locale=\(locale.identifier) buffers=\(buffers.count)")

        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                bridge.install(continuation)

                let task = recognizer.recognitionTask(with: request) { result, error in
                    if let error {
                        DebugLog.write("[LegacyRecognizer] failed mode=\(mode): \(error.localizedDescription)")
                        transcriptLock.lock()
                        let fallbackTranscript = TranscriptResolutionPolicy.normalizedTranscript(latestTranscript)
                        transcriptLock.unlock()

                        if let fallbackTranscript {
                            bridge.succeed(fallbackTranscript)
                        } else {
                            bridge.fail(TranscriberError.legacyRecognitionFailed(error.localizedDescription))
                        }
                        return
                    }

                    guard let result else { return }
                    let transcript = result.bestTranscription.formattedString.trimmingCharacters(in: .whitespacesAndNewlines)
                    if !transcript.isEmpty {
                        transcriptLock.lock()
                        latestTranscript = transcript
                        transcriptLock.unlock()
                    }
                    DebugLog.write("[LegacyRecognizer] result mode=\(mode) isFinal=\(result.isFinal) text=\(transcript.prefix(80))")
                    guard result.isFinal else { return }

                    transcriptLock.lock()
                    let resolvedTranscript = TranscriptResolutionPolicy.preferredTranscript(
                        primary: transcript,
                        alternative: latestTranscript
                    )
                    transcriptLock.unlock()

                    if let resolvedTranscript {
                        bridge.succeed(resolvedTranscript)
                    } else {
                        bridge.fail(TranscriberError.legacyRecognitionFailed("Empty transcript"))
                    }
                }

                bridge.setTask(task)

                for buffer in buffers {
                    request.append(buffer)
                }
                request.endAudio()
            }
        } onCancel: {
            request.endAudio()
            bridge.cancel()
        }
    }

    private func transcribeWithModal(
        buffers: [AVAudioPCMBuffer],
        locale: Locale,
        env: [String: String],
        prompt: String? = nil
    ) async throws -> String {
        let config = try Self.modalSTTConfig(from: env)
        let samples = Self.convertBuffersToFloatSamples(buffers)
        guard !samples.isEmpty else {
            throw TranscriberError.emptyAudio
        }

        return try await transcribeSamplesWithModal(
            samples: samples,
            locale: locale,
            config: config,
            prompt: prompt
        ).text
    }

    private func transcribeSamplesWithModal(
        samples: [Float],
        locale: Locale,
        config: ModalSTTConfig,
        prompt: String? = nil
    ) async throws -> GroqVerboseTranscriptionResponse {
        guard !samples.isEmpty else {
            throw TranscriberError.emptyAudio
        }

        do {
            let audioEncodingStartedAt = Date()
            let audioData = try makeWavData(fromFloatSamples: samples, sampleRate: 16_000)
            let audioEncodingMilliseconds = Int((Date().timeIntervalSince(audioEncodingStartedAt) * 1_000.0).rounded())
            let boundary = "Boundary-\(UUID().uuidString)"

            var body = Data()
            body.appendUTF8("--\(boundary)\r\n")
            body.appendUTF8("Content-Disposition: form-data; name=\"model\"\r\n\r\n")
            body.appendUTF8("\(config.model)\r\n")

            body.appendUTF8("--\(boundary)\r\n")
            body.appendUTF8("Content-Disposition: form-data; name=\"language\"\r\n\r\n")
            body.appendUTF8("\(locale.language.languageCode?.identifier ?? "en")\r\n")

            body.appendUTF8("--\(boundary)\r\n")
            body.appendUTF8("Content-Disposition: form-data; name=\"temperature\"\r\n\r\n")
            body.appendUTF8("0\r\n")

            if let prompt, !prompt.isEmpty {
                body.appendUTF8("--\(boundary)\r\n")
                body.appendUTF8("Content-Disposition: form-data; name=\"prompt\"\r\n\r\n")
                body.appendUTF8("\(prompt)\r\n")
            }

            body.appendUTF8("--\(boundary)\r\n")
            body.appendUTF8("Content-Disposition: form-data; name=\"file\"; filename=\"capture.wav\"\r\n")
            body.appendUTF8("Content-Type: audio/wav\r\n\r\n")
            body.append(audioData)
            body.appendUTF8("\r\n--\(boundary)--\r\n")

            var request = URLRequest(url: config.endpoint)
            request.httpMethod = "POST"
            request.timeoutInterval = config.timeoutSeconds
            request.setValue("Bearer \(config.apiKey)", forHTTPHeaderField: "Authorization")
            request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
            request.httpBody = body

            let requestStartedAt = Date()
            DebugLog.write("[ModalSTT] uploading \(audioData.count) bytes model=\(config.model)")
            let (data, response) = try await URLSession.shared.data(for: request)
            let requestMilliseconds = Int((Date().timeIntervalSince(requestStartedAt) * 1_000.0).rounded())
            guard let httpResponse = response as? HTTPURLResponse else {
                throw TranscriberError.cloudRecognitionFailed("Invalid HTTP response")
            }
            guard (200..<300).contains(httpResponse.statusCode) else {
                let responseText = String(data: data, encoding: .utf8) ?? "<non-utf8>"
                DebugLog.write("[ModalSTT] HTTP \(httpResponse.statusCode): \(responseText.prefix(300))")
                throw TranscriberError.cloudRecognitionFailed("HTTP \(httpResponse.statusCode)")
            }

            let decoded = try JSONDecoder().decode(GroqVerboseTranscriptionResponse.self, from: data)
            let transcript = decoded.text.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !transcript.isEmpty else {
                throw TranscriberError.cloudRecognitionFailed("Empty transcript")
            }

            let serverDecodeMs = decoded.serverDecodeMilliseconds
            let serverRequestReadMs = decoded.serverRequestReadMilliseconds
            let serverAudioPrepareMs = decoded.serverAudioPrepareMilliseconds
            let serverTotalMs = decoded.serverTotalMilliseconds
            let audioSeconds = decoded.reportedAudioSeconds ?? (Double(samples.count) / 16_000.0)
            let textLength = transcript.count
            DebugLog.write(
                "[ModalSTT] success audioSeconds=\(String(format: "%.2f", audioSeconds)) audioEncodeMs=\(audioEncodingMilliseconds) requestMs=\(requestMilliseconds) serverReadMs=\(serverRequestReadMs.map(String.init) ?? "n/a") serverAudioPrepareMs=\(serverAudioPrepareMs.map(String.init) ?? "n/a") serverDecodeMs=\(serverDecodeMs.map(String.init) ?? "n/a") serverTotalMs=\(serverTotalMs.map(String.init) ?? "n/a") textLength=\(textLength) text=\(transcript.prefix(120))"
            )
            return decoded
        } catch let error as TranscriberError {
            throw error
        } catch {
            throw TranscriberError.cloudRecognitionFailed(error.localizedDescription)
        }
    }

    private func writeBuffers(_ buffers: [AVAudioPCMBuffer], toWavFileAt url: URL) throws {
        guard let firstBuffer = buffers.first else {
            throw TranscriberError.emptyAudio
        }

        let audioFile = try AVAudioFile(
            forWriting: url,
            settings: firstBuffer.format.settings,
            commonFormat: firstBuffer.format.commonFormat,
            interleaved: firstBuffer.format.isInterleaved
        )

        for buffer in buffers {
            try audioFile.write(from: buffer)
        }
    }

    private func writeFloatSamples(_ samples: [Float], sampleRate: Double, toWavFileAt url: URL) throws {
        guard let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: sampleRate, channels: 1, interleaved: false) else {
            throw TranscriberError.cloudRecognitionFailed("Unable to allocate WAV format")
        }
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count)) else {
            throw TranscriberError.cloudRecognitionFailed("Unable to allocate WAV buffer")
        }

        buffer.frameLength = AVAudioFrameCount(samples.count)
        if let channelData = buffer.floatChannelData {
            samples.withUnsafeBufferPointer { source in
                channelData[0].initialize(from: source.baseAddress!, count: samples.count)
            }
        }

        try writeBuffers([buffer], toWavFileAt: url)
    }

    private func makeWavData(fromFloatSamples samples: [Float], sampleRate: Int) throws -> Data {
        let bytesPerSample = MemoryLayout<Int16>.size
        let dataSize = samples.count * bytesPerSample
        let riffChunkSize = 36 + dataSize
        let byteRate = sampleRate * bytesPerSample
        let blockAlign = UInt16(bytesPerSample)

        var data = Data(capacity: 44 + dataSize)
        data.appendUTF8("RIFF")
        data.appendLittleEndian(UInt32(riffChunkSize))
        data.appendUTF8("WAVE")
        data.appendUTF8("fmt ")
        data.appendLittleEndian(UInt32(16))
        data.appendLittleEndian(UInt16(1))
        data.appendLittleEndian(UInt16(1))
        data.appendLittleEndian(UInt32(sampleRate))
        data.appendLittleEndian(UInt32(byteRate))
        data.appendLittleEndian(blockAlign)
        data.appendLittleEndian(UInt16(16))
        data.appendUTF8("data")
        data.appendLittleEndian(UInt32(dataSize))

        for sample in samples {
            let scaled = (max(-1.0, min(1.0, sample)) * Float(Int16.max)).rounded()
            let pcmSample = Int16(max(Float(Int16.min), min(Float(Int16.max), scaled)))
            data.appendLittleEndian(pcmSample)
        }

        return data
    }
}

private struct SpeechPreparation {
    let module: any SpeechModule
    let collectorFactory: @Sendable () -> Task<String, Error>
}

private actor SpeechLocaleCoordinator {
    private var reservedLocaleIdentifiers: Set<String> = []

    func resolveAndReserveLocale(for mode: TranscriberMode, requestedLocale: Locale) async throws -> Locale {
        let resolvedLocale = try await resolveSupportedLocale(for: mode, requestedLocale: requestedLocale)
        await reserveIfPossible(resolvedLocale)
        return resolvedLocale
    }

    private func resolveSupportedLocale(for mode: TranscriberMode, requestedLocale: Locale) async throws -> Locale {
        if let supportedLocale = await supportedLocale(for: mode, equivalentTo: requestedLocale) {
            return supportedLocale
        }

        if let languageCode = requestedLocale.language.languageCode?.identifier,
           let supportedLocale = await supportedLocale(for: mode, equivalentTo: Locale(identifier: languageCode)) {
            return supportedLocale
        }

        throw TranscriberError.unsupportedLocale(requestedLocale.identifier)
    }

    private func reserveIfPossible(_ locale: Locale) async {
        let localeIdentifier = locale.identifier
        if reservedLocaleIdentifiers.contains(localeIdentifier) {
            return
        }

        let alreadyReserved = await AssetInventory.reservedLocales.contains { $0.identifier == localeIdentifier }
        if alreadyReserved {
            reservedLocaleIdentifiers.insert(localeIdentifier)
            return
        }

        do {
            let didReserve = try await AssetInventory.reserve(locale: locale)
            if didReserve {
                reservedLocaleIdentifiers.insert(localeIdentifier)
                return
            }
        } catch {
            return
        }

        let reservedAfterAttempt = await AssetInventory.reservedLocales.contains { $0.identifier == localeIdentifier }
        if reservedAfterAttempt {
            reservedLocaleIdentifiers.insert(localeIdentifier)
        }
    }

    private func supportedLocale(for mode: TranscriberMode, equivalentTo locale: Locale) async -> Locale? {
        switch mode {
        case .dictationLong:
            return await DictationTranscriber.supportedLocale(equivalentTo: locale)
        case .speechTranscription:
            return await SpeechTranscriber.supportedLocale(equivalentTo: locale)
        }
    }
}
