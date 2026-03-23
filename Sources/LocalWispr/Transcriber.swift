@preconcurrency import AVFAudio
import Foundation
import Speech

public enum TranscriberError: LocalizedError {
    case emptyAudio
    case sessionInitializationFailed
    case unsupportedLocale(String)
    case localeAllocationFailed(String)

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
        }
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

    func append(_ buffer: AVAudioPCMBuffer) {
        let state = withStateLock {
            (continuation: continuation, isFinished: finished)
        }

        guard !state.isFinished else { return }
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

public final class Transcriber: @unchecked Sendable, Transcribing, LiveTranscribing, SpeechModelWarming {
    public typealias ModelProgressHandler = @Sendable (Progress) -> Void

    public var onModelPreparationProgress: ModelProgressHandler?

    private let localeCoordinator = SpeechLocaleCoordinator()

    public init() {}

    public func prewarm(mode: TranscriberMode, locale: Locale, audioFormat: AVAudioFormat) async {
        do {
            let resolvedLocale = try await localeCoordinator.resolveAndReserveLocale(for: mode, requestedLocale: locale)
            let preparation = makePreparation(mode: mode, locale: resolvedLocale)
            let analyzer = SpeechAnalyzer(
                modules: [preparation.module],
                options: .init(priority: .userInitiated, modelRetention: .lingering)
            )
            let targetFormat = await SpeechAnalyzer.bestAvailableAudioFormat(
                compatibleWith: [preparation.module],
                considering: audioFormat
            ) ?? audioFormat
            try await analyzer.prepareToAnalyze(in: targetFormat) { [weak self] progress in
                self?.onModelPreparationProgress?(progress)
            }
        } catch {
            // Warmup is best-effort.
        }
    }

    public func startSession(
        mode: TranscriberMode,
        locale: Locale,
        audioFormat: AVAudioFormat
    ) async throws -> any LiveTranscriptionSession {
        let resolvedLocale = try await localeCoordinator.resolveAndReserveLocale(for: mode, requestedLocale: locale)
        let preparation = makePreparation(mode: mode, locale: resolvedLocale)
        let analyzer = SpeechAnalyzer(
            modules: [preparation.module],
            options: .init(priority: .userInitiated, modelRetention: .lingering)
        )

        let targetFormat = await SpeechAnalyzer.bestAvailableAudioFormat(
            compatibleWith: [preparation.module],
            considering: audioFormat
        ) ?? audioFormat

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

        return InternalLiveTranscriptionSession(
            analyzer: analyzer,
            continuation: continuationRef,
            startTask: startTask,
            collectorTask: collectorTask
        )
    }

    public func transcribe(buffers: [AVAudioPCMBuffer], mode: TranscriberMode, locale: Locale) async throws -> String {
        guard !buffers.isEmpty else {
            throw TranscriberError.emptyAudio
        }

        let session = try await startSession(mode: mode, locale: locale, audioFormat: buffers[0].format)
        for buffer in buffers {
            session.append(buffer)
        }
        return try await session.finish()
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
