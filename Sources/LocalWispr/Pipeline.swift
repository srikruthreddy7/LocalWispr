import Foundation
import OSLog

public actor Pipeline {
    private static let logger = Logger(subsystem: "LocalWispr", category: "Latency")

    private let cleaner: any Cleaning
    private let inserter: any Inserting

    public init(
        cleaner: any Cleaning = AdaptiveTextCleaner(),
        inserter: any Inserting = TextInserter()
    ) {
        self.cleaner = cleaner
        self.inserter = inserter
    }

    public func process(
        rawText: String,
        stopToTranscriptMilliseconds: Int,
        recordingDurationMilliseconds: Int?,
        stageHandler: (@Sendable (PipelineStage) -> Void)? = nil
    ) async -> PipelineResult {
        let clock = ContinuousClock()
        var cleanupDuration: Duration = .zero
        var insertionDuration: Duration = .zero

        let normalizedRaw = rawText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !normalizedRaw.isEmpty, Self.containsSubstantiveContent(normalizedRaw) else {
            let latency = Self.makeLatency(
                stopToTranscriptMilliseconds: stopToTranscriptMilliseconds,
                cleanupDuration: cleanupDuration,
                insertionDuration: insertionDuration,
                recordingDurationMilliseconds: recordingDurationMilliseconds
            )
            Self.logLatency(latency, outcome: "no_speech")
            return .noSpeech(latency: latency)
        }

        var cleanedText = normalizedRaw
        var warning: String?

        switch cleaner.availability {
        case .available:
            stageHandler?(.cleaning)
            let cleanupStartedAt = clock.now
            do {
                let candidate = try await cleaner.clean(normalizedRaw).trimmingCharacters(in: .whitespacesAndNewlines)
                if !candidate.isEmpty {
                    cleanedText = candidate
                }
                cleanupDuration = cleanupStartedAt.duration(to: clock.now)
            } catch let error as TextCleanerError {
                warning = error.errorDescription ?? "Cleanup failed; inserting raw transcript instead."
                cleanedText = normalizedRaw
                cleanupDuration = cleanupStartedAt.duration(to: clock.now)
            } catch {
                warning = "Cleanup failed; inserting raw transcript instead."
                cleanedText = normalizedRaw
                cleanupDuration = cleanupStartedAt.duration(to: clock.now)
            }
        case .unavailable(let reason):
            warning = "Cleanup unavailable (\(reason)); inserted raw transcript."
        }

        stageHandler?(.inserting)
        let insertionStartedAt = clock.now

        do {
            try await inserter.insertAtCursor(cleanedText)
            insertionDuration = insertionStartedAt.duration(to: clock.now)
        } catch {
            insertionDuration = insertionStartedAt.duration(to: clock.now)
            let latency = Self.makeLatency(
                stopToTranscriptMilliseconds: stopToTranscriptMilliseconds,
                cleanupDuration: cleanupDuration,
                insertionDuration: insertionDuration,
                recordingDurationMilliseconds: recordingDurationMilliseconds
            )
            Self.logLatency(latency, outcome: "insertion_failed")
            return .failed(
                raw: normalizedRaw,
                cleaned: cleanedText,
                error: "Text insertion failed: \(error.localizedDescription)",
                latency: latency
            )
        }

        let latency = Self.makeLatency(
            stopToTranscriptMilliseconds: stopToTranscriptMilliseconds,
            cleanupDuration: cleanupDuration,
            insertionDuration: insertionDuration,
            recordingDurationMilliseconds: recordingDurationMilliseconds
        )
        Self.logLatency(latency, outcome: "inserted")
        return .inserted(raw: normalizedRaw, cleaned: cleanedText, warning: warning, latency: latency)
    }

    private static func makeLatency(
        stopToTranscriptMilliseconds: Int,
        cleanupDuration: Duration,
        insertionDuration: Duration,
        recordingDurationMilliseconds: Int?
    ) -> PipelineLatency {
        PipelineLatency(
            stopToTranscriptMilliseconds: stopToTranscriptMilliseconds,
            cleanupMilliseconds: durationToMilliseconds(cleanupDuration),
            insertionMilliseconds: durationToMilliseconds(insertionDuration),
            recordingDurationMilliseconds: recordingDurationMilliseconds
        )
    }

    private static func durationToMilliseconds(_ duration: Duration) -> Int {
        let components = duration.components
        let milliseconds = (Double(components.seconds) * 1_000.0) + (Double(components.attoseconds) / 1_000_000_000_000_000.0)
        return max(0, Int(milliseconds.rounded()))
    }

    private static func containsSubstantiveContent(_ text: String) -> Bool {
        text.unicodeScalars.contains { scalar in
            CharacterSet.alphanumerics.contains(scalar)
        }
    }

    private static func logLatency(_ latency: PipelineLatency, outcome: StaticString) {
        if let recordingDurationMilliseconds = latency.recordingDurationMilliseconds {
            logger.info(
                "Pipeline \(outcome, privacy: .public) recording=\(recordingDurationMilliseconds, privacy: .public)ms stop_total=\(latency.totalStopToInsertMilliseconds, privacy: .public)ms stop_to_transcript=\(latency.stopToTranscriptMilliseconds, privacy: .public)ms cleanup=\(latency.cleanupMilliseconds, privacy: .public)ms insert=\(latency.insertionMilliseconds, privacy: .public)ms"
            )
        } else {
            logger.info(
                "Pipeline \(outcome, privacy: .public) stop_total=\(latency.totalStopToInsertMilliseconds, privacy: .public)ms stop_to_transcript=\(latency.stopToTranscriptMilliseconds, privacy: .public)ms cleanup=\(latency.cleanupMilliseconds, privacy: .public)ms insert=\(latency.insertionMilliseconds, privacy: .public)ms"
            )
        }
    }
}
