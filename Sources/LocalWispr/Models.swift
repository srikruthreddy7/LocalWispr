@preconcurrency import AVFAudio
import Foundation

extension AVAudioPCMBuffer: @unchecked @retroactive Sendable {}

public enum DictationState: Equatable {
    case idle
    case listening
    case finalizingTranscript
    case cleaning
    case inserting
    case noSpeech
    case error(String)
}

public enum TranscriberMode: String, CaseIterable, Identifiable, Sendable, Codable {
    case dictationLong
    case speechTranscription

    public var id: String { rawValue }

    public var title: String {
        switch self {
        case .dictationLong:
            return "Dictation"
        case .speechTranscription:
            return "Transcription"
        }
    }
}

public enum PipelineStage: Sendable {
    case cleaning
    case inserting
}

public struct PipelineLatency: Sendable, Equatable, Codable {
    public let stopToTranscriptMilliseconds: Int
    public let cleanupMilliseconds: Int
    public let insertionMilliseconds: Int
    public let totalStopToInsertMilliseconds: Int
    public let recordingDurationMilliseconds: Int?

    public init(
        stopToTranscriptMilliseconds: Int,
        cleanupMilliseconds: Int,
        insertionMilliseconds: Int,
        recordingDurationMilliseconds: Int? = nil
    ) {
        self.stopToTranscriptMilliseconds = max(0, stopToTranscriptMilliseconds)
        self.cleanupMilliseconds = max(0, cleanupMilliseconds)
        self.insertionMilliseconds = max(0, insertionMilliseconds)
        self.totalStopToInsertMilliseconds = max(0, stopToTranscriptMilliseconds + cleanupMilliseconds + insertionMilliseconds)
        if let recordingDurationMilliseconds {
            self.recordingDurationMilliseconds = max(0, recordingDurationMilliseconds)
        } else {
            self.recordingDurationMilliseconds = nil
        }
    }

    public var formattedSummary: String {
        let totalSeconds = Double(totalStopToInsertMilliseconds) / 1000.0
        return String(
            format: "Stop latency: %.3fs (T:%dms C:%dms I:%dms)",
            totalSeconds,
            stopToTranscriptMilliseconds,
            cleanupMilliseconds,
            insertionMilliseconds
        )
    }
}

public struct StopLatencyStats: Sendable, Equatable, Codable {
    public let sampleCount: Int
    public let averageMilliseconds: Int
    public let p50Milliseconds: Int
    public let p95Milliseconds: Int

    public var formattedSummary: String {
        "avg \(averageMilliseconds)ms | p50 \(p50Milliseconds)ms | p95 \(p95Milliseconds)ms (\(sampleCount) runs)"
    }

    static func fromRecentTotals(_ totals: [Int], limit: Int = 20) -> StopLatencyStats? {
        let recent = Array(totals.filter { $0 > 0 }.prefix(limit))
        guard !recent.isEmpty else { return nil }

        let sorted = recent.sorted()
        let average = Int((Double(recent.reduce(0, +)) / Double(recent.count)).rounded())
        let p50 = percentile(50, in: sorted)
        let p95 = percentile(95, in: sorted)

        return StopLatencyStats(
            sampleCount: recent.count,
            averageMilliseconds: average,
            p50Milliseconds: p50,
            p95Milliseconds: p95
        )
    }

    private static func percentile(_ value: Int, in sortedValues: [Int]) -> Int {
        guard !sortedValues.isEmpty else { return 0 }
        let clampedValue = min(100, max(0, value))
        let rank = Int(ceil(Double(clampedValue) / 100.0 * Double(sortedValues.count))) - 1
        let index = min(max(rank, 0), sortedValues.count - 1)
        return sortedValues[index]
    }
}

public enum TranscriptResolutionSource: String, Sendable, Equatable, Codable {
    case liveFinalization = "live-final"
    case batchFallback = "batch-fallback"
    case unavailable = "unresolved"
}

public struct StopPathDetails: Sendable, Equatable, Codable {
    public let source: TranscriptResolutionSource
    public let bufferCount: Int
    public let drainMilliseconds: Int
    public let liveFinalizationMilliseconds: Int?
    public let batchFallbackMilliseconds: Int?
    public let liveFailureDescription: String?

    public var formattedSummary: String {
        var parts: [String] = [
            "Path \(source.rawValue)",
            "buffers \(bufferCount)",
            "drain \(drainMilliseconds)ms"
        ]

        if let liveFinalizationMilliseconds {
            parts.append("live \(liveFinalizationMilliseconds)ms")
        }

        if let batchFallbackMilliseconds {
            parts.append("batch \(batchFallbackMilliseconds)ms")
        }

        if liveFailureDescription != nil {
            parts.append("live-failed")
        }

        return parts.joined(separator: " | ")
    }
}

public enum PipelineResult: Sendable, Equatable {
    case noSpeech(latency: PipelineLatency)
    case inserted(raw: String, cleaned: String, warning: String?, latency: PipelineLatency)
    case failed(raw: String?, cleaned: String?, error: String, latency: PipelineLatency)
}

public enum TranscriptRecordOutcome: String, Sendable, Equatable, Codable {
    case inserted
    case insertionFailed
    case transcriptionFailed
}

public enum ControlPanelSection: String, CaseIterable, Identifiable, Sendable, Codable {
    case dashboard
    case history
    case settings

    public var id: String { rawValue }

    public var title: String {
        switch self {
        case .dashboard:
            return "Dashboard"
        case .history:
            return "History"
        case .settings:
            return "Settings"
        }
    }

    public var subtitle: String {
        switch self {
        case .dashboard:
            return "Live controls"
        case .history:
            return "Saved sessions"
        case .settings:
            return "Configuration"
        }
    }

    public var systemImageName: String {
        switch self {
        case .dashboard:
            return "square.grid.2x2"
        case .history:
            return "clock.arrow.circlepath"
        case .settings:
            return "gearshape"
        }
    }
}

public struct TranscriptRecord: Identifiable, Sendable, Equatable, Codable {
    public let id: UUID
    public let createdAt: Date
    public let transcriberMode: TranscriberMode
    public let localeIdentifier: String
    public let rawText: String
    public let cleanedText: String
    public let outcome: TranscriptRecordOutcome
    public let cleanupWarning: String?
    public let errorMessage: String?
    public let latency: PipelineLatency?
    public let stopPath: StopPathDetails?

    public init(
        id: UUID = UUID(),
        createdAt: Date = Date(),
        transcriberMode: TranscriberMode,
        localeIdentifier: String,
        rawText: String,
        cleanedText: String,
        outcome: TranscriptRecordOutcome,
        cleanupWarning: String? = nil,
        errorMessage: String? = nil,
        latency: PipelineLatency? = nil,
        stopPath: StopPathDetails? = nil
    ) {
        self.id = id
        self.createdAt = createdAt
        self.transcriberMode = transcriberMode
        self.localeIdentifier = localeIdentifier
        self.rawText = rawText
        self.cleanedText = cleanedText
        self.outcome = outcome
        self.cleanupWarning = cleanupWarning?.trimmingCharacters(in: .whitespacesAndNewlines)
        self.errorMessage = errorMessage?.trimmingCharacters(in: .whitespacesAndNewlines)
        self.latency = latency
        self.stopPath = stopPath
    }

    public var hasDisplayableText: Bool {
        !rawText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            || !cleanedText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    public var preferredDisplayText: String {
        let cleaned = cleanedText.trimmingCharacters(in: .whitespacesAndNewlines)
        if !cleaned.isEmpty {
            return cleaned
        }

        let raw = rawText.trimmingCharacters(in: .whitespacesAndNewlines)
        if !raw.isEmpty {
            return raw
        }

        let error = errorMessage?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        if !error.isEmpty {
            return error
        }

        return statusSummary
    }

    public var statusSummary: String {
        switch outcome {
        case .inserted:
            return "Completed"
        case .insertionFailed:
            return "Insertion Failed"
        case .transcriptionFailed:
            return "Transcription Failed"
        }
    }
}

public enum CleanerAvailability: Sendable, Equatable {
    case available
    case unavailable(String)
}

public protocol HotkeyMonitoring: AnyObject {
    var onToggleRequested: (@MainActor () -> Void)? { get set }
    func start() throws
    func stop()
}

public protocol AudioCapturing: AnyObject {
    var outputAudioFormat: AVAudioFormat { get }
    var onBufferCaptured: (@Sendable (AVAudioPCMBuffer) -> Void)? { get set }
    func start() throws
    func stopAndDrain() throws -> [AVAudioPCMBuffer]
}

public protocol Transcribing: Sendable {
    func transcribe(buffers: [AVAudioPCMBuffer], mode: TranscriberMode, locale: Locale) async throws -> String
}

public protocol LiveTranscriptionSession: Sendable {
    func append(_ buffer: AVAudioPCMBuffer)
    func finish() async throws -> String
    func cancel() async
}

public protocol LiveTranscribing: Sendable {
    func startSession(
        mode: TranscriberMode,
        locale: Locale,
        audioFormat: AVAudioFormat
    ) async throws -> any LiveTranscriptionSession
}

public protocol SpeechModelWarming: Sendable {
    func prewarm(mode: TranscriberMode, locale: Locale, audioFormat: AVAudioFormat) async
}

public protocol Cleaning: Sendable {
    var availability: CleanerAvailability { get }
    func clean(_ rawText: String) async throws -> String
}

public protocol Inserting: Sendable {
    func insertAtCursor(_ text: String) async throws
}
