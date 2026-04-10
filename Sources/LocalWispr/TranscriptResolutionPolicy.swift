import Foundation

enum TranscriptResolutionPolicy {
    static func normalizedTranscript(_ text: String?) -> String? {
        guard let text else { return nil }
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        return containsSubstantiveContent(trimmed) ? trimmed : nil
    }

    static func resolvedTranscript(primary: String?, fallback: String?) -> String? {
        normalizedTranscript(primary) ?? normalizedTranscript(fallback)
    }

    static func preferredTranscript(primary: String?, alternative: String?) -> String? {
        let primaryNormalized = normalizedTranscript(primary)
        let alternativeNormalized = normalizedTranscript(alternative)

        switch (primaryNormalized, alternativeNormalized) {
        case (nil, nil):
            return nil
        case let (value?, nil), let (nil, value?):
            return value
        case let (primaryValue?, alternativeValue?):
            return transcriptScore(alternativeValue) > transcriptScore(primaryValue) ? alternativeValue : primaryValue
        }
    }

    static func shouldAttemptBatchVerification(
        liveTranscript: String?,
        recordingDurationMilliseconds: Int?
    ) -> Bool {
        guard normalizedTranscript(liveTranscript) == nil else {
            return false
        }

        _ = recordingDurationMilliseconds
        return true
    }

    static func fallbackModes(after preferredMode: TranscriberMode) -> [TranscriberMode] {
        switch preferredMode {
        case .dictationLong:
            return [.dictationLong, .speechTranscription]
        case .speechTranscription:
            return [.speechTranscription]
        }
    }

    private static func transcriptScore(_ text: String) -> Int {
        let words = text.split(whereSeparator: \.isWhitespace).count
        return (words * 32) + text.count
    }

    private static func containsSubstantiveContent(_ text: String) -> Bool {
        text.unicodeScalars.contains { scalar in
            CharacterSet.alphanumerics.contains(scalar)
        }
    }
}
