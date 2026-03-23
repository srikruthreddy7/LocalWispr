import Foundation
import FoundationModels
import OSLog

public enum TextCleanerError: LocalizedError {
    case unavailable(String)
    case exceededContextWindowSize
    case assetsUnavailable
    case guardrailViolation
    case generationFailure(String)

    public var errorDescription: String? {
        switch self {
        case .unavailable(let reason):
            return "LLM cleanup unavailable: \(reason)"
        case .exceededContextWindowSize:
            return "Dictation is too long for cleanup context window."
        case .assetsUnavailable:
            return "Apple Intelligence assets are not available yet."
        case .guardrailViolation:
            return "Cleanup model rejected this content under guardrails."
        case .generationFailure(let details):
            return "Cleanup failed: \(details)"
        }
    }
}

public final class TextCleaner: @unchecked Sendable, Cleaning {
    private static let logger = Logger(subsystem: "LocalWispr", category: "Cleanup")
    private static let codeFenceRegex = try! NSRegularExpression(
        pattern: #"(?s)^\s*```[a-zA-Z0-9_-]*\s*\n(.*)\n```\s*$"#
    )

    private static let leadingPreambleRegexes: [NSRegularExpression] = [
        try! NSRegularExpression(
            pattern: #"(?is)^\s*(?:sure[!.,]?\s*)?(?:here(?:'s| is)\s+(?:the\s+)?(?:cleaned|revised|edited|polished|formatted)\s+text)\s*:\s*"#
        ),
        try! NSRegularExpression(
            pattern: #"(?is)^\s*here(?:'s| is)\s+your\s+(?:cleaned|revised|edited|polished|formatted)?\s*text\s*:\s*"#
        ),
        try! NSRegularExpression(
            pattern: #"(?is)^\s*(?:cleaned|revised|edited|polished|formatted)\s+text\s*:\s*"#
        ),
        try! NSRegularExpression(
            pattern: #"(?is)^\s*output\s*:\s*"#
        ),
    ]

    private static let bulletLineRegex = try! NSRegularExpression(
        pattern: #"(?m)^\s*(?:[-*•]|\d+\.)\s+\S"#
    )

    private static let ordinalMarkerRegex = try! NSRegularExpression(
        pattern: #"(?is)\b(?:the\s+)?(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\b(?:\s+one)?\s*(?:would be|is|:)?\s*"#
    )
    private static let fillerRegexes: [NSRegularExpression] = [
        try! NSRegularExpression(pattern: #"(?is)\b(?:um+|uh+|erm|hmm+|mm-hmm)\b"#),
        try! NSRegularExpression(pattern: #"(?is)\b(?:you know|i mean)\b"#),
        try! NSRegularExpression(pattern: #"(?is)\b(?:basically|literally)\b"#),
    ]
    private static let leadingConversationFillerRegex = try! NSRegularExpression(
        pattern: #"(?is)^\s*(?:(?:oh|hey|yeah|alright|all right)\b[\s,.-]*)+"#
    )
    private static let generativeDirectiveRegex = try! NSRegularExpression(
        pattern: #"(?is)\b(?:rewrite|rephrase|reword|formalize|formalise|polish|summarize|summarise|draft|compose|write\s+an?\s+email|email\s+to|subject\s+line|headline|tweet|linkedin|markdown|json|table|sql|code\s+block)\b"#
    )
    private static let instructions = """
    You clean dictation into polished written text.
    Preserve meaning.
    Remove filler words when they are verbal fillers.
    Fix grammar, punctuation, and capitalization.
    Follow explicit formatting instructions in the transcription.
    Convert spoken ordinal list cues into actual list items when a list is requested.
    Return only the cleaned text.
    Never add commentary, labels, or markdown fences.
    """

    private let model: SystemLanguageModel
    private let warmupSession: LanguageModelSession

    public init() {
        model = SystemLanguageModel(useCase: .general, guardrails: .permissiveContentTransformations)

        warmupSession = Self.makeSession(with: model)
        warmupSession.prewarm(promptPrefix: Prompt("Raw transcription:"))
    }

    public var availability: CleanerAvailability {
        switch SystemLanguageModel.default.availability {
        case .available:
            return .available
        case .unavailable(let reason):
            return .unavailable(unavailableReasonDescription(reason))
        }
    }

    public func clean(_ rawText: String) async throws -> String {
        let trimmed = rawText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            return ""
        }

        if !Self.shouldUseLanguageModel(for: trimmed) {
            let cleanedText = Self.fastClean(rawTranscription: trimmed)
            Self.logger.info(
                "Cleanup strategy=fast chars_in=\(trimmed.count, privacy: .public) chars_out=\(cleanedText.count, privacy: .public)"
            )
            return cleanedText
        }

        guard case .available = SystemLanguageModel.default.availability else {
            if case .unavailable(let reason) = SystemLanguageModel.default.availability {
                throw TextCleanerError.unavailable(unavailableReasonDescription(reason))
            }
            throw TextCleanerError.unavailable("Unknown availability state")
        }

        let prompt = """
        Raw transcription:
        \(trimmed)
        """
        let generationOptions = Self.generationOptions(for: trimmed)
        let session = Self.makeSession(with: model)

        do {
            let response = try await session.respond(to: prompt, options: generationOptions)
            let sanitized = Self.sanitizeModelOutput(response.content)
            let cleanedText = Self.applyFormattingDirectives(rawTranscription: trimmed, cleanedOutput: sanitized)
            Self.logger.info(
                "Cleanup strategy=model chars_in=\(trimmed.count, privacy: .public) chars_out=\(cleanedText.count, privacy: .public) max_tokens=\(generationOptions.maximumResponseTokens ?? -1, privacy: .public)"
            )
            return cleanedText
        } catch let error as LanguageModelSession.GenerationError {
            throw Self.mapGenerationError(error)
        } catch {
            throw TextCleanerError.generationFailure(error.localizedDescription)
        }
    }

    private static func makeSession(with model: SystemLanguageModel) -> LanguageModelSession {
        LanguageModelSession(
            model: model,
            instructions: instructions
        )
    }

    private static func generationOptions(for rawText: String) -> GenerationOptions {
        let estimatedTokens = max(16, Int(ceil(Double(rawText.count) / 4.0)))
        let maximumResponseTokens = min(72, max(24, estimatedTokens + 8))
        return GenerationOptions(
            sampling: .greedy,
            temperature: 0,
            maximumResponseTokens: maximumResponseTokens
        )
    }

    static func mapGenerationError(_ error: LanguageModelSession.GenerationError) -> TextCleanerError {
        switch error {
        case .exceededContextWindowSize:
            return .exceededContextWindowSize
        case .assetsUnavailable:
            return .assetsUnavailable
        case .guardrailViolation:
            return .guardrailViolation
        default:
            return .generationFailure(error.localizedDescription)
        }
    }

    static func sanitizeModelOutput(_ rawOutput: String) -> String {
        var text = rawOutput.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return "" }

        let fullRange = NSRange(text.startIndex..<text.endIndex, in: text)
        if let match = codeFenceRegex.firstMatch(in: text, options: [], range: fullRange),
           let contentRange = Range(match.range(at: 1), in: text) {
            text = String(text[contentRange]).trimmingCharacters(in: .whitespacesAndNewlines)
        }

        var didStrip = true
        while didStrip, !text.isEmpty {
            didStrip = false
            let range = NSRange(text.startIndex..<text.endIndex, in: text)
            for regex in leadingPreambleRegexes {
                guard let match = regex.firstMatch(in: text, options: [], range: range),
                      match.range.location == 0,
                      let swiftRange = Range(match.range, in: text) else {
                    continue
                }

                text.removeSubrange(swiftRange)
                text = text.trimmingCharacters(in: .whitespacesAndNewlines)
                didStrip = true
                break
            }
        }

        return text
    }

    static func shouldUseLanguageModel(for rawTranscription: String) -> Bool {
        let normalized = rawTranscription.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !normalized.isEmpty else { return false }
        guard !containsBulletRequest(normalized) else { return false }

        let range = NSRange(normalized.startIndex..<normalized.endIndex, in: normalized)
        return generativeDirectiveRegex.firstMatch(in: normalized, options: [], range: range) != nil
    }

    static func fastClean(rawTranscription: String) -> String {
        var text = rawTranscription.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return "" }

        let fullRange = NSRange(text.startIndex..<text.endIndex, in: text)
        text = leadingConversationFillerRegex.stringByReplacingMatches(in: text, options: [], range: fullRange, withTemplate: "")

        for regex in fillerRegexes {
            let range = NSRange(text.startIndex..<text.endIndex, in: text)
            text = regex.stringByReplacingMatches(in: text, options: [], range: range, withTemplate: "")
        }

        text = text.replacingOccurrences(of: #"\s+([,.;:!?])"#, with: "$1", options: .regularExpression)
        text = text.replacingOccurrences(of: #"([,.;:!?])([^\s])"#, with: "$1 $2", options: .regularExpression)
        text = text.replacingOccurrences(of: #"\s{2,}"#, with: " ", options: .regularExpression)
        text = text.replacingOccurrences(of: #"(?:,\s*){2,}"#, with: ", ", options: .regularExpression)
        text = text.replacingOccurrences(of: #"^\s*[,.;:!?-]+\s*"#, with: "", options: .regularExpression)
        text = text.replacingOccurrences(of: #"\s+([’'])"#, with: "$1", options: .regularExpression)
        text = text.trimmingCharacters(in: .whitespacesAndNewlines)

        guard !text.isEmpty else { return "" }

        text = capitalizeSentences(in: text)

        if !isBulletList(text), !hasTerminalPunctuation(text) {
            text += looksLikeQuestion(text) ? "?" : "."
        }

        return applyFormattingDirectives(rawTranscription: rawTranscription, cleanedOutput: text)
    }

    static func applyFormattingDirectives(rawTranscription: String, cleanedOutput: String) -> String {
        let cleaned = cleanedOutput.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !cleaned.isEmpty else { return "" }

        guard containsBulletRequest(rawTranscription) else {
            return cleaned
        }

        if isBulletList(cleaned) {
            return cleaned
        }

        let rawOrdinalItems = extractOrdinalBulletItems(from: rawTranscription)
        let cleanedOrdinalItems = extractOrdinalBulletItems(from: cleaned)
        let ordinalItems: [String]
        if !cleanedOrdinalItems.isEmpty, cleanedOrdinalItems.count >= rawOrdinalItems.count {
            ordinalItems = cleanedOrdinalItems
        } else {
            ordinalItems = rawOrdinalItems
        }
        if !ordinalItems.isEmpty {
            return ordinalItems.map { "- \($0)" }.joined(separator: "\n")
        }

        return cleaned
    }

    private static func containsBulletRequest(_ text: String) -> Bool {
        let normalized = text.lowercased()
        return normalized.contains("bullet point")
            || normalized.contains("bullet points")
            || normalized.contains("bulleted")
            || normalized.contains("bullet list")
            || normalized.contains("put this in bullets")
    }

    private static func isBulletList(_ text: String) -> Bool {
        let range = NSRange(text.startIndex..<text.endIndex, in: text)
        return bulletLineRegex.firstMatch(in: text, options: [], range: range) != nil
    }

    private static func extractOrdinalBulletItems(from text: String) -> [String] {
        let range = NSRange(text.startIndex..<text.endIndex, in: text)
        let matches = ordinalMarkerRegex.matches(in: text, options: [], range: range)
        guard !matches.isEmpty else { return [] }

        var items: [String] = []
        for (index, match) in matches.enumerated() {
            let segmentStart = match.range.location + match.range.length
            let segmentEnd: Int
            if index + 1 < matches.count {
                segmentEnd = matches[index + 1].range.location
            } else {
                segmentEnd = range.location + range.length
            }

            guard segmentEnd > segmentStart else { continue }
            let segmentRange = NSRange(location: segmentStart, length: segmentEnd - segmentStart)
            guard let swiftRange = Range(segmentRange, in: text) else { continue }

            var item = String(text[swiftRange])
            item = item.replacingOccurrences(
                of: #"(?is)^[\s\.,;:!\-]*(?:and\s+)?"#,
                with: "",
                options: .regularExpression
            )
            item = item.replacingOccurrences(
                of: #"(?is)(?:\s*,?\s*(?:and|then)(?:\s+the)?\s*)+$"#,
                with: "",
                options: .regularExpression
            )
            item = item.replacingOccurrences(
                of: #"(?is)[\s\.,;:!\-]+$"#,
                with: "",
                options: .regularExpression
            )

            if item.lowercased().hasPrefix("the ") {
                item.removeFirst(4)
            }

            let trimmed = item.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { continue }

            if let first = trimmed.first {
                items.append(String(first).uppercased() + trimmed.dropFirst())
            } else {
                items.append(trimmed)
            }
        }

        return items
    }

    private static func capitalizeSentences(in text: String) -> String {
        var output = ""
        var shouldCapitalizeNextLetter = true

        for character in text {
            if shouldCapitalizeNextLetter, character.isLetter {
                output.append(contentsOf: String(character).uppercased())
                shouldCapitalizeNextLetter = false
                continue
            }

            output.append(character)

            if ".!?".contains(character) {
                shouldCapitalizeNextLetter = true
            }
        }

        return output
    }

    private static func hasTerminalPunctuation(_ text: String) -> Bool {
        guard let last = text.last else { return false }
        return ".!?".contains(last)
    }

    private static func looksLikeQuestion(_ text: String) -> Bool {
        let normalized = text
            .lowercased()
            .replacingOccurrences(
                of: #"^\s*(?:(?:okay|ok|so)\b[\s,.-]*)+"#,
                with: "",
                options: .regularExpression
            )
        return normalized.hasPrefix("who ")
            || normalized.hasPrefix("what ")
            || normalized.hasPrefix("when ")
            || normalized.hasPrefix("where ")
            || normalized.hasPrefix("why ")
            || normalized.hasPrefix("how ")
            || normalized.hasPrefix("can ")
            || normalized.hasPrefix("could ")
            || normalized.hasPrefix("would ")
            || normalized.hasPrefix("should ")
            || normalized.hasPrefix("do ")
            || normalized.hasPrefix("does ")
            || normalized.hasPrefix("did ")
            || normalized.hasPrefix("is ")
            || normalized.hasPrefix("are ")
            || normalized.hasPrefix("will ")
    }

    private func unavailableReasonDescription(_ reason: SystemLanguageModel.Availability.UnavailableReason) -> String {
        switch reason {
        case .deviceNotEligible:
            return "Device not eligible for Apple Intelligence"
        case .appleIntelligenceNotEnabled:
            return "Apple Intelligence is not enabled"
        case .modelNotReady:
            return "Language model is still preparing"
        @unknown default:
            return "Language model unavailable"
        }
    }
}
