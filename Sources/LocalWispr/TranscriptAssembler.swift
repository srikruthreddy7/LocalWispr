import CoreMedia
import Foundation

struct TranscriptAssembler: Sendable {
    private struct Segment: Sendable, Equatable {
        let range: CMTimeRange
        let text: String
        let isFinal: Bool
    }

    private var finalizedSegments: [Segment] = []
    private var volatileSegment: Segment?

    mutating func consume(text: String, range: CMTimeRange, isFinal: Bool) {
        let normalizedText = normalize(text)
        guard !normalizedText.isEmpty else { return }

        let segment = Segment(range: range, text: normalizedText, isFinal: isFinal)

        if isFinal {
            let segmentRange = segment.range
            finalizedSegments.removeAll { Self.overlaps($0.range, segmentRange) }
            insertFinal(segment)

            if let volatileSegment, Self.overlaps(volatileSegment.range, segment.range) {
                self.volatileSegment = nil
            }
        } else {
            volatileSegment = segment
        }
    }

    var transcript: String {
        var allSegments = finalizedSegments
        if let volatileSegment {
            allSegments.append(volatileSegment)
        }

        let orderedSegments = allSegments.sorted(by: segmentPrecedes)
        var merged: [Segment] = []

        for segment in orderedSegments {
            if let last = merged.last, Self.overlaps(last.range, segment.range) {
                if last.isFinal && !segment.isFinal {
                    continue
                }
                merged.removeLast()
            }
            merged.append(segment)
        }

        return joinNaturally(merged.map(\.text))
    }

    private mutating func insertFinal(_ segment: Segment) {
        let insertionIndex = finalizedSegments.firstIndex { existing in
            segmentPrecedes(segment, existing)
        } ?? finalizedSegments.endIndex
        finalizedSegments.insert(segment, at: insertionIndex)
    }

    private func segmentPrecedes(_ lhs: Segment, _ rhs: Segment) -> Bool {
        let startComparison = CMTimeCompare(lhs.range.start, rhs.range.start)
        if startComparison != 0 {
            return startComparison < 0
        }

        return CMTimeCompare(CMTimeRangeGetEnd(lhs.range), CMTimeRangeGetEnd(rhs.range)) < 0
    }

    private static func overlaps(_ lhs: CMTimeRange, _ rhs: CMTimeRange) -> Bool {
        let lhsEnd = CMTimeRangeGetEnd(lhs)
        let rhsEnd = CMTimeRangeGetEnd(rhs)
        return CMTimeCompare(lhs.start, rhsEnd) < 0 && CMTimeCompare(rhs.start, lhsEnd) < 0
    }

    private func normalize(_ text: String) -> String {
        text.replacingOccurrences(of: "\\s+", with: " ", options: .regularExpression)
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func joinNaturally(_ parts: [String]) -> String {
        var combined = ""

        for part in parts where !part.isEmpty {
            if combined.isEmpty {
                combined = part
                continue
            }

            if combined.last?.isWhitespace == true || part.first?.isWhitespace == true || startsWithClosingPunctuation(part) {
                combined += part
            } else {
                combined += " " + part
            }
        }

        return combined
    }

    private func startsWithClosingPunctuation(_ text: String) -> Bool {
        guard let scalar = text.unicodeScalars.first else { return false }
        return CharacterSet(charactersIn: ".,!?;:)").contains(scalar)
    }
}
