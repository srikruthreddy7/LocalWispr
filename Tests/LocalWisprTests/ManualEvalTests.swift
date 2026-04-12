import AVFoundation
import XCTest
@testable import LocalWispr

final class ManualEvalTests: XCTestCase {
    private final class ConversionState: @unchecked Sendable {
        var didProvideInput = false
    }

    func testParakeetOnEvalSet() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let evalDir = env["LOCALWISPR_EVAL_DIR"], !evalDir.isEmpty else {
            throw XCTSkip("LOCALWISPR_EVAL_DIR not set")
        }

        let rootURL = URL(fileURLWithPath: evalDir, isDirectory: true)
        let clips = try Self.discoverClips(in: rootURL)
        guard !clips.isEmpty else {
            throw XCTSkip("No eval clips found in \(evalDir)")
        }

        let provider = ParakeetRealtimeProvider()
        try await provider.prepare()

        var summaries: [String] = []

        for clip in clips {
            let samples = try Self.loadFloatSamples(from: clip.audioURL)
            let predicted = try await provider.transcribeFinal(samples)
            let normalizedPredicted = Self.normalize(predicted)
            let normalizedReference = Self.normalize(clip.referenceText)
            let wer = Self.wordErrorRate(reference: normalizedReference, hypothesis: normalizedPredicted)

            let summary = """
            FILE: \(clip.audioURL.lastPathComponent)
            REF: \(clip.referenceText)
            HYP: \(predicted)
            WER: \(String(format: "%.3f", wer))
            """
            summaries.append(summary)
        }

        print("LOCALWISPR_EVAL_RESULTS_BEGIN")
        for summary in summaries {
            print(summary)
            print("---")
        }
        print("LOCALWISPR_EVAL_RESULTS_END")

        XCTAssertFalse(summaries.isEmpty)
    }

    private struct EvalClip {
        let audioURL: URL
        let referenceText: String
    }

    private static func discoverClips(in rootURL: URL) throws -> [EvalClip] {
        let fm = FileManager.default
        let contents = try fm.contentsOfDirectory(
            at: rootURL,
            includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles]
        )

        let audioFiles = contents.filter { url in
            let ext = url.pathExtension.lowercased()
            return ["m4a", "wav", "aiff", "mp3"].contains(ext)
        }.sorted { $0.lastPathComponent < $1.lastPathComponent }

        return try audioFiles.compactMap { audioURL in
            let base = audioURL.deletingPathExtension().lastPathComponent
            let txtURL = rootURL.appendingPathComponent("\(base)_eng.txt")
            let rtfURL = rootURL.appendingPathComponent("\(base)_eng.rtf")

            if fm.fileExists(atPath: txtURL.path) {
                let text = try String(contentsOf: txtURL, encoding: .utf8)
                return EvalClip(audioURL: audioURL, referenceText: text.trimmingCharacters(in: .whitespacesAndNewlines))
            }

            if fm.fileExists(atPath: rtfURL.path) {
                let data = try Data(contentsOf: rtfURL)
                let attributed = try NSAttributedString(
                    data: data,
                    options: [.documentType: NSAttributedString.DocumentType.rtf],
                    documentAttributes: nil
                )
                guard !attributed.string.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                    return nil
                }
                return EvalClip(audioURL: audioURL, referenceText: attributed.string.trimmingCharacters(in: .whitespacesAndNewlines))
            }

            return nil
        }
    }

    private static func loadFloatSamples(from url: URL) throws -> [Float] {
        let file = try AVAudioFile(forReading: url)
        let sourceFormat = file.processingFormat
        guard let sourceBuffer = AVAudioPCMBuffer(
            pcmFormat: sourceFormat,
            frameCapacity: AVAudioFrameCount(file.length)
        ) else {
            throw NSError(
                domain: "ManualEvalTests",
                code: -1,
                userInfo: [NSLocalizedDescriptionKey: "Unable to allocate audio buffer"]
            )
        }

        try file.read(into: sourceBuffer)
        guard sourceBuffer.frameLength > 0 else { return [] }

        let targetFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: 16_000,
            channels: 1,
            interleaved: false
        )!

        let normalizedBuffer: AVAudioPCMBuffer
        if sourceFormat.sampleRate == targetFormat.sampleRate,
           sourceFormat.channelCount == targetFormat.channelCount,
           sourceFormat.commonFormat == targetFormat.commonFormat,
           sourceFormat.isInterleaved == targetFormat.isInterleaved {
            normalizedBuffer = sourceBuffer
        } else {
            guard let converter = AVAudioConverter(from: sourceFormat, to: targetFormat) else {
                throw NSError(
                    domain: "ManualEvalTests",
                    code: -2,
                    userInfo: [NSLocalizedDescriptionKey: "Unable to create audio converter"]
                )
            }

            let ratio = targetFormat.sampleRate / sourceFormat.sampleRate
            let capacity = AVAudioFrameCount((Double(sourceBuffer.frameLength) * ratio).rounded(.up)) + 32
            guard let convertedBuffer = AVAudioPCMBuffer(
                pcmFormat: targetFormat,
                frameCapacity: max(capacity, 64)
            ) else {
                throw NSError(
                    domain: "ManualEvalTests",
                    code: -3,
                    userInfo: [NSLocalizedDescriptionKey: "Unable to allocate converted buffer"]
                )
            }

            let conversionState = ConversionState()
            var conversionError: NSError?
            let status = converter.convert(to: convertedBuffer, error: &conversionError) { _, outStatus in
                if conversionState.didProvideInput {
                    outStatus.pointee = .endOfStream
                    return nil
                }

                conversionState.didProvideInput = true
                outStatus.pointee = .haveData
                return sourceBuffer
            }

            if status == .error || convertedBuffer.frameLength == 0 {
                throw conversionError ?? NSError(
                    domain: "ManualEvalTests",
                    code: -4,
                    userInfo: [NSLocalizedDescriptionKey: "Audio conversion failed"]
                )
            }

            normalizedBuffer = convertedBuffer
        }

        guard let channels = normalizedBuffer.floatChannelData else {
            throw NSError(
                domain: "ManualEvalTests",
                code: -5,
                userInfo: [NSLocalizedDescriptionKey: "Converted buffer missing float channel data"]
            )
        }

        return Array(UnsafeBufferPointer(start: channels[0], count: Int(normalizedBuffer.frameLength)))
    }

    private static func normalize(_ text: String) -> String {
        text
            .lowercased()
            .replacingOccurrences(of: "[^a-z0-9\\s]", with: " ", options: .regularExpression)
            .components(separatedBy: .whitespacesAndNewlines)
            .filter { !$0.isEmpty }
            .joined(separator: " ")
    }

    private static func wordErrorRate(reference: String, hypothesis: String) -> Double {
        let refWords = reference.split(separator: " ").map(String.init)
        let hypWords = hypothesis.split(separator: " ").map(String.init)

        if refWords.isEmpty {
            return hypWords.isEmpty ? 0.0 : 1.0
        }

        let distance = editDistance(refWords, hypWords)
        return Double(distance) / Double(refWords.count)
    }

    private static func editDistance(_ lhs: [String], _ rhs: [String]) -> Int {
        if lhs.isEmpty { return rhs.count }
        if rhs.isEmpty { return lhs.count }

        var previous = Array(0...rhs.count)
        var current = Array(repeating: 0, count: rhs.count + 1)

        for i in 1...lhs.count {
            current[0] = i
            for j in 1...rhs.count {
                let substitutionCost = lhs[i - 1] == rhs[j - 1] ? 0 : 1
                current[j] = min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + substitutionCost
                )
            }
            swap(&previous, &current)
        }

        return previous[rhs.count]
    }
}
