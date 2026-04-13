import AVFoundation
import XCTest
@testable import LocalWispr

final class TranscriberIntegrationTests: XCTestCase {
    private final class ConversionState: @unchecked Sendable {
        var didProvideInput = false
    }

    func testModalSTTTranscribesProvidedAudioFile() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let audioPath = env["LOCALWISPR_TRANSCRIBER_AUDIO"], !audioPath.isEmpty else {
            throw XCTSkip("LOCALWISPR_TRANSCRIBER_AUDIO not set")
        }

        guard env["LOCALWISPR_MODAL_STT_ENDPOINT"]?.isEmpty == false else {
            throw XCTSkip("LOCALWISPR_MODAL_STT_ENDPOINT not set")
        }

        let url = URL(fileURLWithPath: audioPath)
        let buffers = try Self.loadTargetBuffers(from: url)
        let transcriber = Transcriber()

        let transcript = try await transcriber.transcribe(
            buffers: buffers,
            mode: .dictationLong,
            locale: Locale(identifier: "en_US"),
            contextualStrings: []
        )

        XCTAssertFalse(transcript.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        print("LOCALWISPR_TRANSCRIBER_RESULT_BEGIN")
        print(transcript)
        print("LOCALWISPR_TRANSCRIBER_RESULT_END")
    }

    private static func loadTargetBuffers(from url: URL) throws -> [AVAudioPCMBuffer] {
        let file = try AVAudioFile(forReading: url)
        let sourceFormat = file.processingFormat
        let targetFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: 16_000,
            channels: 1,
            interleaved: false
        )!

        let chunkFrames = AVAudioFrameCount(16_000)
        var buffers: [AVAudioPCMBuffer] = []

        while true {
            guard let sourceBuffer = AVAudioPCMBuffer(pcmFormat: sourceFormat, frameCapacity: chunkFrames) else {
                throw NSError(domain: "TranscriberIntegrationTests", code: -1, userInfo: [NSLocalizedDescriptionKey: "Unable to allocate source buffer"])
            }

            try file.read(into: sourceBuffer, frameCount: chunkFrames)
            if sourceBuffer.frameLength == 0 {
                break
            }

            if sourceFormat.sampleRate == targetFormat.sampleRate,
               sourceFormat.channelCount == targetFormat.channelCount,
               sourceFormat.commonFormat == targetFormat.commonFormat,
               sourceFormat.isInterleaved == targetFormat.isInterleaved {
                buffers.append(sourceBuffer)
                continue
            }

            guard let converter = AVAudioConverter(from: sourceFormat, to: targetFormat) else {
                throw NSError(domain: "TranscriberIntegrationTests", code: -2, userInfo: [NSLocalizedDescriptionKey: "Unable to create converter"])
            }

            let ratio = targetFormat.sampleRate / sourceFormat.sampleRate
            let capacity = AVAudioFrameCount((Double(sourceBuffer.frameLength) * ratio).rounded(.up)) + 32
            guard let convertedBuffer = AVAudioPCMBuffer(
                pcmFormat: targetFormat,
                frameCapacity: max(capacity, 64)
            ) else {
                throw NSError(domain: "TranscriberIntegrationTests", code: -3, userInfo: [NSLocalizedDescriptionKey: "Unable to allocate converted buffer"])
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
                throw conversionError ?? NSError(domain: "TranscriberIntegrationTests", code: -4, userInfo: [NSLocalizedDescriptionKey: "Audio conversion failed"])
            }

            buffers.append(convertedBuffer)
        }

        return buffers
    }
}
