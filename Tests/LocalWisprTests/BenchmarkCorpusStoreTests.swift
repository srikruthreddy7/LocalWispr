import AVFoundation
import XCTest
@testable import LocalWispr

final class BenchmarkCorpusStoreTests: XCTestCase {
    func testSaveClipCopiesAudioReferenceAndManifest() throws {
        let rootURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("LocalWisprBenchmarkCorpusStoreTests-\(UUID().uuidString)", isDirectory: true)
        let sourceURL = rootURL.appendingPathComponent("source.wav")
        try FileManager.default.createDirectory(at: rootURL, withIntermediateDirectories: true)
        try Self.writeSilentWav(to: sourceURL)

        let store = BenchmarkCorpusStore(rootURL: rootURL.appendingPathComponent("Corpus", isDirectory: true))
        let entry = try store.saveClip(
            audioURL: sourceURL,
            category: .hardAccent,
            promptText: " say this ",
            referenceText: " hello from hyderabad ",
            inputDeviceName: "Unit Test Mic"
        )

        XCTAssertEqual(entry.category, .hardAccent)
        XCTAssertEqual(entry.promptText, "say this")
        XCTAssertEqual(entry.referenceText, "hello from hyderabad")
        XCTAssertEqual(entry.inputDeviceName, "Unit Test Mic")
        XCTAssertNotNil(entry.durationMilliseconds)
        XCTAssertTrue(FileManager.default.fileExists(atPath: entry.audioPath))

        let referenceURL = URL(fileURLWithPath: entry.audioPath).deletingPathExtension().appendingPathExtension("txt")
        XCTAssertEqual(try String(contentsOf: referenceURL, encoding: .utf8), "hello from hyderabad")

        let manifest = try store.loadManifest()
        XCTAssertEqual(manifest.count, 1)
        XCTAssertEqual(manifest.first?.id, entry.id)
        XCTAssertEqual(manifest.first?.category, entry.category)
        XCTAssertEqual(manifest.first?.referenceText, entry.referenceText)
        XCTAssertEqual(manifest.first?.audioPath, entry.audioPath)
    }

    func testSaveClipRejectsEmptyReference() throws {
        let rootURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("LocalWisprBenchmarkCorpusStoreTests-\(UUID().uuidString)", isDirectory: true)
        let sourceURL = rootURL.appendingPathComponent("source.wav")
        try FileManager.default.createDirectory(at: rootURL, withIntermediateDirectories: true)
        try Self.writeSilentWav(to: sourceURL)

        let store = BenchmarkCorpusStore(rootURL: rootURL.appendingPathComponent("Corpus", isDirectory: true))

        XCTAssertThrowsError(
            try store.saveClip(
                audioURL: sourceURL,
                category: .dictation,
                promptText: "",
                referenceText: "   ",
                inputDeviceName: "Unit Test Mic"
            )
        ) { error in
            XCTAssertEqual(error as? BenchmarkCorpusStoreError, .emptyReference)
        }
    }

    private static func writeSilentWav(to url: URL) throws {
        let format = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: 16_000,
            channels: 1,
            interleaved: false
        )!
        let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 16_000)!
        buffer.frameLength = 16_000
        let file = try AVAudioFile(
            forWriting: url,
            settings: format.settings,
            commonFormat: format.commonFormat,
            interleaved: format.isInterleaved
        )
        try file.write(from: buffer)
    }
}
