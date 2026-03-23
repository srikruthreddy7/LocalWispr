import CoreMedia
import XCTest
@testable import LocalWispr

final class TranscriptAssemblerTests: XCTestCase {
    func testBuildsTranscriptFromMultipleFinalSegments() {
        var assembler = TranscriptAssembler()

        assembler.consume(text: "this is", range: range(start: 0, duration: 1), isFinal: true)
        assembler.consume(text: "a longer", range: range(start: 1, duration: 1), isFinal: true)
        assembler.consume(text: "dictation test", range: range(start: 2, duration: 1), isFinal: true)

        XCTAssertEqual(assembler.transcript, "this is a longer dictation test")
    }

    func testKeepsFinalizedPrefixWhileVolatileTailChanges() {
        var assembler = TranscriptAssembler()

        assembler.consume(text: "one two", range: range(start: 0, duration: 2), isFinal: true)
        assembler.consume(text: "three fou", range: range(start: 2, duration: 2), isFinal: false)
        XCTAssertEqual(assembler.transcript, "one two three fou")

        assembler.consume(text: "three four", range: range(start: 2, duration: 2), isFinal: false)
        XCTAssertEqual(assembler.transcript, "one two three four")
    }

    func testFinalTailReplacesVolatileTail() {
        var assembler = TranscriptAssembler()

        assembler.consume(text: "the first half", range: range(start: 0, duration: 2), isFinal: true)
        assembler.consume(text: "second ha", range: range(start: 2, duration: 2), isFinal: false)
        assembler.consume(text: "second half", range: range(start: 2, duration: 2), isFinal: true)

        XCTAssertEqual(assembler.transcript, "the first half second half")
    }

    private func range(start: Double, duration: Double) -> CMTimeRange {
        CMTimeRange(
            start: CMTime(seconds: start, preferredTimescale: 600),
            duration: CMTime(seconds: duration, preferredTimescale: 600)
        )
    }
}
