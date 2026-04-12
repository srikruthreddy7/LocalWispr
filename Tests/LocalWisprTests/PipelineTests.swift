import XCTest
@testable import LocalWispr

private struct MockCleaner: Cleaning {
    let availability: CleanerAvailability
    let result: Result<String, Error>

    func clean(_ rawText: String) async throws -> String {
        try result.get()
    }
}

private actor MockInserter: Inserting {
    private(set) var inserted: [String] = []
    var shouldFail: Bool

    init(shouldFail: Bool = false) {
        self.shouldFail = shouldFail
    }

    func insertAtCursor(_ text: String) async throws {
        if shouldFail {
            throw TestError.insertionFailed
        }
        inserted.append(text)
    }

    func allInserted() -> [String] {
        inserted
    }
}

private enum TestError: Error {
    case cleanerFailed
    case insertionFailed
}

final class PipelineTests: XCTestCase {
    func testNoSpeechSkipsCleanerAndInserter() async {
        let inserter = MockInserter()
        let pipeline = Pipeline(
            cleaner: MockCleaner(availability: .available, result: .success("unused")),
            inserter: inserter
        )

        let result = await pipeline.process(
            rawText: "   ",
            stopToTranscriptMilliseconds: 212,
            recordingDurationMilliseconds: 1_500
        )

        guard case .noSpeech(let latency) = result else {
            return XCTFail("Expected noSpeech result")
        }
        XCTAssertEqual(latency.stopToTranscriptMilliseconds, 212)
        XCTAssertEqual(latency.recordingDurationMilliseconds, 1_500)
        XCTAssertGreaterThanOrEqual(latency.totalStopToInsertMilliseconds, 0)
        let inserted = await inserter.allInserted()
        XCTAssertTrue(inserted.isEmpty)
    }

    func testUnavailableCleanerFailsPipeline() async {
        let inserter = MockInserter()
        let pipeline = Pipeline(
            cleaner: MockCleaner(availability: .unavailable("model not ready"), result: .failure(TestError.cleanerFailed)),
            inserter: inserter
        )

        let result = await pipeline.process(
            rawText: "hello there",
            stopToTranscriptMilliseconds: 120,
            recordingDurationMilliseconds: 2_400
        )

        guard case .failed(let raw, let cleaned, let error, let latency) = result else {
            return XCTFail("Expected failed result")
        }

        XCTAssertEqual(raw, "hello there")
        XCTAssertNil(cleaned)
        XCTAssertTrue(error.contains("Text cleanup unavailable"))
        XCTAssertEqual(latency.stopToTranscriptMilliseconds, 120)
        XCTAssertGreaterThanOrEqual(latency.totalStopToInsertMilliseconds, 0)

        let inserted = await inserter.allInserted()
        XCTAssertTrue(inserted.isEmpty)
    }

    func testCleanerFailureFailsPipeline() async {
        let inserter = MockInserter()
        let pipeline = Pipeline(
            cleaner: MockCleaner(availability: .available, result: .failure(TestError.cleanerFailed)),
            inserter: inserter
        )

        let result = await pipeline.process(
            rawText: "raw speech",
            stopToTranscriptMilliseconds: 300,
            recordingDurationMilliseconds: 2_100
        )

        guard case .failed(let raw, let cleaned, let error, let latency) = result else {
            return XCTFail("Expected failed result")
        }

        XCTAssertEqual(raw, "raw speech")
        XCTAssertNil(cleaned)
        XCTAssertTrue(error.contains("Text cleanup failed"))
        XCTAssertEqual(latency.stopToTranscriptMilliseconds, 300)
        XCTAssertGreaterThanOrEqual(latency.totalStopToInsertMilliseconds, 0)

        let inserted = await inserter.allInserted()
        XCTAssertTrue(inserted.isEmpty)
    }

    func testInsertionFailureReturnsFailedResult() async {
        let inserter = MockInserter(shouldFail: true)
        let pipeline = Pipeline(
            cleaner: MockCleaner(availability: .available, result: .success("clean phrase")),
            inserter: inserter
        )

        let result = await pipeline.process(
            rawText: "test phrase",
            stopToTranscriptMilliseconds: 88,
            recordingDurationMilliseconds: 500
        )

        guard case .failed(let raw, let cleaned, let error, let latency) = result else {
            return XCTFail("Expected failed result")
        }

        XCTAssertEqual(raw, "test phrase")
        XCTAssertEqual(cleaned, "clean phrase")
        XCTAssertTrue(error.contains("Text insertion failed"))
        XCTAssertEqual(latency.stopToTranscriptMilliseconds, 88)
        XCTAssertGreaterThanOrEqual(latency.totalStopToInsertMilliseconds, 0)
    }
}
