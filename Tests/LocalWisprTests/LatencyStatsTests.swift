import XCTest
@testable import LocalWispr

final class LatencyStatsTests: XCTestCase {
    func testStopLatencyStatsFromRecentTotalsComputesAverageAndPercentiles() {
        let stats = StopLatencyStats.fromRecentTotals([132, 210, 180, 90])

        XCTAssertNotNil(stats)
        XCTAssertEqual(stats?.sampleCount, 4)
        XCTAssertEqual(stats?.averageMilliseconds, 153)
        XCTAssertEqual(stats?.p50Milliseconds, 132)
        XCTAssertEqual(stats?.p95Milliseconds, 210)
    }

    func testRecentLatencyTotalsUsesNewestPersistedRecordsAndSkipsMissingLatency() {
        let base = Date(timeIntervalSinceReferenceDate: 1_000)
        let records = [
            makeRecord(
                createdAt: base.addingTimeInterval(-30),
                latency: PipelineLatency(stopToTranscriptMilliseconds: 80, cleanupMilliseconds: 20, insertionMilliseconds: 32)
            ),
            makeRecord(
                createdAt: base.addingTimeInterval(10),
                latency: PipelineLatency(stopToTranscriptMilliseconds: 100, cleanupMilliseconds: 40, insertionMilliseconds: 70)
            ),
            makeRecord(
                createdAt: base.addingTimeInterval(20),
                latency: nil
            ),
            makeRecord(
                createdAt: base.addingTimeInterval(30),
                latency: PipelineLatency(stopToTranscriptMilliseconds: 0, cleanupMilliseconds: 0, insertionMilliseconds: 0)
            ),
            makeRecord(
                createdAt: base,
                latency: PipelineLatency(stopToTranscriptMilliseconds: 60, cleanupMilliseconds: 10, insertionMilliseconds: 20)
            )
        ]

        let totals = AppState.recentLatencyTotals(from: records)

        XCTAssertEqual(totals, [210, 90, 132])
    }

    private func makeRecord(createdAt: Date, latency: PipelineLatency?) -> TranscriptRecord {
        TranscriptRecord(
            createdAt: createdAt,
            transcriberMode: .dictationLong,
            localeIdentifier: "en_US",
            rawText: "raw",
            cleanedText: "cleaned",
            outcome: .inserted,
            latency: latency
        )
    }
}
