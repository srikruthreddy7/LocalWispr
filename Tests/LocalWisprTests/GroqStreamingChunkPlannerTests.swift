import XCTest
@testable import LocalWispr

final class GroqStreamingChunkPlannerTests: XCTestCase {
    func testDoesNotDispatchRegularWindowBeforeThreshold() {
        var planner = GroqStreamingChunkPlanner(chunkDurationSeconds: 10, overlapSeconds: 2)

        XCTAssertNil(planner.nextRegularWindow(bufferedDurationSeconds: 9.99))
        XCTAssertEqual(planner.nextRegularWindow(bufferedDurationSeconds: 10), 0...10)
    }

    func testDispatchesOverlappingRegularWindows() {
        var planner = GroqStreamingChunkPlanner(chunkDurationSeconds: 10, overlapSeconds: 2)

        XCTAssertEqual(planner.nextRegularWindow(bufferedDurationSeconds: 10), 0...10)
        XCTAssertNil(planner.nextRegularWindow(bufferedDurationSeconds: 17.99))
        XCTAssertEqual(planner.nextRegularWindow(bufferedDurationSeconds: 18), 8...18)
        XCTAssertEqual(planner.nextRegularWindow(bufferedDurationSeconds: 26), 16...26)
    }

    func testFinalWindowCoversTailOfLongRecording() {
        var planner = GroqStreamingChunkPlanner(chunkDurationSeconds: 10, overlapSeconds: 2)
        _ = planner.nextRegularWindow(bufferedDurationSeconds: 10)
        _ = planner.nextRegularWindow(bufferedDurationSeconds: 18)

        XCTAssertEqual(planner.finalWindow(totalDurationSeconds: 19), 9...19)
    }

    func testFinalWindowFallsBackToFullShortRecording() {
        var planner = GroqStreamingChunkPlanner(chunkDurationSeconds: 10, overlapSeconds: 2)

        XCTAssertEqual(planner.finalWindow(totalDurationSeconds: 4.5), 0...4.5)
    }

    func testFinalWindowSkipsDuplicateOfLastRegularWindow() {
        var planner = GroqStreamingChunkPlanner(chunkDurationSeconds: 10, overlapSeconds: 2)
        _ = planner.nextRegularWindow(bufferedDurationSeconds: 10)

        XCTAssertNil(planner.finalWindow(totalDurationSeconds: 10))
    }
}
