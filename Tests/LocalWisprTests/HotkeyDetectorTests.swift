import XCTest
@testable import LocalWispr

final class HotkeyDetectorTests: XCTestCase {
    func testSingleTapDoesNotTriggerToggle() {
        var detector = RightCommandTapDetector(tapWindow: 0.4)

        let firstDown = detector.handleRightCommandDown(now: 0.000)
        let firstUp = detector.handleRightCommandUp(now: 0.080)

        XCTAssertFalse(firstDown.toggleRequested)
        XCTAssertFalse(firstUp.toggleRequested)
    }

    func testValidDoubleTapTriggersExactlyOnce() {
        var detector = RightCommandTapDetector(tapWindow: 0.4)

        _ = detector.handleRightCommandDown(now: 0.000)
        _ = detector.handleRightCommandUp(now: 0.080)
        let secondDown = detector.handleRightCommandDown(now: 0.200)
        let secondUp = detector.handleRightCommandUp(now: 0.260)

        XCTAssertTrue(secondDown.toggleRequested)
        XCTAssertTrue(secondDown.consumeEvent)
        XCTAssertFalse(secondUp.toggleRequested)
        XCTAssertTrue(secondUp.consumeEvent)
    }

    func testLongPressNeverTriggers() {
        var detector = RightCommandTapDetector(tapWindow: 0.4)

        _ = detector.handleRightCommandDown(now: 0.000)
        detector.handleHoldTimeout(pressStartTime: 0.000)
        let up = detector.handleRightCommandUp(now: 0.700)

        XCTAssertFalse(up.toggleRequested)
        XCTAssertFalse(up.consumeEvent)
    }

    func testCommandChordInvalidatesTapCandidate() {
        var detector = RightCommandTapDetector(tapWindow: 0.4)

        _ = detector.handleRightCommandDown(now: 0.000)
        detector.registerNonRightCommandKeyEvent() // e.g. Cmd+C
        _ = detector.handleRightCommandUp(now: 0.120)
        let secondDown = detector.handleRightCommandDown(now: 0.220)

        XCTAssertFalse(secondDown.toggleRequested)
        XCTAssertFalse(secondDown.consumeEvent)
    }

    func testInterveningKeyBetweenTapsInvalidatesDoubleTap() {
        var detector = RightCommandTapDetector(tapWindow: 0.4)

        _ = detector.handleRightCommandDown(now: 0.000)
        _ = detector.handleRightCommandUp(now: 0.080)
        detector.registerNonRightCommandKeyEvent()
        let secondDown = detector.handleRightCommandDown(now: 0.220)

        XCTAssertFalse(secondDown.toggleRequested)
        XCTAssertFalse(secondDown.consumeEvent)
    }

    func testLeftCommandEquivalentEventIsIgnored() {
        var detector = RightCommandTapDetector(tapWindow: 0.4)

        detector.registerNonRightCommandKeyEvent()
        let down = detector.handleRightCommandDown(now: 1.000)
        let up = detector.handleRightCommandUp(now: 1.050)

        XCTAssertFalse(down.toggleRequested)
        XCTAssertFalse(up.toggleRequested)
    }
}
