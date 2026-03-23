import AppKit
import Foundation

struct RightCommandDetectionDecision: Equatable {
    var consumeEvent: Bool = false
    var toggleRequested: Bool = false
    var beganPress: Bool = false
    var endedPress: Bool = false
}

struct RightCommandTapDetector {
    private let tapWindow: TimeInterval

    private var rightCommandDown = false
    private var rightCommandDownAt: TimeInterval?
    private var consumeCurrentRightCommandPress = false

    private var firstTapDownAt: TimeInterval?
    private var firstTapUpAt: TimeInterval?
    private var awaitingSecondTap = false
    private var interveningKeyBetweenTaps = false
    private var nonRightKeyDuringCurrentPress = false

    init(tapWindow: TimeInterval) {
        self.tapWindow = tapWindow
    }

    mutating func handleRightCommandDown(now: TimeInterval) -> RightCommandDetectionDecision {
        expireTapCandidateIfNeeded(now: now)

        if rightCommandDown {
            return .init()
        }

        rightCommandDown = true
        rightCommandDownAt = now
        nonRightKeyDuringCurrentPress = false

        if awaitingSecondTap,
           !interveningKeyBetweenTaps,
           let firstDown = firstTapDownAt,
           let firstUp = firstTapUpAt,
           now - firstDown <= tapWindow,
           now - firstUp <= tapWindow {
            consumeCurrentRightCommandPress = true
            resetTapCandidateKeepingPressState()
            return .init(consumeEvent: true, toggleRequested: true, beganPress: true)
        }

        firstTapDownAt = now
        firstTapUpAt = nil
        awaitingSecondTap = false
        interveningKeyBetweenTaps = false
        consumeCurrentRightCommandPress = false

        return .init(beganPress: true)
    }

    mutating func handleRightCommandUp(now: TimeInterval) -> RightCommandDetectionDecision {
        guard rightCommandDown else {
            return .init()
        }

        rightCommandDown = false
        defer {
            rightCommandDownAt = nil
            nonRightKeyDuringCurrentPress = false
        }

        if consumeCurrentRightCommandPress {
            consumeCurrentRightCommandPress = false
            return .init(consumeEvent: true, endedPress: true)
        }

        guard let pressStart = rightCommandDownAt else {
            return .init(endedPress: true)
        }

        let duration = now - pressStart
        if duration > tapWindow || nonRightKeyDuringCurrentPress {
            resetTapCandidateKeepingPressState()
            return .init(endedPress: true)
        }

        if firstTapDownAt != nil, firstTapUpAt == nil {
            firstTapUpAt = now
            awaitingSecondTap = true
            interveningKeyBetweenTaps = false
        }

        return .init(endedPress: true)
    }

    mutating func registerNonRightCommandKeyEvent() {
        if rightCommandDown {
            nonRightKeyDuringCurrentPress = true
        }
        if awaitingSecondTap {
            interveningKeyBetweenTaps = true
        }
    }

    mutating func handleHoldTimeout(pressStartTime: TimeInterval) {
        guard rightCommandDown,
              let currentStart = rightCommandDownAt,
              abs(currentStart - pressStartTime) < 0.005 else {
            return
        }

        resetTapCandidateKeepingPressState()
    }

    mutating func expireTapCandidateIfNeeded(now: TimeInterval) {
        guard awaitingSecondTap, let firstTapUpAt else { return }

        if now - firstTapUpAt > tapWindow {
            resetTapCandidateKeepingPressState()
        }
    }

    mutating func resetAll() {
        rightCommandDown = false
        rightCommandDownAt = nil
        consumeCurrentRightCommandPress = false
        nonRightKeyDuringCurrentPress = false
        resetTapCandidateKeepingPressState()
    }

    private mutating func resetTapCandidateKeepingPressState() {
        firstTapDownAt = nil
        firstTapUpAt = nil
        awaitingSecondTap = false
        interveningKeyBetweenTaps = false
    }
}

public enum HotkeyMonitorError: LocalizedError {
    case eventTapCreationFailed
    case hotkeyRegistrationFailed(String)

    public var errorDescription: String? {
        switch self {
        case .eventTapCreationFailed:
            return "Unable to create keyboard event tap. Grant Accessibility access in System Settings."
        case .hotkeyRegistrationFailed(let message):
            return message
        }
    }
}

public final class HotkeyMonitor: @unchecked Sendable, HotkeyMonitoring {
    public var onToggleRequested: (@MainActor () -> Void)?

    private enum CommandTransition {
        case down
        case up
    }

    /// Physical key: 54 = right ⌘, 55 = left ⌘ (ANSI / most Apple keyboards).
    private let watchedKeyCode: Int64
    private let tapWindow: TimeInterval = 0.55

    private let lock = NSLock()
    private var globalFlagsMonitor: Any?
    private var globalKeyDownMonitor: Any?
    private var localFlagsMonitor: Any?
    private var localKeyDownMonitor: Any?

    private var detector: RightCommandTapDetector

    private let timerQueue = DispatchQueue(label: "LocalWispr.HotkeyMonitor.timer")
    private var holdWorkItem: DispatchWorkItem?

    public init(keyCode: Int64 = 54) {
        self.watchedKeyCode = keyCode
        detector = RightCommandTapDetector(tapWindow: 0.55)
    }

    deinit {
        stop()
    }

    public func start() throws {
        lock.lock()
        defer { lock.unlock() }

        guard globalFlagsMonitor == nil,
              globalKeyDownMonitor == nil,
              localFlagsMonitor == nil,
              localKeyDownMonitor == nil else {
            return
        }

        guard let globalFlagsMonitor = NSEvent.addGlobalMonitorForEvents(matching: .flagsChanged, handler: { [weak self] event in
            self?.handleEvent(event)
        }),
        let globalKeyDownMonitor = NSEvent.addGlobalMonitorForEvents(matching: .keyDown, handler: { [weak self] event in
            self?.handleEvent(event)
        }),
        let localFlagsMonitor = NSEvent.addLocalMonitorForEvents(matching: .flagsChanged, handler: { [weak self] event in
            self?.handleEvent(event)
            return event
        }),
        let localKeyDownMonitor = NSEvent.addLocalMonitorForEvents(matching: .keyDown, handler: { [weak self] event in
            self?.handleEvent(event)
            return event
        }) else {
            stopWithoutLock()
            throw HotkeyMonitorError.eventTapCreationFailed
        }

        self.globalFlagsMonitor = globalFlagsMonitor
        self.globalKeyDownMonitor = globalKeyDownMonitor
        self.localFlagsMonitor = localFlagsMonitor
        self.localKeyDownMonitor = localKeyDownMonitor
    }

    public func stop() {
        lock.lock()
        stopWithoutLock()
        lock.unlock()
    }

    private func handleEvent(_ event: NSEvent) {
        let now = ProcessInfo.processInfo.systemUptime

        lock.lock()
        defer { lock.unlock() }

        detector.expireTapCandidateIfNeeded(now: now)

        let keyCode = Int64(event.keyCode)

        if keyCode != watchedKeyCode {
            if isKeyboardLikeEvent(event.type) {
                detector.registerNonRightCommandKeyEvent()
            }
            return
        }

        // Modifier keys surface as `flagsChanged`; ignore any synthetic keyDown/keyUp duplicates.
        if event.type == .keyDown || event.type == .keyUp {
            return
        }

        guard let transition = rightCommandTransition(for: event) else {
            return
        }

        switch transition {
        case .down:
            handleRightCommandDown(now: now)
        case .up:
            handleRightCommandUp(now: now)
        }
    }

    private func handleRightCommandDown(now: TimeInterval) {
        let decision = detector.handleRightCommandDown(now: now)

        if decision.beganPress {
            scheduleHoldTimeout(pressStartTime: now)
        }

        if decision.toggleRequested {
            Task { @MainActor [weak self] in
                self?.onToggleRequested?()
            }
        }
    }

    private func handleRightCommandUp(now: TimeInterval) {
        let decision = detector.handleRightCommandUp(now: now)
        if decision.endedPress {
            holdWorkItem?.cancel()
            holdWorkItem = nil
        }
    }

    private func rightCommandTransition(for event: NSEvent) -> CommandTransition? {
        guard event.type == .flagsChanged else { return nil }
        return event.modifierFlags.contains(.command) ? .down : .up
    }

    private func scheduleHoldTimeout(pressStartTime: TimeInterval) {
        holdWorkItem?.cancel()

        let workItem = DispatchWorkItem { [weak self] in
            guard let self else { return }

            self.lock.lock()
            defer { self.lock.unlock() }

            self.detector.handleHoldTimeout(pressStartTime: pressStartTime)
        }

        holdWorkItem = workItem
        timerQueue.asyncAfter(deadline: .now() + tapWindow, execute: workItem)
    }

    private func isKeyboardLikeEvent(_ type: NSEvent.EventType) -> Bool {
        type == .keyDown || type == .keyUp || type == .flagsChanged
    }

    private func stopWithoutLock() {
        if let globalFlagsMonitor {
            NSEvent.removeMonitor(globalFlagsMonitor)
        }
        if let globalKeyDownMonitor {
            NSEvent.removeMonitor(globalKeyDownMonitor)
        }
        if let localFlagsMonitor {
            NSEvent.removeMonitor(localFlagsMonitor)
        }
        if let localKeyDownMonitor {
            NSEvent.removeMonitor(localKeyDownMonitor)
        }

        globalFlagsMonitor = nil
        globalKeyDownMonitor = nil
        localFlagsMonitor = nil
        localKeyDownMonitor = nil
        detector.resetAll()

        holdWorkItem?.cancel()
        holdWorkItem = nil
    }
}
