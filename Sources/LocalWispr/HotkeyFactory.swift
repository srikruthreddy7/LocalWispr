import AppKit
import ApplicationServices
import Foundation

/// User-selectable global shortcut.
public enum GlobalHotkeyBinding: String, CaseIterable, Identifiable, Sendable {
    case rightCommandDoubleTap
    case leftCommandDoubleTap
    case controlOptionSpace
    case controlOptionD
    case none

    public var id: String { rawValue }

    public var menuTitle: String {
        switch self {
        case .rightCommandDoubleTap:
            return "Right ⌘"
        case .leftCommandDoubleTap:
            return "Left ⌘"
        case .controlOptionSpace:
            return "⌃⌥ Space"
        case .controlOptionD:
            return "⌃⌥ D"
        case .none:
            return "Off"
        }
    }

    /// Short label for the control panel hero pills.
    public var shortPillLabel: String {
        switch self {
        case .rightCommandDoubleTap:
            return "R CMD"
        case .leftCommandDoubleTap:
            return "L CMD"
        case .controlOptionSpace:
            return "⌃⌥ SPACE"
        case .controlOptionD:
            return "⌃⌥ D"
        case .none:
            return "NO SHORTCUT"
        }
    }

    public var requiresAccessibility: Bool {
        self != .none
    }
}

public enum GlobalHotkeyInteractionMode: String, CaseIterable, Identifiable, Sendable {
    case toggle
    case pressAndHold
    case doubleTap

    public var id: String { rawValue }

    public var menuTitle: String {
        switch self {
        case .toggle:
            return "Toggle"
        case .pressAndHold:
            return "Press and Hold"
        case .doubleTap:
            return "Double Tap"
        }
    }

    public var statusSuffix: String {
        switch self {
        case .toggle:
            return "toggle"
        case .pressAndHold:
            return "hold"
        case .doubleTap:
            return "double-tap"
        }
    }
}

public enum HotkeyRegistrationStatus: Equatable, Sendable {
    case pending
    case inactive(String)
    case listening(String)
    case failed(String)

    public var detailLine: String {
        switch self {
        case .pending:
            return "Checking shortcut registration…"
        case .inactive(let reason):
            return reason
        case .listening(let description):
            return "Active — \(description)"
        case .failed(let message):
            return "Not active — \(message)"
        }
    }

    /// Shorter copy for the menu bar extra.
    public var menuBarLine: String {
        switch self {
        case .pending:
            return "Shortcut: checking…"
        case .inactive(let reason):
            return reason
        case .listening(let description):
            return "Shortcut active: \(description)"
        case .failed(let message):
            return "Shortcut not registered — \(message)"
        }
    }
}

private struct HotkeyShortcutDescriptor {
    let keyCode: UInt16
    let modifierFlags: NSEvent.ModifierFlags

    var isModifierOnly: Bool {
        modifierFlags.isEmpty && Self.isModifierKey(keyCode)
    }

    static func isModifierKey(_ keyCode: UInt16) -> Bool {
        switch keyCode {
        case 54, 55, // Command
             58, 61, // Option
             59, 62, // Control
             56, 60, // Shift
             63: // Function
            return true
        default:
            return false
        }
    }
}

public enum HotkeyFactory {
    public static func makeMonitor(
        for binding: GlobalHotkeyBinding,
        interactionMode: GlobalHotkeyInteractionMode
    ) -> HotkeyMonitoring {
        guard let shortcut = shortcut(for: binding) else {
            return NoOpHotkeyMonitor()
        }

        return FluidHotkeyMonitor(shortcut: shortcut, interactionMode: interactionMode)
    }

    private static func shortcut(for binding: GlobalHotkeyBinding) -> HotkeyShortcutDescriptor? {
        switch binding {
        case .rightCommandDoubleTap:
            return HotkeyShortcutDescriptor(keyCode: 54, modifierFlags: [])
        case .leftCommandDoubleTap:
            return HotkeyShortcutDescriptor(keyCode: 55, modifierFlags: [])
        case .controlOptionSpace:
            return HotkeyShortcutDescriptor(keyCode: 49, modifierFlags: [.control, .option])
        case .controlOptionD:
            return HotkeyShortcutDescriptor(keyCode: 2, modifierFlags: [.control, .option])
        case .none:
            return nil
        }
    }
}

/// Fluid-style event-tap monitor with support for toggle, press-and-hold, and double-tap.
private final class FluidHotkeyMonitor: NSObject, @unchecked Sendable, HotkeyMonitoring {
    var onToggleRequested: (@MainActor () -> Void)?
    var onStartRequested: (@MainActor () -> Void)?
    var onStopRequested: (@MainActor () -> Void)?

    private let shortcut: HotkeyShortcutDescriptor
    private let interactionMode: GlobalHotkeyInteractionMode

    private let lock = NSLock()
    private var eventTap: CFMachPort?
    private var runLoopSource: CFRunLoopSource?

    private var isShortcutPressed = false
    private var otherKeyPressedDuringPress = false
    private var holdModeStarted = false
    private var pendingHoldStart: DispatchWorkItem?
    private var lastTapTimestamp: TimeInterval?

    private let holdStartDelay: TimeInterval = 0.15
    private let doubleTapWindow: TimeInterval = 0.40

    init(shortcut: HotkeyShortcutDescriptor, interactionMode: GlobalHotkeyInteractionMode) {
        self.shortcut = shortcut
        self.interactionMode = interactionMode
        super.init()
    }

    deinit {
        stop()
    }

    func start() throws {
        lock.lock()
        defer { lock.unlock() }

        guard eventTap == nil, runLoopSource == nil else { return }

        guard AXIsProcessTrusted() else {
            throw HotkeyMonitorError.eventTapCreationFailed
        }

        let eventMask: CGEventMask = (1 << CGEventType.keyDown.rawValue)
            | (1 << CGEventType.keyUp.rawValue)
            | (1 << CGEventType.flagsChanged.rawValue)
            | (1 << CGEventType.tapDisabledByTimeout.rawValue)
            | (1 << CGEventType.tapDisabledByUserInput.rawValue)

        eventTap = CGEvent.tapCreate(
            tap: .cgSessionEventTap,
            place: .headInsertEventTap,
            options: .defaultTap,
            eventsOfInterest: eventMask,
            callback: { _, type, event, refcon -> Unmanaged<CGEvent>? in
                guard let refcon else {
                    return Unmanaged.passUnretained(event)
                }

                let monitor = Unmanaged<FluidHotkeyMonitor>.fromOpaque(refcon).takeUnretainedValue()
                return monitor.handleEvent(type: type, event: event)
            },
            userInfo: Unmanaged.passUnretained(self).toOpaque()
        )

        guard let eventTap else {
            cleanupEventTapLocked()
            throw HotkeyMonitorError.eventTapCreationFailed
        }

        runLoopSource = CFMachPortCreateRunLoopSource(kCFAllocatorDefault, eventTap, 0)
        guard let runLoopSource else {
            cleanupEventTapLocked()
            throw HotkeyMonitorError.eventTapCreationFailed
        }

        CFRunLoopAddSource(CFRunLoopGetMain(), runLoopSource, .commonModes)
        CGEvent.tapEnable(tap: eventTap, enable: true)

        guard CGEvent.tapIsEnabled(tap: eventTap) else {
            cleanupEventTapLocked()
            throw HotkeyMonitorError.eventTapCreationFailed
        }
    }

    func stop() {
        lock.lock()
        cleanupEventTapLocked()
        lock.unlock()
    }

    private func handleEvent(type: CGEventType, event: CGEvent) -> Unmanaged<CGEvent>? {
        lock.lock()
        defer { lock.unlock() }

        if type == .tapDisabledByTimeout || type == .tapDisabledByUserInput {
            if let eventTap {
                CGEvent.tapEnable(tap: eventTap, enable: true)
            }
            return Unmanaged.passUnretained(event)
        }

        let now = ProcessInfo.processInfo.systemUptime
        let keyCode = UInt16(event.getIntegerValueField(.keyboardEventKeycode))
        let modifiers = relevantModifiers(from: event.flags)

        if type == .keyDown, keyCode != shortcut.keyCode {
            if isShortcutPressed {
                otherKeyPressedDuringPress = true
            }
            pendingHoldStart?.cancel()
            pendingHoldStart = nil
            if interactionMode == .doubleTap {
                lastTapTimestamp = nil
            }
        }

        switch type {
        case .keyDown:
            guard !shortcut.isModifierOnly else {
                return Unmanaged.passUnretained(event)
            }
            guard matchesShortcut(keyCode: keyCode, modifiers: modifiers) else {
                return Unmanaged.passUnretained(event)
            }
            if event.getIntegerValueField(.keyboardEventAutorepeat) == 1 {
                return Unmanaged.passUnretained(event)
            }
            handleShortcutDown(now: now)
            return nil

        case .keyUp:
            guard !shortcut.isModifierOnly else {
                return Unmanaged.passUnretained(event)
            }
            guard matchesShortcut(keyCode: keyCode, modifiers: modifiers) else {
                return Unmanaged.passUnretained(event)
            }
            handleShortcutUp(now: now)
            return nil

        case .flagsChanged:
            guard shortcut.isModifierOnly, keyCode == shortcut.keyCode else {
                return Unmanaged.passUnretained(event)
            }

            let isPressed = isModifierPressed(keyCode: keyCode, flags: event.flags)
            if isPressed {
                handleShortcutDown(now: now)
            } else {
                handleShortcutUp(now: now)
            }
            return nil

        default:
            break
        }

        return Unmanaged.passUnretained(event)
    }

    private func handleShortcutDown(now: TimeInterval) {
        guard !isShortcutPressed else { return }
        isShortcutPressed = true
        otherKeyPressedDuringPress = false

        switch interactionMode {
        case .toggle:
            if !shortcut.isModifierOnly {
                fireToggle()
            }
        case .pressAndHold:
            if shortcut.isModifierOnly {
                scheduleHoldStart()
            } else {
                holdModeStarted = true
                fireStart()
            }
        case .doubleTap:
            if !shortcut.isModifierOnly {
                registerTap(now: now)
            }
        }
    }

    private func handleShortcutUp(now: TimeInterval) {
        guard isShortcutPressed else { return }

        isShortcutPressed = false
        pendingHoldStart?.cancel()
        pendingHoldStart = nil

        let cleanPress = !otherKeyPressedDuringPress
        otherKeyPressedDuringPress = false

        switch interactionMode {
        case .toggle:
            if shortcut.isModifierOnly, cleanPress {
                fireToggle()
            }
        case .pressAndHold:
            defer { holdModeStarted = false }
            if holdModeStarted {
                fireStop()
            }
        case .doubleTap:
            if shortcut.isModifierOnly, cleanPress {
                registerTap(now: now)
            }
        }
    }

    private func scheduleHoldStart() {
        pendingHoldStart?.cancel()

        let workItem = DispatchWorkItem { [weak self] in
            guard let self else { return }

            self.lock.lock()
            let shouldStart = self.isShortcutPressed && !self.otherKeyPressedDuringPress
            if shouldStart {
                self.holdModeStarted = true
            }
            self.lock.unlock()

            if shouldStart {
                self.fireStart()
            }
        }

        pendingHoldStart = workItem
        DispatchQueue.main.asyncAfter(deadline: .now() + holdStartDelay, execute: workItem)
    }

    private func registerTap(now: TimeInterval) {
        if let lastTapTimestamp, now - lastTapTimestamp <= doubleTapWindow {
            self.lastTapTimestamp = nil
            fireToggle()
        } else {
            self.lastTapTimestamp = now
        }
    }

    private func fireToggle() {
        Task { @MainActor [weak self] in
            self?.onToggleRequested?()
        }
    }

    private func fireStart() {
        Task { @MainActor [weak self] in
            if let onStartRequested = self?.onStartRequested {
                onStartRequested()
            } else {
                self?.onToggleRequested?()
            }
        }
    }

    private func fireStop() {
        Task { @MainActor [weak self] in
            if let onStopRequested = self?.onStopRequested {
                onStopRequested()
            } else {
                self?.onToggleRequested?()
            }
        }
    }

    private func matchesShortcut(keyCode: UInt16, modifiers: NSEvent.ModifierFlags) -> Bool {
        guard keyCode == shortcut.keyCode else { return false }
        guard !shortcut.isModifierOnly else { return true }
        let shortcutModifiers = shortcut.modifierFlags.intersection([.function, .command, .option, .control, .shift])
        return modifiers == shortcutModifiers
    }

    private func relevantModifiers(from flags: CGEventFlags) -> NSEvent.ModifierFlags {
        var modifiers: NSEvent.ModifierFlags = []
        if flags.contains(.maskSecondaryFn) { modifiers.insert(.function) }
        if flags.contains(.maskCommand) { modifiers.insert(.command) }
        if flags.contains(.maskAlternate) { modifiers.insert(.option) }
        if flags.contains(.maskControl) { modifiers.insert(.control) }
        if flags.contains(.maskShift) { modifiers.insert(.shift) }
        return modifiers
    }

    private func isModifierPressed(keyCode: UInt16, flags: CGEventFlags) -> Bool {
        switch keyCode {
        case 54, 55:
            return flags.contains(.maskCommand)
        case 58, 61:
            return flags.contains(.maskAlternate)
        case 59, 62:
            return flags.contains(.maskControl)
        case 56, 60:
            return flags.contains(.maskShift)
        case 63:
            return flags.contains(.maskSecondaryFn)
        default:
            return false
        }
    }

    private func cleanupEventTapLocked() {
        pendingHoldStart?.cancel()
        pendingHoldStart = nil

        if let eventTap {
            CGEvent.tapEnable(tap: eventTap, enable: false)
        }

        if let runLoopSource {
            CFRunLoopRemoveSource(CFRunLoopGetMain(), runLoopSource, .commonModes)
        }

        eventTap = nil
        runLoopSource = nil

        isShortcutPressed = false
        otherKeyPressedDuringPress = false
        holdModeStarted = false
        lastTapTimestamp = nil
    }
}

public final class NoOpHotkeyMonitor: HotkeyMonitoring {
    public var onToggleRequested: (@MainActor () -> Void)?
    public var onStartRequested: (@MainActor () -> Void)?
    public var onStopRequested: (@MainActor () -> Void)?

    public init() {}

    public func start() throws {}

    public func stop() {}
}
