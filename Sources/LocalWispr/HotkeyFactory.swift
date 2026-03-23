import AppKit
import Carbon
import Foundation

/// User-selectable global shortcut. Double-tap bindings require Accessibility; Carbon hotkeys do not.
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
            return "Right ⌘ double-tap"
        case .leftCommandDoubleTap:
            return "Left ⌘ double-tap"
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
            return "R CMD × 2"
        case .leftCommandDoubleTap:
            return "L CMD × 2"
        case .controlOptionSpace:
            return "⌃⌥ SPACE"
        case .controlOptionD:
            return "⌃⌥ D"
        case .none:
            return "NO SHORTCUT"
        }
    }

    public var requiresAccessibility: Bool {
        switch self {
        case .rightCommandDoubleTap, .leftCommandDoubleTap:
            return true
        case .controlOptionSpace, .controlOptionD, .none:
            return false
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

public enum HotkeyFactory {
    public static func makeMonitor(for binding: GlobalHotkeyBinding) -> HotkeyMonitoring {
        switch binding {
        case .rightCommandDoubleTap:
            return HotkeyMonitor(keyCode: 54)
        case .leftCommandDoubleTap:
            return HotkeyMonitor(keyCode: 55)
        case .controlOptionSpace, .controlOptionD:
            return CarbonHotkeyMonitor(binding: binding)
        case .none:
            return NoOpHotkeyMonitor()
        }
    }
}

/// Native Carbon global hotkey registration. This is much more reliable than event taps for modifier chords.
public final class CarbonHotkeyMonitor: HotkeyMonitoring, @unchecked Sendable {
    public var onToggleRequested: (@MainActor () -> Void)?

    private static let signature = fourCharCode("LWPR")

    private var handlerRef: EventHandlerRef?
    private var hotKeyRef: EventHotKeyRef?
    private let binding: GlobalHotkeyBinding

    public init(binding: GlobalHotkeyBinding) {
        self.binding = binding
    }

    deinit {
        stop()
    }

    public func start() throws {
        guard hotKeyRef == nil, handlerRef == nil else { return }

        let eventSpec = EventTypeSpec(
            eventClass: OSType(kEventClassKeyboard),
            eventKind: UInt32(kEventHotKeyPressed)
        )

        let installStatus = InstallEventHandler(
            GetApplicationEventTarget(),
            { _, _, userData in
                guard let userData else { return noErr }
                let monitor = Unmanaged<CarbonHotkeyMonitor>.fromOpaque(userData).takeUnretainedValue()
                Task { @MainActor [weak monitor] in
                    monitor?.onToggleRequested?()
                }
                return noErr
            },
            1,
            [eventSpec],
            Unmanaged.passUnretained(self).toOpaque(),
            &handlerRef
        )

        guard installStatus == noErr else {
            throw HotkeyMonitorError.hotkeyRegistrationFailed(
                "Unable to install global hotkey handler (OSStatus \(installStatus))"
            )
        }

        let descriptor = descriptor(for: binding)
        var ref: EventHotKeyRef?

        let registerStatus = RegisterEventHotKey(
            UInt32(descriptor.keyCode),
            descriptor.modifiers,
            EventHotKeyID(signature: Self.signature, id: descriptor.id),
            GetApplicationEventTarget(),
            0,
            &ref
        )

        guard registerStatus == noErr, let ref else {
            if let handlerRef {
                RemoveEventHandler(handlerRef)
                self.handlerRef = nil
            }
            throw HotkeyMonitorError.hotkeyRegistrationFailed(Self.message(for: registerStatus))
        }

        hotKeyRef = ref
    }

    public func stop() {
        if let hotKeyRef {
            UnregisterEventHotKey(hotKeyRef)
        }
        hotKeyRef = nil

        if let handlerRef {
            RemoveEventHandler(handlerRef)
        }
        handlerRef = nil
    }

    private func descriptor(for binding: GlobalHotkeyBinding) -> (keyCode: Int, modifiers: UInt32, id: UInt32) {
        switch binding {
        case .controlOptionSpace:
            return (49, UInt32(controlKey | optionKey), 1)
        case .controlOptionD:
            return (2, UInt32(controlKey | optionKey), 2)
        default:
            return (0, 0, 0)
        }
    }

    private static func message(for status: OSStatus) -> String {
        switch status {
        case noErr:
            return "Registered"
        case OSStatus(eventHotKeyExistsErr):
            return "That shortcut is already in use by another app"
        default:
            return "Registration failed (OSStatus \(status))"
        }
    }
}

private func fourCharCode(_ string: String) -> OSType {
    precondition(string.utf16.count == 4)
    return string.utf16.reduce(0) { ($0 << 8) + OSType($1) }
}

public final class NoOpHotkeyMonitor: HotkeyMonitoring {
    public var onToggleRequested: (@MainActor () -> Void)?

    public init() {}

    public func start() throws {}

    public func stop() {}
}
