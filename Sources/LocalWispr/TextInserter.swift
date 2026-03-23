import ApplicationServices
import AppKit
import CoreGraphics
import Foundation

public enum TextInserterError: LocalizedError {
    case failedToWritePasteboard
    case failedToCreateEventSource
    case failedToCreateKeyboardEvent

    public var errorDescription: String? {
        switch self {
        case .failedToWritePasteboard:
            return "Failed to write cleaned text to pasteboard."
        case .failedToCreateEventSource:
            return "Failed to create CGEvent source for text insertion."
        case .failedToCreateKeyboardEvent:
            return "Failed to synthesize Cmd+V keyboard event."
        }
    }
}

private struct PasteboardSnapshot: Sendable {
    let items: [[NSPasteboard.PasteboardType: Data]]

    init(from pasteboard: NSPasteboard) {
        var captured: [[NSPasteboard.PasteboardType: Data]] = []
        for item in pasteboard.pasteboardItems ?? [] {
            var entry: [NSPasteboard.PasteboardType: Data] = [:]
            for type in item.types {
                if let data = item.data(forType: type) {
                    entry[type] = data
                }
            }
            captured.append(entry)
        }
        self.items = captured
    }

    func restore(to pasteboard: NSPasteboard) {
        pasteboard.clearContents()

        guard !items.isEmpty else { return }

        let restoredItems = items.map { itemMap -> NSPasteboardItem in
            let item = NSPasteboardItem()
            for (type, data) in itemMap {
                item.setData(data, forType: type)
            }
            return item
        }

        pasteboard.writeObjects(restoredItems)
    }
}

public final class TextInserter: @unchecked Sendable, Inserting {
    private let activationSettleDelayNanos: UInt64 = 60_000_000
    private let clipboardRestoreDelayNanos: UInt64 = 280_000_000

    private let stateLock = NSLock()
    private let hostBundleIdentifier = Bundle.main.bundleIdentifier
    private var lastExternalAppPID: pid_t?
    private var activationObserver: NSObjectProtocol?

    public init() {
        Task { @MainActor [weak self] in
            self?.installFrontmostAppTracking()
        }
    }

    deinit {
        let observer: NSObjectProtocol?
        stateLock.lock()
        observer = activationObserver
        activationObserver = nil
        stateLock.unlock()

        if let observer {
            NSWorkspace.shared.notificationCenter.removeObserver(observer)
        }
    }

    public func insertAtCursor(_ text: String) async throws {
        let cleaned = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !cleaned.isEmpty else { return }

        let target = await MainActor.run { () -> (pid: pid_t?, didActivate: Bool) in
            let pid = determinePasteTargetPID()
            let didActivate = activatePasteTargetIfNeeded(targetPID: pid)
            return (pid, didActivate)
        }

        if target.didActivate {
            try? await Task.sleep(nanoseconds: activationSettleDelayNanos)
        }

        let didInsertViaAX = await MainActor.run {
            attemptDirectAXInsertion(cleaned)
        }

        if didInsertViaAX {
            return
        }

        let writeResult = try await MainActor.run { () throws -> (snapshot: PasteboardSnapshot, injectedChangeCount: Int) in
            let pasteboard = NSPasteboard.general
            let snapshot = PasteboardSnapshot(from: pasteboard)

            pasteboard.clearContents()
            guard pasteboard.setString(cleaned, forType: .string) else {
                throw TextInserterError.failedToWritePasteboard
            }

            return (snapshot: snapshot, injectedChangeCount: pasteboard.changeCount)
        }

        try await MainActor.run {
            try postPasteShortcut(targetPID: target.pid)
        }

        let snapshot = writeResult.snapshot
        let injectedChangeCount = writeResult.injectedChangeCount
        let restoreDelay = clipboardRestoreDelayNanos

        Task.detached(priority: .utility) {
            try? await Task.sleep(nanoseconds: restoreDelay)
            await MainActor.run {
                let pasteboard = NSPasteboard.general
                if pasteboard.changeCount == injectedChangeCount {
                    snapshot.restore(to: pasteboard)
                }
            }
        }
    }

    @MainActor
    private func installFrontmostAppTracking() {
        guard activationObserver == nil else { return }

        if let current = NSWorkspace.shared.frontmostApplication {
            recordExternalIfNeeded(current)
        }

        let observer = NSWorkspace.shared.notificationCenter.addObserver(
            forName: NSWorkspace.didActivateApplicationNotification,
            object: NSWorkspace.shared,
            queue: nil
        ) { [weak self] notification in
            guard let app = notification.userInfo?[NSWorkspace.applicationUserInfoKey] as? NSRunningApplication else {
                return
            }
            self?.recordExternalIfNeeded(app)
        }

        stateLock.lock()
        activationObserver = observer
        stateLock.unlock()
    }

    private func recordExternalIfNeeded(_ app: NSRunningApplication) {
        guard !app.isTerminated else { return }
        if let hostBundleIdentifier, app.bundleIdentifier == hostBundleIdentifier {
            return
        }
        if app.bundleIdentifier == "com.apple.controlcenter" {
            return
        }

        stateLock.lock()
        lastExternalAppPID = app.processIdentifier
        stateLock.unlock()
    }

    @MainActor
    private func determinePasteTargetPID() -> pid_t? {
        if let frontmost = NSWorkspace.shared.frontmostApplication {
            let isHostFrontmost = frontmost.bundleIdentifier == hostBundleIdentifier
            let isControlCenterFrontmost = frontmost.bundleIdentifier == "com.apple.controlcenter"

            if !isHostFrontmost && !isControlCenterFrontmost {
                recordExternalIfNeeded(frontmost)
                return frontmost.processIdentifier
            }
        }

        let targetPID: pid_t?
        stateLock.lock()
        targetPID = lastExternalAppPID
        stateLock.unlock()

        return targetPID
    }

    @discardableResult
    @MainActor
    private func activatePasteTargetIfNeeded(targetPID: pid_t?) -> Bool {
        guard let targetPID else { return false }

        guard let targetApp = NSRunningApplication(processIdentifier: targetPID), !targetApp.isTerminated else {
            return false
        }

        guard !targetApp.isActive else { return false }
        return targetApp.activate()
    }

    private func postPasteShortcut(targetPID: pid_t?) throws {
        guard let source = CGEventSource(stateID: .hidSystemState) else {
            throw TextInserterError.failedToCreateEventSource
        }

        guard let keyDown = CGEvent(keyboardEventSource: source, virtualKey: 0x09, keyDown: true),
              let keyUp = CGEvent(keyboardEventSource: source, virtualKey: 0x09, keyDown: false) else {
            throw TextInserterError.failedToCreateKeyboardEvent
        }

        keyDown.flags = .maskCommand
        keyUp.flags = .maskCommand

        if let targetPID {
            keyDown.postToPid(targetPID)
            keyUp.postToPid(targetPID)
        } else {
            keyDown.post(tap: .cghidEventTap)
            keyUp.post(tap: .cghidEventTap)
        }
    }

    @MainActor
    private func attemptDirectAXInsertion(_ text: String) -> Bool {
        let systemWide = AXUIElementCreateSystemWide()

        var focusedRef: CFTypeRef?
        guard AXUIElementCopyAttributeValue(systemWide, kAXFocusedUIElementAttribute as CFString, &focusedRef) == .success,
              let focusedRef,
              CFGetTypeID(focusedRef) == AXUIElementGetTypeID() else {
            return false
        }

        let focusedElement = unsafeDowncast(focusedRef, to: AXUIElement.self)

        var valueRef: CFTypeRef?
        guard AXUIElementCopyAttributeValue(focusedElement, kAXValueAttribute as CFString, &valueRef) == .success,
              let currentValue = valueRef as? String else {
            return false
        }

        var insertionLocation = currentValue.utf16.count
        var selectionLength = 0

        var selectedRangeRef: CFTypeRef?
        if AXUIElementCopyAttributeValue(focusedElement, kAXSelectedTextRangeAttribute as CFString, &selectedRangeRef) == .success,
           let selectedRangeValue = selectedRangeRef {
            let axValue = unsafeDowncast(selectedRangeValue, to: AXValue.self)
            if AXValueGetType(axValue) == .cfRange {
                var selectedRange = CFRange()
                if AXValueGetValue(axValue, .cfRange, &selectedRange) {
                    insertionLocation = max(0, min(selectedRange.location, currentValue.utf16.count))
                    selectionLength = max(0, min(selectedRange.length, currentValue.utf16.count - insertionLocation))
                }
            }
        }

        let startIndex = String.Index(utf16Offset: insertionLocation, in: currentValue)
        let endIndex = String.Index(utf16Offset: insertionLocation + selectionLength, in: currentValue)
        let updatedValue = String(currentValue[..<startIndex]) + text + String(currentValue[endIndex...])

        guard AXUIElementSetAttributeValue(focusedElement, kAXValueAttribute as CFString, updatedValue as CFTypeRef) == .success else {
            return false
        }

        var postInsertRange = CFRange(location: insertionLocation + text.utf16.count, length: 0)
        if let postInsertAXRange = AXValueCreate(.cfRange, &postInsertRange) {
            _ = AXUIElementSetAttributeValue(focusedElement, kAXSelectedTextRangeAttribute as CFString, postInsertAXRange)
        }

        return true
    }
}
