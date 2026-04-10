import AppKit
import Carbon.HIToolbox
import LocalWispr
import SwiftUI

private final class ClickThroughHostingView<Content: View>: NSHostingView<Content> {
    override func acceptsFirstMouse(for event: NSEvent?) -> Bool {
        true
    }

    /// Avoid shrink-wrapping to SwiftUI’s ideal size so the root always fills the window;
    /// otherwise short tabs (Settings/History) make the sidebar shorter than Dashboard.
    override var intrinsicContentSize: NSSize {
        NSSize(width: NSView.noIntrinsicMetric, height: NSView.noIntrinsicMetric)
    }
}

private final class ClickThroughHostingController<Content: View>: NSViewController {
    private let hostingView: ClickThroughHostingView<Content>

    init(rootView: Content) {
        hostingView = ClickThroughHostingView(rootView: rootView)
        super.init(nibName: nil, bundle: nil)
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    override func loadView() {
        view = hostingView
        hostingView.setContentHuggingPriority(.defaultLow, for: .horizontal)
        hostingView.setContentHuggingPriority(.defaultLow, for: .vertical)
    }
}

@MainActor
final class ControlPanelWindowController: NSWindowController {
    static let shared = ControlPanelWindowController()

    private init() {
        let rootView = ControlPanelView(appState: AppState.shared)
        let hostingController = ClickThroughHostingController(rootView: rootView)

        let window = NSWindow(contentViewController: hostingController)
        window.title = "LocalWispr"
        window.setContentSize(NSSize(width: 1360, height: 900))
        window.minSize = NSSize(width: 1260, height: 800)
        window.styleMask = [.titled, .closable, .miniaturizable, .resizable, .fullSizeContentView]
        window.titleVisibility = .hidden
        window.titlebarAppearsTransparent = true
        window.isReleasedWhenClosed = false
        window.isMovableByWindowBackground = false
        window.level = .normal
        window.backgroundColor = .clear
        window.isOpaque = false
        window.center()
        window.collectionBehavior = [.moveToActiveSpace]
        window.tabbingMode = .disallowed

        super.init(window: window)
        shouldCascadeWindows = true
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    func showControlPanel() {
        guard let window else { return }
        if let frontmost = NSWorkspace.shared.frontmostApplication {
            AppContextCapture.noteActivatedApplication(frontmost)
        }
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }
}

final class LocalWisprHostAppDelegate: NSObject, NSApplicationDelegate {
    private var controlPanelObserver: NSObjectProtocol?
    private var didBecomeActiveObserver: NSObjectProtocol?
    private var workspaceActivationObserver: NSObjectProtocol?

    func applicationWillFinishLaunching(_ notification: Notification) {
        // LSUIElement apps have no Dock icon, so `applicationShouldHandleReopen` may never run.
        // Finder “open” while the app is already running sends kAEOpenApplication to the running process.
        NSAppleEventManager.shared().setEventHandler(
            self,
            andSelector: #selector(handleOpenApplicationEvent(_:withReplyEvent:)),
            forEventClass: AEEventClass(kCoreEventClass),
            andEventID: AEEventID(kAEOpenApplication)
        )
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        controlPanelObserver = NotificationCenter.default.addObserver(
            forName: .localWisprShowControlPanel,
            object: nil,
            queue: .main
        ) { _ in
            Task { @MainActor in
                ControlPanelWindowController.shared.showControlPanel()
            }
        }

        // Defer one turn so MenuBarExtra / SwiftUI runtime are ready before ordering the window front.
        Task { @MainActor in
            ControlPanelWindowController.shared.showControlPanel()
        }

        didBecomeActiveObserver = NotificationCenter.default.addObserver(
            forName: NSApplication.didBecomeActiveNotification,
            object: nil,
            queue: .main
        ) { _ in
            Task { @MainActor in
                AppState.shared.refreshAccessibilityAndHotkey()
            }
        }

        workspaceActivationObserver = NSWorkspace.shared.notificationCenter.addObserver(
            forName: NSWorkspace.didActivateApplicationNotification,
            object: nil,
            queue: .main
        ) { notification in
            guard let application = notification.userInfo?[NSWorkspace.applicationUserInfoKey] as? NSRunningApplication else {
                return
            }

            Task { @MainActor in
                AppContextCapture.noteActivatedApplication(application)
            }
        }
    }

    func applicationShouldHandleReopen(_ sender: NSApplication, hasVisibleWindows flag: Bool) -> Bool {
        Task { @MainActor in
            ControlPanelWindowController.shared.showControlPanel()
        }
        return true
    }

    @objc private func handleOpenApplicationEvent(
        _ event: NSAppleEventDescriptor?,
        withReplyEvent reply: NSAppleEventDescriptor?
    ) {
        Task { @MainActor in
            ControlPanelWindowController.shared.showControlPanel()
        }
    }

    func applicationWillTerminate(_ notification: Notification) {
        if let controlPanelObserver {
            NotificationCenter.default.removeObserver(controlPanelObserver)
            self.controlPanelObserver = nil
        }
        if let didBecomeActiveObserver {
            NotificationCenter.default.removeObserver(didBecomeActiveObserver)
            self.didBecomeActiveObserver = nil
        }
        if let workspaceActivationObserver {
            NSWorkspace.shared.notificationCenter.removeObserver(workspaceActivationObserver)
            self.workspaceActivationObserver = nil
        }
    }
}
