import AppKit
import Darwin
import LocalWispr
import SwiftUI

@main
struct LocalWisprHostApp: App {
    @NSApplicationDelegateAdaptor(LocalWisprHostAppDelegate.self) private var appDelegate
    @StateObject private var appState = AppState.shared

    init() {
        Self.terminateDuplicateInstances()
        AppTheme.bootstrap()

        Task { @MainActor in
            AppState.shared.bootstrap()
        }
    }

    var body: some Scene {
        MenuBarExtra {
            MenuBarView(appState: appState)
        } label: {
            MenuBarStatusIconView(state: appState.state)
        }
        .menuBarExtraStyle(.window)
    }

    private static func terminateDuplicateInstances() {
        guard let bundleIdentifier = Bundle.main.bundleIdentifier else { return }

        let currentPID = ProcessInfo.processInfo.processIdentifier
        let duplicates = NSRunningApplication
            .runningApplications(withBundleIdentifier: bundleIdentifier)
            .filter { $0.processIdentifier != currentPID }

        for app in duplicates {
            if !app.terminate() {
                _ = app.forceTerminate()
            }
        }
    }
}
