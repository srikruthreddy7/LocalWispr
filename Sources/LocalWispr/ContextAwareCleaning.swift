import AppKit
import ApplicationServices
import Foundation
import OSLog

// MARK: - .env File Loader

enum DebugLog {
    private static let logPath = "/tmp/localwispr-debug.log"
    static func write(_ message: String) {
        let line = "[\(ISO8601DateFormatter().string(from: Date()))] \(message)\n"
        if let handle = FileHandle(forWritingAtPath: logPath) {
            handle.seekToEndOfFile()
            handle.write(Data(line.utf8))
            handle.closeFile()
        } else {
            FileManager.default.createFile(atPath: logPath, contents: Data(line.utf8))
        }
    }
}

enum DotEnv {
    /// Returns process environment merged with values from `.env` files.
    /// App-owned `LOCALWISPR_*` settings prefer checked-in/local file values so GUI launches
    /// are not accidentally overridden by ambient shell or launchd environment state.
    static func merged() -> [String: String] {
        let fileValues = load()
        var env = ProcessInfo.processInfo.environment

        for (key, value) in fileValues {
            if key.hasPrefix("LOCALWISPR_") || env[key] == nil {
                env[key] = value
            }
        }
        return env
    }

    /// Parses `.env` files from the app's working directory and the project's source root.
    private static func load() -> [String: String] {
        var values: [String: String] = [:]
        var candidates: [String] = [
            FileManager.default.currentDirectoryPath,
            Bundle.main.bundleURL.deletingLastPathComponent().path,
        ]
        if let sourceFileRoot = sourceFileRoot() { candidates.append(sourceFileRoot) }
        if let root = sourceRoot() { candidates.append(root) }
        for dir in Array(NSOrderedSet(array: candidates)) as? [String] ?? candidates {
            let path = (dir as NSString).appendingPathComponent(".env")
            guard let contents = try? String(contentsOfFile: path, encoding: .utf8) else { continue }
            for (key, value) in parse(contents) where values[key] == nil {
                values[key] = value
            }
        }
        return values
    }

    private static func sourceFileRoot() -> String? {
        locatePackageRoot(startingAt: (#filePath as NSString).deletingLastPathComponent)
    }

    private static func parse(_ contents: String) -> [(String, String)] {
        contents.components(separatedBy: .newlines).compactMap { line in
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            guard !trimmed.isEmpty, !trimmed.hasPrefix("#") else { return nil }
            guard let eqIndex = trimmed.firstIndex(of: "=") else { return nil }
            let key = String(trimmed[trimmed.startIndex..<eqIndex]).trimmingCharacters(in: .whitespaces)
            var value = String(trimmed[trimmed.index(after: eqIndex)...]).trimmingCharacters(in: .whitespaces)
            // Strip surrounding quotes
            if (value.hasPrefix("\"") && value.hasSuffix("\"")) || (value.hasPrefix("'") && value.hasSuffix("'")) {
                value = String(value.dropFirst().dropLast())
            }
            guard !key.isEmpty else { return nil }
            return (key, value)
        }
    }

    private static func sourceRoot() -> String? {
        locatePackageRoot(startingAt: Bundle.main.bundleURL.deletingLastPathComponent().path)
    }

    private static func locatePackageRoot(startingAt startPath: String) -> String? {
        var dir = startPath
        for _ in 0..<10 {
            let marker = (dir as NSString).appendingPathComponent("Package.swift")
            if FileManager.default.fileExists(atPath: marker) { return dir }
            let parent = (dir as NSString).deletingLastPathComponent
            if parent == dir { break }
            dir = parent
        }
        return nil
    }
}

enum ContextUsagePolicy {
    static func isEnabled(environment: [String: String]) -> Bool {
        guard let rawValue = environment["LOCALWISPR_DISABLE_CONTEXT"] else {
            return true
        }

        switch rawValue.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
        case "1", "true", "yes", "on":
            return false
        default:
            return true
        }
    }
}

public actor SessionContextStore {
    public static let shared = SessionContextStore()

    private var currentContext: DictationAppContext?

    public func set(_ context: DictationAppContext?) {
        currentContext = context
    }

    public func get() -> DictationAppContext? {
        currentContext
    }

    public func clear() {
        currentContext = nil
    }
}

public struct DictationAppContext: Sendable, Equatable {
    public enum Surface: String, Sendable, Equatable {
        case ide
        case terminal
        case browser
        case other
    }

    public let capturedAt: Date
    public let appName: String
    public let bundleIdentifier: String
    public let windowTitle: String
    public let surface: Surface
    public let projectName: String?
    public let projectPathHint: String?
    public let activeDocumentHint: String?
    public let browserTabHint: String?
    public let browserURL: String?
    public let browserHost: String?
    public let browserPathHint: String?

    public init(
        capturedAt: Date = Date(),
        appName: String,
        bundleIdentifier: String,
        windowTitle: String,
        surface: Surface,
        projectName: String? = nil,
        projectPathHint: String? = nil,
        activeDocumentHint: String? = nil,
        browserTabHint: String? = nil,
        browserURL: String? = nil,
        browserHost: String? = nil,
        browserPathHint: String? = nil
    ) {
        self.capturedAt = capturedAt
        self.appName = appName
        self.bundleIdentifier = bundleIdentifier
        self.windowTitle = windowTitle
        self.surface = surface
        self.projectName = projectName?.trimmingCharacters(in: .whitespacesAndNewlines)
        self.projectPathHint = projectPathHint?.trimmingCharacters(in: .whitespacesAndNewlines)
        self.activeDocumentHint = activeDocumentHint?.trimmingCharacters(in: .whitespacesAndNewlines)
        self.browserTabHint = browserTabHint?.trimmingCharacters(in: .whitespacesAndNewlines)
        self.browserURL = browserURL?.trimmingCharacters(in: .whitespacesAndNewlines)
        self.browserHost = browserHost?.trimmingCharacters(in: .whitespacesAndNewlines)
        self.browserPathHint = browserPathHint?.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    public static let unknown = DictationAppContext(
        appName: "Unknown",
        bundleIdentifier: "unknown",
        windowTitle: "",
        surface: .other
    )
}

public enum AppContextCapture {
    @MainActor private static var lastExternalContext: DictationAppContext?
    @MainActor private static var lastBrowserContext: DictationAppContext?

    @MainActor
    public static func captureFrontmost() -> DictationAppContext {
        guard let frontmost = NSWorkspace.shared.frontmostApplication else {
            return lastExternalContext ?? .unknown
        }

        let context = capture(application: frontmost)
        let effectiveContext = effectiveContext(frontmost: context, rememberedExternal: lastExternalContext)

        if !shouldPreferRememberedExternalContext(for: context) {
            lastExternalContext = context
        }

        if effectiveContext != context {
            DebugLog.write("[Context] using remembered \(debugSummary(for: effectiveContext)) instead of LocalWispr")
        } else {
            DebugLog.write("[Context] frontmost \(debugSummary(for: effectiveContext))")
        }

        return effectiveContext
    }

    @MainActor
    public static func captureForDictationStart() -> DictationAppContext {
        guard let frontmost = NSWorkspace.shared.frontmostApplication else {
            return lastExternalContext ?? .unknown
        }

        let frontmostContext = capture(application: frontmost)
        if shouldPreferRememberedExternalContext(for: frontmostContext),
           let liveBrowserContext = preferredRunningBrowserContextIfPossible() {
            lastExternalContext = liveBrowserContext
            lastBrowserContext = liveBrowserContext
            DebugLog.write("[Context] dictation-start using live-browser \(debugSummary(for: liveBrowserContext)) instead of LocalWispr")
            return liveBrowserContext
        }

        if shouldPreferRememberedExternalContext(for: frontmostContext),
           let refreshedRemembered = refreshedRememberedExternalContextIfPossible() {
            lastExternalContext = refreshedRemembered
            DebugLog.write("[Context] dictation-start using remembered \(debugSummary(for: refreshedRemembered)) instead of LocalWispr")
            return refreshedRemembered
        }

        if shouldPreferRememberedExternalContext(for: frontmostContext),
           let refreshedBrowser = refreshedRecentBrowserContextIfPossible() {
            lastExternalContext = refreshedBrowser
            DebugLog.write("[Context] dictation-start using recent-browser \(debugSummary(for: refreshedBrowser)) instead of LocalWispr")
            return refreshedBrowser
        }

        if !shouldPreferRememberedExternalContext(for: frontmostContext) {
            lastExternalContext = frontmostContext
            if frontmostContext.surface == .browser {
                lastBrowserContext = frontmostContext
            }
        }

        DebugLog.write("[Context] dictation-start frontmost \(debugSummary(for: frontmostContext))")
        return frontmostContext
    }

    @MainActor
    public static func noteActivatedApplication(_ application: NSRunningApplication) {
        let context = capture(application: application)
        guard !shouldPreferRememberedExternalContext(for: context) else { return }
        lastExternalContext = context
        if context.surface == .browser {
            lastBrowserContext = context
        }
        DebugLog.write("[Context] remembered \(debugSummary(for: context))")
    }

    static func effectiveContext(frontmost: DictationAppContext, rememberedExternal: DictationAppContext?) -> DictationAppContext {
        if shouldPreferRememberedExternalContext(for: frontmost), let rememberedExternal {
            return rememberedExternal
        }
        return frontmost
    }

    @MainActor
    private static func capture(application: NSRunningApplication) -> DictationAppContext {
        let bundleID = application.bundleIdentifier ?? "unknown"
        let appName = application.localizedName ?? bundleID
        let browserSnapshot = BrowserSnapshotProvider.snapshot(for: application)
        let title = browserSnapshot?.title ?? focusedWindowTitle(for: application.processIdentifier) ?? ""
        return DictationContextParser.parse(
            appName: appName,
            bundleIdentifier: bundleID,
            windowTitle: title,
            browserURL: browserSnapshot?.url
        )
    }

    private static func focusedWindowTitle(for pid: pid_t) -> String? {
        let appElement = AXUIElementCreateApplication(pid)
        var focusedWindowRef: CFTypeRef?
        guard AXUIElementCopyAttributeValue(appElement, kAXFocusedWindowAttribute as CFString, &focusedWindowRef) == .success,
              let focusedWindowRef,
              CFGetTypeID(focusedWindowRef) == AXUIElementGetTypeID() else {
            return nil
        }

        let focusedWindow = unsafeDowncast(focusedWindowRef, to: AXUIElement.self)
        var titleRef: CFTypeRef?
        guard AXUIElementCopyAttributeValue(focusedWindow, kAXTitleAttribute as CFString, &titleRef) == .success else {
            return nil
        }

        return titleRef as? String
    }

    private static func shouldPreferRememberedExternalContext(for context: DictationAppContext) -> Bool {
        context.bundleIdentifier == "com.localwispr.host" || context.appName == "LocalWispr"
    }

    @MainActor
    private static func refreshedRememberedExternalContextIfPossible() -> DictationAppContext? {
        guard let remembered = lastExternalContext else { return nil }
        guard remembered.surface == .browser else { return remembered }

        guard let application = NSRunningApplication
            .runningApplications(withBundleIdentifier: remembered.bundleIdentifier)
            .first(where: { !$0.isTerminated }) else {
            return remembered
        }

        let refreshed = capture(application: application)
        return refreshed.bundleIdentifier == remembered.bundleIdentifier ? refreshed : remembered
    }

    @MainActor
    private static func refreshedRecentBrowserContextIfPossible(maxAgeSeconds: TimeInterval = 12) -> DictationAppContext? {
        guard let rememberedBrowser = lastBrowserContext else { return nil }
        guard Date().timeIntervalSince(rememberedBrowser.capturedAt) <= maxAgeSeconds else { return nil }

        guard let application = NSRunningApplication
            .runningApplications(withBundleIdentifier: rememberedBrowser.bundleIdentifier)
            .first(where: { !$0.isTerminated }) else {
            return rememberedBrowser
        }

        let refreshed = capture(application: application)
        let candidate = refreshed.bundleIdentifier == rememberedBrowser.bundleIdentifier ? refreshed : rememberedBrowser
        return isUsefulBrowserContext(candidate) ? candidate : nil
    }

    @MainActor
    private static func preferredRunningBrowserContextIfPossible() -> DictationAppContext? {
        let preferredBundleIdentifiers: [String] = {
            if let recentBundleIdentifier = lastBrowserContext?.bundleIdentifier {
                return [recentBundleIdentifier, "com.google.Chrome", "com.brave.Browser", "com.microsoft.edgemac", "org.chromium.Chromium"]
            }
            return ["com.google.Chrome", "com.brave.Browser", "com.microsoft.edgemac", "org.chromium.Chromium"]
        }()

        for bundleIdentifier in preferredBundleIdentifiers {
            guard let application = NSRunningApplication
                .runningApplications(withBundleIdentifier: bundleIdentifier)
                .first(where: { !$0.isTerminated }) else {
                continue
            }

            let candidate = capture(application: application)
            if isUsefulBrowserContext(candidate) {
                return candidate
            }
        }

        return nil
    }

    private static func isUsefulBrowserContext(_ context: DictationAppContext) -> Bool {
        guard context.surface == .browser else { return false }
        guard let host = context.browserHost?.lowercased(), !host.isEmpty else { return false }
        if host == "127.0.0.1" || host == "localhost" {
            return false
        }
        return true
    }

    private static func debugSummary(for context: DictationAppContext) -> String {
        [
            "app=\(context.appName)",
            "surface=\(context.surface.rawValue)",
            "project=\(context.projectName ?? "nil")",
            "file=\(context.activeDocumentHint ?? "nil")",
            "browserTab=\(context.browserTabHint ?? "nil")",
            "browserURL=\(context.browserURL ?? "nil")",
            "browserHost=\(context.browserHost ?? "nil")",
            "browserPath=\(context.browserPathHint ?? "nil")"
        ].joined(separator: " ")
    }
}

private struct BrowserSnapshot: Sendable {
    let title: String
    let url: String
}

private enum BrowserSnapshotProvider {
    private static let chromeFamilyApplications: [String: String] = [
        "com.google.Chrome": "Google Chrome",
        "com.brave.Browser": "Brave Browser",
        "com.microsoft.edgemac": "Microsoft Edge",
        "org.chromium.Chromium": "Chromium",
    ]

    static func snapshot(for application: NSRunningApplication) -> BrowserSnapshot? {
        guard let bundleIdentifier = application.bundleIdentifier,
              let applicationName = chromeFamilyApplications[bundleIdentifier] else {
            return nil
        }

        return runChromeFamilyAppleScript(applicationName: applicationName)
    }

    private static func runChromeFamilyAppleScript(applicationName: String) -> BrowserSnapshot? {
        let separator = "<<LOCALWISPR_BROWSER_SEPARATOR>>"
        let script = """
        tell application "\(applicationName)"
            if not (exists front window) then return ""
            set tabTitle to title of active tab of front window
            set tabURL to URL of active tab of front window
            return tabTitle & "\(separator)" & tabURL
        end tell
        """

        let process = Process()
        let outputPipe = Pipe()
        let errorPipe = Pipe()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/osascript")
        process.arguments = ["-e", script]
        process.standardOutput = outputPipe
        process.standardError = errorPipe

        do {
            try process.run()
            process.waitUntilExit()
        } catch {
            DebugLog.write("[BrowserContext] osascript launch failed for \(applicationName): \(error.localizedDescription)")
            return nil
        }

        guard process.terminationStatus == 0 else {
            let errorData = errorPipe.fileHandleForReading.readDataToEndOfFile()
            let message = String(data: errorData, encoding: .utf8)?
                .trimmingCharacters(in: .whitespacesAndNewlines) ?? "unknown error"
            DebugLog.write("[BrowserContext] osascript failed for \(applicationName): \(message)")
            return nil
        }

        let outputData = outputPipe.fileHandleForReading.readDataToEndOfFile()
        let output = String(data: outputData, encoding: .utf8)?
            .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        guard !output.isEmpty else { return nil }

        let parts = output.components(separatedBy: separator)
        guard parts.count == 2 else { return nil }

        let title = parts[0].trimmingCharacters(in: .whitespacesAndNewlines)
        let url = parts[1].trimmingCharacters(in: .whitespacesAndNewlines)
        guard !url.isEmpty else { return nil }

        DebugLog.write("[BrowserContext] app=\(applicationName) title=\(title) url=\(url)")
        return BrowserSnapshot(title: title, url: url)
    }
}

public enum DictationContextParser {
    private static let ideBundleIDs: Set<String> = [
        "com.microsoft.VSCode",
        "com.todesktop.230313mzl4w4u92",
        "com.jetbrains.intellij",
        "com.jetbrains.pycharm",
        "com.jetbrains.webstorm",
        "com.apple.dt.Xcode"
    ]

    private static let terminalBundleIDs: Set<String> = [
        "com.apple.Terminal",
        "com.googlecode.iterm2",
        "dev.warp.Warp-Stable",
        "com.mitchellh.ghostty"
    ]

    private static let browserBundleIDs: Set<String> = [
        "com.google.Chrome",
        "com.apple.Safari",
        "org.mozilla.firefox",
        "com.brave.Browser",
        "com.microsoft.edgemac"
    ]

    public static func parse(
        appName: String,
        bundleIdentifier: String,
        windowTitle: String,
        browserURL: String? = nil
    ) -> DictationAppContext {
        if browserBundleIDs.contains(bundleIdentifier) {
            return parseBrowser(
                appName: appName,
                bundleIdentifier: bundleIdentifier,
                windowTitle: windowTitle,
                browserURL: browserURL
            )
        }

        if ideBundleIDs.contains(bundleIdentifier) || appName.contains("Code") || appName.contains("Cursor") {
            return parseIDE(appName: appName, bundleIdentifier: bundleIdentifier, windowTitle: windowTitle)
        }

        if terminalBundleIDs.contains(bundleIdentifier) || appName.localizedCaseInsensitiveContains("terminal") {
            return parseTerminal(appName: appName, bundleIdentifier: bundleIdentifier, windowTitle: windowTitle)
        }

        return DictationAppContext(appName: appName, bundleIdentifier: bundleIdentifier, windowTitle: windowTitle, surface: .other)
    }

    private static func parseIDE(appName: String, bundleIdentifier: String, windowTitle: String) -> DictationAppContext {
        let chunks = windowTitle.split(separator: "—").map(String.init)
        let fallbackChunks = windowTitle.split(separator: "-").map { $0.trimmingCharacters(in: .whitespaces) }
        let parts = chunks.count > 1 ? chunks : fallbackChunks

        let projectName = parts.count >= 2 ? parts[parts.count - 2].trimmingCharacters(in: .whitespaces) : nil
        let fileName = parts.first?.trimmingCharacters(in: .whitespaces)
        let pathHint = extractPathCandidate(from: windowTitle)

        return DictationAppContext(
            appName: appName,
            bundleIdentifier: bundleIdentifier,
            windowTitle: windowTitle,
            surface: .ide,
            projectName: projectName,
            projectPathHint: pathHint,
            activeDocumentHint: fileName
        )
    }

    private static func parseTerminal(appName: String, bundleIdentifier: String, windowTitle: String) -> DictationAppContext {
        let pathHint = extractPathCandidate(from: windowTitle)

        return DictationAppContext(
            appName: appName,
            bundleIdentifier: bundleIdentifier,
            windowTitle: windowTitle,
            surface: .terminal,
            projectName: pathHint.flatMap { URL(fileURLWithPath: $0).lastPathComponent },
            projectPathHint: pathHint,
            activeDocumentHint: nil
        )
    }

    private static func parseBrowser(
        appName: String,
        bundleIdentifier: String,
        windowTitle: String,
        browserURL: String?
    ) -> DictationAppContext {
        let stripped = windowTitle
            .replacingOccurrences(of: " - Google Chrome", with: "")
            .replacingOccurrences(of: " - Safari", with: "")
            .replacingOccurrences(of: " - Mozilla Firefox", with: "")
            .replacingOccurrences(of: " - Brave", with: "")
            .replacingOccurrences(of: " - Microsoft Edge", with: "")
            .trimmingCharacters(in: .whitespacesAndNewlines)

        let components = browserURL.flatMap(URLComponents.init(string:))
        let titleDerivedHost = extractDomainCandidate(from: stripped)
        let host = components?.host?.lowercased() ?? titleDerivedHost
        let path = components?.percentEncodedPath.removingPercentEncoding
        let pathSegments = (path ?? "")
            .split(separator: "/")
            .map(String.init)
            .filter { !$0.isEmpty }

        var projectName: String?
        var activeDocumentHint: String?
        if host == "github.com" || host == "www.github.com" {
            if pathSegments.count >= 2 {
                projectName = pathSegments[1]
            }
            if let blobIndex = pathSegments.firstIndex(of: "blob"), blobIndex + 2 < pathSegments.count {
                activeDocumentHint = pathSegments.last
            }
        }

        return DictationAppContext(
            appName: appName,
            bundleIdentifier: bundleIdentifier,
            windowTitle: windowTitle,
            surface: .browser,
            projectName: projectName,
            activeDocumentHint: activeDocumentHint,
            browserTabHint: stripped.isEmpty ? nil : stripped,
            browserURL: browserURL,
            browserHost: host,
            browserPathHint: path
        )
    }

    private static func extractDomainCandidate(from text: String) -> String? {
        guard let match = text.range(
            of: #"\b(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}\b"#,
            options: .regularExpression
        ) else {
            return nil
        }

        let candidate = String(text[match]).lowercased()
        guard candidate != "localhost", candidate != "127.0.0.1" else {
            return nil
        }
        return candidate
    }

    private static func extractPathCandidate(from text: String) -> String? {
        guard let match = text.range(of: #"(/Users/[^\s\]\)]+(?:/[^\s\]\)]+)*)"#, options: .regularExpression) else {
            return nil
        }
        return String(text[match])
    }
}

public actor ProjectIdentifierIndex {
    private struct CacheEntry {
        let scannedAt: Date
        let identifiers: [String]
    }

    private var projectCache: [String: CacheEntry] = [:]
    private let logger = Logger(subsystem: "LocalWispr", category: "ContextIndex")

    public init() {}

    /// Returns identifiers ordered by relevance: active file > sibling files > project-wide.
    public func tieredIdentifiers(context: DictationAppContext, limit: Int = 120) -> [String] {
        if context.surface == .browser {
            let hints = Self.browserHints(for: context, limit: limit)
            logger.info("Browser index: hints=\(hints.count)")
            return hints
        }

        let projectRoot = resolveProjectRoot(context: context)

        // Tier 1: Active file identifiers (highest priority)
        var tier1: Set<String> = []
        if let activeFile = resolveActiveFilePath(context: context, projectRoot: projectRoot) {
            if let text = try? String(contentsOfFile: activeFile, encoding: .utf8), text.count <= 350_000 {
                tier1 = Self.extractIdentifiers(from: text)
            }
        }

        // Tier 2: Sibling files in same directory
        var tier2: Set<String> = []
        if let activeFile = resolveActiveFilePath(context: context, projectRoot: projectRoot) {
            let dir = (activeFile as NSString).deletingLastPathComponent
            tier2 = Self.scanDirectory(dir, excluding: activeFile)
        }

        // Tier 3: Project-wide (cached, frequency-sorted)
        var tier3: [String] = []
        if let projectRoot {
            tier3 = cachedProjectIdentifiers(rootPath: projectRoot, limit: max(limit, 200))
        }

        // Merge with tier priority: tier1 first, then tier2 (minus tier1), then tier3 (minus tier1+2)
        var result: [String] = []
        var seen = Set<String>()

        for id in tier1.sorted() {
            if seen.insert(id).inserted { result.append(id) }
        }
        for id in tier2.sorted() {
            if seen.insert(id).inserted { result.append(id) }
        }
        for id in tier3 {
            if seen.insert(id).inserted { result.append(id) }
        }

        logger.info("Tiered index: tier1=\(tier1.count) tier2=\(tier2.count) tier3=\(tier3.count) total=\(result.prefix(limit).count)")
        return Array(result.prefix(limit))
    }

    /// Legacy single-path API for backward compatibility.
    public func identifiers(forProjectPath path: String?, limit: Int = 80) -> [String] {
        guard let path, !path.isEmpty else { return [] }
        return cachedProjectIdentifiers(rootPath: path, limit: limit)
    }

    // MARK: - Project Root Detection

    static func findProjectRoot(from path: String) -> String? {
        let markers: Set<String> = [".git", "Package.swift", "package.json", "Cargo.toml", "pyproject.toml", "go.mod", ".xcodeproj", ".xcworkspace"]
        var current = (path as NSString).deletingLastPathComponent
        let fm = FileManager.default
        while current != "/" && !current.isEmpty {
            for marker in markers {
                let candidate = (current as NSString).appendingPathComponent(marker)
                if fm.fileExists(atPath: candidate) {
                    return current
                }
            }
            current = (current as NSString).deletingLastPathComponent
        }
        return nil
    }

    private func resolveProjectRoot(context: DictationAppContext) -> String? {
        // Try projectPathHint first (e.g. from JetBrains [/path/to/project] in title)
        if let hint = context.projectPathHint, FileManager.default.fileExists(atPath: hint) {
            return Self.findProjectRoot(from: hint + "/dummy") ?? hint
        }
        // Try resolving from activeDocumentHint if it looks like an absolute path
        if let doc = context.activeDocumentHint, doc.hasPrefix("/") {
            return Self.findProjectRoot(from: doc)
        }
        // For IDEs: try common workspace paths matching projectName
        if let projectName = context.projectName, context.surface == .ide {
            let home = NSHomeDirectory()
            let candidates = [
                "\(home)/Developer/\(projectName)",
                "\(home)/Projects/\(projectName)",
                "\(home)/Code/\(projectName)",
                "\(home)/Documents/\(projectName)",
                "\(home)/\(projectName)",
            ]
            for candidate in candidates {
                if FileManager.default.fileExists(atPath: candidate) {
                    return candidate
                }
            }
        }
        return nil
    }

    private func resolveActiveFilePath(context: DictationAppContext, projectRoot: String?) -> String? {
        guard let fileName = context.activeDocumentHint, !fileName.isEmpty else { return nil }

        // Already absolute
        if fileName.hasPrefix("/"), FileManager.default.fileExists(atPath: fileName) {
            return fileName
        }

        // Relative to project root
        guard let root = projectRoot else { return nil }
        let direct = (root as NSString).appendingPathComponent(fileName)
        if FileManager.default.fileExists(atPath: direct) {
            return direct
        }

        // Quick search in project for the filename
        let fm = FileManager.default
        guard let enumerator = fm.enumerator(at: URL(fileURLWithPath: root), includingPropertiesForKeys: nil, options: [.skipsHiddenFiles]) else {
            return nil
        }
        let baseName = (fileName as NSString).lastPathComponent
        var checked = 0
        while let url = enumerator.nextObject() as? URL {
            checked += 1
            if checked > 2000 { break }
            let p = url.path
            if p.contains("/.git/") || p.contains("/node_modules/") || p.contains("/.build/") { continue }
            if (p as NSString).lastPathComponent == baseName {
                return p
            }
        }
        return nil
    }

    // MARK: - Scanning

    private static func scanDirectory(_ dir: String, excluding: String? = nil) -> Set<String> {
        let fm = FileManager.default
        guard let contents = try? fm.contentsOfDirectory(atPath: dir) else { return [] }
        var identifiers = Set<String>()
        for name in contents {
            let full = (dir as NSString).appendingPathComponent(name)
            if full == excluding { continue }
            guard isCodeFile(path: full),
                  let text = try? String(contentsOfFile: full, encoding: .utf8),
                  text.count <= 350_000 else { continue }
            identifiers.formUnion(extractIdentifiers(from: text))
        }
        return identifiers
    }

    private func cachedProjectIdentifiers(rootPath: String, limit: Int) -> [String] {
        if let cached = projectCache[rootPath], Date().timeIntervalSince(cached.scannedAt) < 60 {
            return Array(cached.identifiers.prefix(limit))
        }

        let extracted = Self.scanProjectIdentifiers(rootPath: rootPath, limit: max(limit, 200))
        projectCache[rootPath] = CacheEntry(scannedAt: Date(), identifiers: extracted)
        return Array(extracted.prefix(limit))
    }

    static func scanProjectIdentifiers(rootPath: String, limit: Int) -> [String] {
        let fileManager = FileManager.default
        guard let enumerator = fileManager.enumerator(at: URL(fileURLWithPath: rootPath), includingPropertiesForKeys: [.isRegularFileKey], options: [.skipsHiddenFiles]) else {
            return []
        }

        var frequencies: [String: Int] = [:]
        var visitedFiles = 0

        while let url = enumerator.nextObject() as? URL {
            if visitedFiles >= 600 { break }
            let path = url.path
            if path.contains("/.git/") || path.contains("/node_modules/") || path.contains("/.build/") {
                continue
            }

            guard isCodeFile(path: path),
                  let text = try? String(contentsOf: url, encoding: .utf8),
                  text.count <= 350_000 else {
                continue
            }
            visitedFiles += 1
            for identifier in extractIdentifiers(from: text) {
                frequencies[identifier, default: 0] += 1
            }
        }

        return frequencies
            .sorted { lhs, rhs in
                if lhs.value == rhs.value { return lhs.key < rhs.key }
                return lhs.value > rhs.value
            }
            .prefix(limit)
            .map(\.key)
    }

    static func extractIdentifiers(from text: String) -> Set<String> {
        let pattern = #"\b([A-Z][a-zA-Z0-9]{2,}|[a-z][a-z0-9]+(?:[A-Z][a-zA-Z0-9]+)+|[a-z][a-z0-9]*(?:_[a-z0-9]+)+|[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+)\b"#
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return [] }
        let range = NSRange(text.startIndex..<text.endIndex, in: text)
        let matches = regex.matches(in: text, options: [], range: range)

        var values = Set<String>()
        for match in matches {
            guard let groupRange = Range(match.range(at: 1), in: text) else { continue }
            let candidate = String(text[groupRange])
            if candidate.count >= 3 && candidate.count <= 80 {
                values.insert(candidate)
            }
        }
        return values
    }

    static func isCodeFile(path: String) -> Bool {
        let allowed: Set<String> = ["swift", "py", "js", "ts", "tsx", "jsx", "java", "kt", "go", "rs", "c", "cc", "cpp", "h", "hpp", "cs", "rb", "php", "md", "json", "yaml", "yml", "toml"]
        guard let ext = path.split(separator: ".").last else { return false }
        return allowed.contains(String(ext).lowercased())
    }

    private static func browserHints(for context: DictationAppContext, limit: Int) -> [String] {
        var result: [String] = []
        var seen = Set<String>()

        func append(_ candidate: String?) {
            guard let candidate else { return }
            let trimmed = candidate.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { return }
            guard seen.insert(trimmed.lowercased()).inserted else { return }
            result.append(trimmed)
        }

        func append(contentsOf candidates: [String]) {
            for candidate in candidates {
                append(candidate)
                if result.count >= limit { return }
            }
        }

        let components = context.browserURL.flatMap(URLComponents.init(string:))
        let host = context.browserHost ?? components?.host?.lowercased()
        let canonicalHost: String? = host.map { rawHost in
            rawHost.hasPrefix("www.") ? String(rawHost.dropFirst(4)) : rawHost
        }
        let pathSegments = (components?.percentEncodedPath.removingPercentEncoding ?? context.browserPathHint ?? "")
            .split(separator: "/")
            .map(String.init)
            .filter { !$0.isEmpty }

        if let canonicalHost {
            append(canonicalHost)
            append(contentsOf: browserTokens(from: canonicalHost))

            if let primaryHostToken = canonicalHost.split(separator: ".").first.map(String.init), primaryHostToken.count >= 3 {
                append(primaryHostToken.uppercased())
                append(primaryHostToken)
            }

            switch canonicalHost {
            case "github.com", "www.github.com":
                if pathSegments.count >= 2 {
                    append(pathSegments[0])
                    append(pathSegments[1])
                    append("\(pathSegments[0])/\(pathSegments[1])")
                }
                if let blobIndex = pathSegments.firstIndex(of: "blob"), blobIndex + 2 < pathSegments.count {
                    append(pathSegments.last)
                    append(contentsOf: browserTokens(from: Array(pathSegments[(blobIndex + 2)...]).joined(separator: "/")))
                }
            case "linear.app", "www.linear.app":
                append(contentsOf: browserTokens(from: pathSegments.joined(separator: " ")))
            default:
                append(contentsOf: browserTokens(from: host))
                append(contentsOf: browserTokens(from: pathSegments.joined(separator: " ")))
            }
        }

        if let queryItems = components?.queryItems {
            for item in queryItems where item.name.lowercased() == "q" || item.name.lowercased() == "query" {
                append(contentsOf: browserTokens(from: item.value))
            }
        }

        append(contentsOf: browserTitleTokens(from: context.browserTabHint, canonicalHost: canonicalHost))

        return Array(result.prefix(limit))
    }

    private static func browserTitleTokens(from title: String?, canonicalHost: String?) -> [String] {
        guard let title, !title.isEmpty else { return [] }

        let hostTokens = Set(browserTokens(from: canonicalHost).map { $0.lowercased() })
        return browserTokens(from: title).filter { isUsefulBrowserTitleToken($0, hostTokens: hostTokens) }
    }

    private static func browserTokens(from text: String?) -> [String] {
        guard let text, !text.isEmpty else { return [] }

        let delimiters = CharacterSet.alphanumerics.inverted.subtracting(CharacterSet(charactersIn: "_-./#"))
        let rawTokens = text
            .components(separatedBy: delimiters)
            .flatMap { token -> [String] in
                token
                    .split(separator: "/")
                    .flatMap { $0.split(separator: "-") }
                    .flatMap { $0.split(separator: "_") }
                    .flatMap { $0.split(separator: ".") }
                    .map(String.init)
            }

        return rawTokens.filter(Self.isUsefulBrowserToken)
    }

    private static func isUsefulBrowserToken(_ token: String) -> Bool {
        let trimmed = token.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.count >= 3, trimmed.count <= 80 else { return false }

        let lowercase = trimmed.lowercased()
        if browserStopWords.contains(lowercase) {
            return false
        }

        if trimmed.rangeOfCharacter(from: .decimalDigits) != nil {
            return true
        }

        if trimmed.contains(".") || trimmed.contains("#") {
            return true
        }

        if trimmed.first?.isUppercase == true || trimmed.dropFirst().contains(where: \.isUppercase) {
            return true
        }

        if trimmed.contains("_") || trimmed.contains("-") {
            return true
        }

        return lowercase.count >= 4
    }

    private static func isUsefulBrowserTitleToken(_ token: String, hostTokens: Set<String>) -> Bool {
        guard isUsefulBrowserToken(token) else { return false }

        let lowercase = token.lowercased()
        if hostTokens.contains(lowercase) {
            return true
        }

        guard token.rangeOfCharacter(from: .decimalDigits) == nil else {
            return false
        }

        let lettersOnly = String(token.filter(\.isLetter))
        if lettersOnly.count >= 4, lettersOnly == lettersOnly.uppercased() {
            return true
        }

        if token.dropFirst().contains(where: \.isUppercase) {
            return true
        }

        if token.contains("_") || token.contains("-") {
            return true
        }

        return false
    }

    private static let browserStopWords: Set<String> = [
        "about", "account", "accounts", "all", "app", "apps", "article", "articles", "browser",
        "chrome", "code", "community", "dashboard", "developer", "developers", "docs", "documentation",
        "edge", "file", "files", "forum", "forums", "google", "home", "index", "issue", "issues",
        "learn", "login", "page", "pages", "pull", "request", "results", "safari", "search",
        "settings", "sign", "source", "support", "tab", "the", "view", "window", "www"
    ]
}

public final class ContextAwareCloudCleaner: @unchecked Sendable, Cleaning {
    private struct ProviderConfig {
        let name: String
        let endpoint: URL
        let model: String
        let apiKey: String
    }

    private static let logger = Logger(subsystem: "LocalWispr", category: "CloudCleanup")

    private let timeoutNanoseconds: UInt64
    private let session: URLSession
    private let retrySession: URLSession
    private let projectIndex: ProjectIdentifierIndex

    public init(timeoutNanoseconds: UInt64 = 850_000_000, session: URLSession? = nil, projectIndex: ProjectIdentifierIndex = ProjectIdentifierIndex()) {
        self.timeoutNanoseconds = timeoutNanoseconds
        self.session = session ?? Self.makeCloudSession()
        self.retrySession = Self.makeCloudSession()
        self.projectIndex = projectIndex
    }

    public var availability: CleanerAvailability {
        let provider = configuredProvider()
        DebugLog.write("[CloudCleaner] availability check: provider=\(provider?.name ?? "nil")")
        return provider == nil ? .unavailable("Missing cloud API key configuration") : .available
    }

    public func clean(_ rawText: String) async throws -> String {
        let trimmed = rawText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return "" }

        guard let provider = configuredProvider() else {
            DebugLog.write("[CloudCleaner] ERROR: no provider configured")
            throw TextCleanerError.unavailable("Cloud provider not configured")
        }
        DebugLog.write("[CloudCleaner] using provider=\(provider.name) model=\(provider.model)")

        let context: DictationAppContext
        if let pinnedContext = await SessionContextStore.shared.get() {
            context = pinnedContext
        } else {
            context = await MainActor.run { AppContextCapture.captureFrontmost() }
        }
        DebugLog.write(
            "[CloudCleaner] context: app=\(context.appName) surface=\(context.surface.rawValue) project=\(context.projectName ?? "nil") file=\(context.activeDocumentHint ?? "nil") browserTab=\(context.browserTabHint ?? "nil") browserURL=\(context.browserURL ?? "nil") browserHost=\(context.browserHost ?? "nil") browserPath=\(context.browserPathHint ?? "nil")"
        )
        let identifiers = await projectIndex.tieredIdentifiers(context: context)
        DebugLog.write("[CloudCleaner] identifiers: \(identifiers.count) top=\(identifiers.prefix(10))")
        let prompt = Self.userPrompt(rawText: trimmed, context: context, identifiers: identifiers)

        do {
            let response = try await withTimeout(nanoseconds: timeoutNanoseconds) {
                try await self.requestCompletion(provider: provider, prompt: prompt)
            }
            let sanitized = TextCleaner.sanitizeModelOutput(response)
            let cleaned = TextCleaner.applyFormattingDirectives(rawTranscription: trimmed, cleanedOutput: sanitized)
            guard let guarded = Self.guardAgainstExpansiveRewrite(raw: trimmed, cleaned: cleaned) else {
                DebugLog.write("[CloudCleaner] rejected expansive rewrite: raw=\(trimmed) cleaned=\(cleaned)")
                throw TextCleanerError.generationFailure("Cloud cleanup rewrite rejected")
            }
            DebugLog.write("[CloudCleaner] SUCCESS: raw=\(trimmed) → cleaned=\(guarded)")
            Self.logger.info("Cloud cleanup provider=\(provider.name, privacy: .public) app=\(context.appName, privacy: .public) ids=\(identifiers.count, privacy: .public)")
            guard !guarded.isEmpty else {
                throw TextCleanerError.generationFailure("Cloud cleanup returned empty output")
            }
            return guarded
        } catch {
            DebugLog.write("[CloudCleaner] FAILED: \(error)")
            Self.logger.error("Cloud cleanup failed: \(error.localizedDescription, privacy: .public)")
            if let cleanerError = error as? TextCleanerError {
                throw cleanerError
            }
            throw TextCleanerError.generationFailure(error.localizedDescription)
        }
    }

    private func requestCompletion(provider: ProviderConfig, prompt: String) async throws -> String {
        var request = URLRequest(url: provider.endpoint)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        request.setValue("identity", forHTTPHeaderField: "Accept-Encoding")
        request.setValue("Bearer \(provider.apiKey)", forHTTPHeaderField: "Authorization")

        let body: [String: Any] = [
            "model": provider.model,
            "temperature": 0,
            "max_tokens": 220,
            "messages": [
                ["role": "system", "content": Self.systemPrompt],
                ["role": "user", "content": prompt]
            ]
        ]

        request.httpBody = try JSONSerialization.data(withJSONObject: body)
        let (data, response) = try await performRequest(request)
        guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            let payload = String(data: data, encoding: .utf8) ?? "<empty>"
            throw TextCleanerError.generationFailure("HTTP error: \(payload)")
        }

        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let choices = json["choices"] as? [[String: Any]],
              let first = choices.first,
              let message = first["message"] as? [String: Any],
              let content = message["content"] as? String else {
            throw TextCleanerError.generationFailure("Malformed response")
        }

        return content
    }

    private func performRequest(_ request: URLRequest) async throws -> (Data, URLResponse) {
        do {
            return try await session.data(for: request)
        } catch {
            guard Self.isRetryableTransportError(error) else {
                throw error
            }

            DebugLog.write("[CloudCleaner] retrying request after transport failure: \(error)")
            return try await retrySession.data(for: request)
        }
    }

    private static func makeCloudSession() -> URLSession {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.requestCachePolicy = .reloadIgnoringLocalCacheData
        configuration.urlCache = nil
        configuration.httpCookieStorage = nil
        configuration.httpShouldSetCookies = false
        configuration.waitsForConnectivity = false
        configuration.timeoutIntervalForRequest = 4
        configuration.timeoutIntervalForResource = 6
        return URLSession(configuration: configuration)
    }

    private static func isRetryableTransportError(_ error: Error) -> Bool {
        let nsError = error as NSError
        guard nsError.domain == NSURLErrorDomain else { return false }

        switch nsError.code {
        case NSURLErrorCannotParseResponse,
             NSURLErrorBadServerResponse,
             NSURLErrorNetworkConnectionLost,
             NSURLErrorTimedOut,
             NSURLErrorCannotDecodeRawData,
             NSURLErrorCannotDecodeContentData:
            return true
        default:
            return false
        }
    }

    private static let systemPrompt = """
    You are a speech-to-text correction engine.
    Return only corrected text.
    Keep original meaning.
    Fix punctuation, capitalization, spacing, and common transcription errors.
    Prefer technical terms present in provided context identifiers.
    Do not add new claims, descriptive phrases, URLs, or product details that are not already supported by the raw transcription.
    Prefer substitutions and formatting fixes over rewriting.
    If uncertain, keep the original wording.
    Do not add explanations.
    """

    static func userPrompt(rawText: String, context: DictationAppContext, identifiers: [String]) -> String {
        var lines: [String] = []
        lines.append("Raw transcription:")
        lines.append(rawText)
        lines.append("")
        lines.append("Context:")
        lines.append("- App: \(context.appName) [\(context.surface.rawValue)]")
        if let projectName = context.projectName, !projectName.isEmpty {
            lines.append("- Project: \(projectName)")
        }
        if let activeDocumentHint = context.activeDocumentHint, !activeDocumentHint.isEmpty {
            lines.append("- Active file: \(activeDocumentHint)")
        }
        if let browserURL = context.browserURL, !browserURL.isEmpty {
            lines.append("- Browser URL: \(browserURL)")
        }
        if let browserHost = context.browserHost, !browserHost.isEmpty {
            lines.append("- Browser host: \(browserHost)")
        }
        if let browserPathHint = context.browserPathHint, !browserPathHint.isEmpty, browserPathHint != "/" {
            lines.append("- Browser path: \(browserPathHint)")
        }
        if !identifiers.isEmpty {
            lines.append("- Preferred identifiers: \(identifiers.prefix(80).joined(separator: ", "))")
        }
        return lines.joined(separator: "\n")
    }

    private func configuredProvider() -> ProviderConfig? {
        let env = DotEnv.merged()

        if let key = env["GROQ_API_KEY"], !key.isEmpty {
            return ProviderConfig(
                name: "groq",
                endpoint: URL(string: "https://api.groq.com/openai/v1/chat/completions")!,
                model: env["LOCALWISPR_GROQ_MODEL"] ?? "llama-3.3-70b-versatile",
                apiKey: key
            )
        }

        return nil
    }

    private func withTimeout<T: Sendable>(nanoseconds: UInt64, operation: @escaping @Sendable () async throws -> T) async throws -> T {
        try await withThrowingTaskGroup(of: T.self) { group in
            group.addTask {
                try await operation()
            }
            group.addTask {
                try await Task.sleep(nanoseconds: nanoseconds)
                throw TextCleanerError.generationFailure("Request timed out")
            }

            let value = try await group.next()!
            group.cancelAll()
            return value
        }
    }

    private static func guardAgainstExpansiveRewrite(raw: String, cleaned: String) -> String? {
        let trimmedCleaned = cleaned.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedCleaned.isEmpty else { return nil }

        let rawTokens = lexicalTokens(in: raw)
        let cleanedTokens = lexicalTokens(in: trimmedCleaned)

        guard !rawTokens.isEmpty, !cleanedTokens.isEmpty else {
            return nil
        }

        let rawCount = rawTokens.count
        let cleanedCount = cleanedTokens.count
        if cleanedCount > rawCount + 1 || Double(cleanedCount) > (Double(rawCount) * 1.25) {
            return nil
        }

        let rawTokenSet = Set(rawTokens)
        let cleanedTokenSet = Set(cleanedTokens)
        let addedTokenCount = cleanedTokenSet.subtracting(rawTokenSet).count
        let removedTokenCount = rawTokenSet.subtracting(cleanedTokenSet).count
        if addedTokenCount > removedTokenCount + 1 {
            return nil
        }

        if trimmedCleaned.contains("://") || trimmedCleaned.contains("www.") || trimmedCleaned.contains(".com") || trimmedCleaned.contains(".ai") {
            let rawLower = raw.lowercased()
            if !rawLower.contains("://") && !rawLower.contains("www.") && !rawLower.contains(".com") && !rawLower.contains(".ai") {
                return nil
            }
        }

        return trimmedCleaned
    }

    private static func lexicalTokens(in text: String) -> [String] {
        text.lowercased()
            .split { !$0.isLetter && !$0.isNumber }
            .map(String.init)
    }
}

// MARK: - Fuzzy Post-Processing

public enum FuzzyIdentifierMatcher {
    /// Runs compound-word joining and Levenshtein matching against known identifiers.
    /// Call this on cleaned text before insertion.
    public static func postProcess(_ text: String, identifiers: [String]) -> String {
        guard !identifiers.isEmpty else { return text }

        // Build lookup: lowercased joined form → original identifier
        // e.g. "fetchusers" → "fetchUsers", "api_key" → "API_KEY"
        var joinedLookup: [String: String] = [:]
        for id in identifiers {
            joinedLookup[id.lowercased()] = id
            // Also index the space-separated spoken form
            let spoken = spokenForm(of: id).lowercased()
            if spoken != id.lowercased() {
                joinedLookup[spoken.replacingOccurrences(of: " ", with: "")] = id
            }
        }

        let words = tokenize(text)
        var result = text

        // Try joining consecutive word pairs and triples
        for windowSize in [3, 2] {
            guard words.count >= windowSize else { continue }
            // Process from end to start so replacements don't shift indices
            var i = words.count - windowSize
            while i >= 0 {
                let window = words[i..<(i + windowSize)]
                let joined = window.map(\.text).joined().lowercased()
                if let match = joinedLookup[joined] {
                    let rangeStart = window.first!.range.lowerBound
                    let rangeEnd = window.last!.range.upperBound
                    result = String(result[result.startIndex..<rangeStart]) + match + String(result[rangeEnd...])
                }
                i -= 1
            }
        }

        // Single-word Levenshtein matching for remaining words
        let updatedWords = tokenize(result)
        for token in updatedWords.reversed() {
            let word = token.text
            guard word.count >= 4 else { continue } // skip short words
            if joinedLookup[word.lowercased()] != nil { continue } // already exact match

            if let bestMatch = closestIdentifier(to: word, from: identifiers, maxDistance: 2) {
                result = String(result[result.startIndex..<token.range.lowerBound]) + bestMatch + String(result[token.range.upperBound...])
            }
        }

        return result
    }

    struct WordToken {
        let text: String
        let range: Range<String.Index>
    }

    static func tokenize(_ text: String) -> [WordToken] {
        var tokens: [WordToken] = []
        var i = text.startIndex
        while i < text.endIndex {
            if text[i].isLetter || text[i] == "_" {
                let start = i
                while i < text.endIndex && (text[i].isLetter || text[i].isNumber || text[i] == "_") {
                    i = text.index(after: i)
                }
                tokens.append(WordToken(text: String(text[start..<i]), range: start..<i))
            } else {
                i = text.index(after: i)
            }
        }
        return tokens
    }

    /// Converts camelCase/snake_case identifiers to spoken form:
    /// "fetchUsers" → "fetch users", "API_KEY" → "api key"
    static func spokenForm(of identifier: String) -> String {
        if identifier.contains("_") {
            return identifier.replacingOccurrences(of: "_", with: " ").lowercased()
        }
        // Split camelCase
        var parts: [String] = []
        var current = ""
        for char in identifier {
            if char.isUppercase && !current.isEmpty {
                // Check if we're in an acronym run (e.g. "API" in "APIClient")
                if current.last?.isUppercase == true {
                    current.append(char)
                } else {
                    parts.append(current)
                    current = String(char)
                }
            } else {
                if current.count > 1 && current.allSatisfy(\.isUppercase) && char.isLowercase {
                    // End of acronym: "APIClient" → ["API", "Client"]
                    let split = String(current.dropLast())
                    parts.append(split)
                    current = String(current.last!) + String(char)
                } else {
                    current.append(char)
                }
            }
        }
        if !current.isEmpty { parts.append(current) }
        return parts.joined(separator: " ").lowercased()
    }

    static func closestIdentifier(to word: String, from identifiers: [String], maxDistance: Int) -> String? {
        let wordLower = word.lowercased()
        var best: (identifier: String, distance: Int)?
        for id in identifiers {
            let idLower = id.lowercased()
            // Quick length check to skip obvious non-matches
            if abs(wordLower.count - idLower.count) > maxDistance { continue }
            let dist = levenshteinDistance(wordLower, idLower)
            if dist <= maxDistance && dist > 0 {
                if best == nil || dist < best!.distance {
                    best = (id, dist)
                }
            }
        }
        return best?.identifier
    }

    static func levenshteinDistance(_ s: String, _ t: String) -> Int {
        let sChars = Array(s)
        let tChars = Array(t)
        let m = sChars.count
        let n = tChars.count
        if m == 0 { return n }
        if n == 0 { return m }

        var prev = Array(0...n)
        var curr = Array(repeating: 0, count: n + 1)

        for i in 1...m {
            curr[0] = i
            for j in 1...n {
                let cost = sChars[i - 1] == tChars[j - 1] ? 0 : 1
                curr[j] = min(
                    prev[j] + 1,       // deletion
                    curr[j - 1] + 1,   // insertion
                    prev[j - 1] + cost  // substitution
                )
            }
            swap(&prev, &curr)
        }
        return prev[n]
    }
}

// MARK: - Adaptive Cleaner

public final class AdaptiveTextCleaner: @unchecked Sendable, Cleaning {
    private let cloudCleaner: ContextAwareCloudCleaner

    public init(
        cloudCleaner: ContextAwareCloudCleaner = ContextAwareCloudCleaner(),
        localCleaner: TextCleaner = TextCleaner(),
        projectIndex: ProjectIdentifierIndex = ProjectIdentifierIndex()
    ) {
        self.cloudCleaner = cloudCleaner
        _ = localCleaner
        _ = projectIndex
    }

    public var availability: CleanerAvailability {
        cloudCleaner.availability
    }

    public func clean(_ rawText: String) async throws -> String {
        DebugLog.write("[AdaptiveCleaner] clean called: rawText=\(rawText.prefix(80))")
        DebugLog.write("[AdaptiveCleaner] using Groq cleaner")
        let cleaned = try await cloudCleaner.clean(rawText)
        DebugLog.write("[AdaptiveCleaner] final output: \(cleaned)")
        return cleaned
    }
}
