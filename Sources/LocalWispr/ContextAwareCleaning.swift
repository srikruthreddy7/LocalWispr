import AppKit
import ApplicationServices
import Foundation
import OSLog

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

    public init(
        capturedAt: Date = Date(),
        appName: String,
        bundleIdentifier: String,
        windowTitle: String,
        surface: Surface,
        projectName: String? = nil,
        projectPathHint: String? = nil,
        activeDocumentHint: String? = nil,
        browserTabHint: String? = nil
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
    }

    public static let unknown = DictationAppContext(
        appName: "Unknown",
        bundleIdentifier: "unknown",
        windowTitle: "",
        surface: .other
    )
}

public enum AppContextCapture {
    @MainActor
    public static func captureFrontmost() -> DictationAppContext {
        guard let frontmost = NSWorkspace.shared.frontmostApplication else {
            return .unknown
        }

        let bundleID = frontmost.bundleIdentifier ?? "unknown"
        let appName = frontmost.localizedName ?? bundleID
        let title = focusedWindowTitle(for: frontmost.processIdentifier) ?? ""
        return DictationContextParser.parse(appName: appName, bundleIdentifier: bundleID, windowTitle: title)
    }

    @MainActor
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

    public static func parse(appName: String, bundleIdentifier: String, windowTitle: String) -> DictationAppContext {
        if browserBundleIDs.contains(bundleIdentifier) {
            return parseBrowser(appName: appName, bundleIdentifier: bundleIdentifier, windowTitle: windowTitle)
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

    private static func parseBrowser(appName: String, bundleIdentifier: String, windowTitle: String) -> DictationAppContext {
        let stripped = windowTitle
            .replacingOccurrences(of: " - Google Chrome", with: "")
            .replacingOccurrences(of: " - Safari", with: "")
            .replacingOccurrences(of: " - Mozilla Firefox", with: "")
            .trimmingCharacters(in: .whitespacesAndNewlines)

        return DictationAppContext(
            appName: appName,
            bundleIdentifier: bundleIdentifier,
            windowTitle: windowTitle,
            surface: .browser,
            browserTabHint: stripped.isEmpty ? nil : stripped
        )
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

    private var cache: [String: CacheEntry] = [:]
    private let logger = Logger(subsystem: "LocalWispr", category: "ContextIndex")

    public init() {}

    public func identifiers(forProjectPath path: String?, limit: Int = 80) -> [String] {
        guard let path, !path.isEmpty else { return [] }

        if let cached = cache[path], Date().timeIntervalSince(cached.scannedAt) < 20 {
            return Array(cached.identifiers.prefix(limit))
        }

        let extracted = Self.scanIdentifiers(rootPath: path, limit: max(limit, 120))
        cache[path] = CacheEntry(scannedAt: Date(), identifiers: extracted)
        logger.info("Indexed identifiers root=\(path, privacy: .public) count=\(extracted.count, privacy: .public)")
        return Array(extracted.prefix(limit))
    }

    static func scanIdentifiers(rootPath: String, limit: Int) -> [String] {
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

            guard Self.isCodeFile(path: path),
                  let text = try? String(contentsOf: url, encoding: .utf8),
                  text.count <= 350_000 else {
                continue
            }
            visitedFiles += 1
            for identifier in Self.extractIdentifiers(from: text) {
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

    private static func isCodeFile(path: String) -> Bool {
        let allowed = ["swift", "py", "js", "ts", "tsx", "jsx", "java", "kt", "go", "rs", "c", "cc", "cpp", "h", "hpp", "cs", "rb", "php", "md", "json", "yaml", "yml", "toml"]
        guard let ext = path.split(separator: ".").last else { return false }
        return allowed.contains(String(ext).lowercased())
    }
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
    private let projectIndex: ProjectIdentifierIndex

    public init(timeoutNanoseconds: UInt64 = 850_000_000, session: URLSession = .shared, projectIndex: ProjectIdentifierIndex = ProjectIdentifierIndex()) {
        self.timeoutNanoseconds = timeoutNanoseconds
        self.session = session
        self.projectIndex = projectIndex
    }

    public var availability: CleanerAvailability {
        configuredProvider() == nil ? .unavailable("Missing cloud API key configuration") : .available
    }

    public func clean(_ rawText: String) async throws -> String {
        let trimmed = rawText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return "" }

        guard let provider = configuredProvider() else {
            throw TextCleanerError.unavailable("Cloud provider not configured")
        }

        let context = await MainActor.run { AppContextCapture.captureFrontmost() }
        let identifiers = await projectIndex.identifiers(forProjectPath: context.projectPathHint)
        let prompt = Self.userPrompt(rawText: trimmed, context: context, identifiers: identifiers)

        do {
            let response = try await withTimeout(nanoseconds: timeoutNanoseconds) {
                try await requestCompletion(provider: provider, prompt: prompt)
            }
            let sanitized = TextCleaner.sanitizeModelOutput(response)
            let cleaned = TextCleaner.applyFormattingDirectives(rawTranscription: trimmed, cleanedOutput: sanitized)
            Self.logger.info("Cloud cleanup provider=\(provider.name, privacy: .public) app=\(context.appName, privacy: .public) ids=\(identifiers.count, privacy: .public)")
            return cleaned.isEmpty ? TextCleaner.fastClean(rawTranscription: trimmed) : cleaned
        } catch {
            Self.logger.error("Cloud cleanup failed: \(error.localizedDescription, privacy: .public)")
            return TextCleaner.fastClean(rawTranscription: trimmed)
        }
    }

    private func requestCompletion(provider: ProviderConfig, prompt: String) async throws -> String {
        var request = URLRequest(url: provider.endpoint)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
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
        let (data, response) = try await session.data(for: request)
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

    private static let systemPrompt = """
    You are a speech-to-text correction engine.
    Return only corrected text.
    Keep original meaning.
    Fix punctuation, capitalization, spacing, and common transcription errors.
    Prefer technical terms present in provided context identifiers.
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
        if let browserTabHint = context.browserTabHint, !browserTabHint.isEmpty {
            lines.append("- Browser tab: \(browserTabHint)")
        }
        if !identifiers.isEmpty {
            lines.append("- Preferred identifiers: \(identifiers.prefix(80).joined(separator: ", "))")
        }
        return lines.joined(separator: "\n")
    }

    private func configuredProvider() -> ProviderConfig? {
        let env = ProcessInfo.processInfo.environment

        if let key = env["CEREBRAS_API_KEY"] ?? env["CEREPLUS_API_KEY"], !key.isEmpty {
            return ProviderConfig(
                name: "cerebras",
                endpoint: URL(string: "https://api.cerebras.ai/v1/chat/completions")!,
                model: env["LOCALWISPR_CEREBRAS_MODEL"] ?? "llama-3.3-70b",
                apiKey: key
            )
        }

        if let key = env["GROK_API_KEY"] ?? env["XAI_API_KEY"], !key.isEmpty {
            return ProviderConfig(
                name: "grok",
                endpoint: URL(string: "https://api.x.ai/v1/chat/completions")!,
                model: env["LOCALWISPR_GROK_MODEL"] ?? "grok-3-latest",
                apiKey: key
            )
        }

        return nil
    }

    private func withTimeout<T>(nanoseconds: UInt64, operation: @escaping @Sendable () async throws -> T) async throws -> T {
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
}

public final class AdaptiveTextCleaner: @unchecked Sendable, Cleaning {
    private let cloudCleaner: ContextAwareCloudCleaner
    private let localCleaner: TextCleaner

    public init(cloudCleaner: ContextAwareCloudCleaner = ContextAwareCloudCleaner(), localCleaner: TextCleaner = TextCleaner()) {
        self.cloudCleaner = cloudCleaner
        self.localCleaner = localCleaner
    }

    public var availability: CleanerAvailability {
        switch (cloudCleaner.availability, localCleaner.availability) {
        case (.available, _), (_, .available):
            return .available
        case (.unavailable(let cloudReason), .unavailable(let localReason)):
            return .unavailable("cloud: \(cloudReason); local: \(localReason)")
        }
    }

    public func clean(_ rawText: String) async throws -> String {
        if case .available = cloudCleaner.availability {
            return try await cloudCleaner.clean(rawText)
        }

        do {
            return try await localCleaner.clean(rawText)
        } catch {
            return TextCleaner.fastClean(rawTranscription: rawText)
        }
    }
}
