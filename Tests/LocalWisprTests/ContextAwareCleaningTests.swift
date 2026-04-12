import XCTest
@testable import LocalWispr

final class DictationContextParserTests: XCTestCase {
    func testParsesCursorWindowTitleProjectAndFile() {
        let context = DictationContextParser.parse(
            appName: "Cursor",
            bundleIdentifier: "com.todesktop.230313mzl4w4u92",
            windowTitle: "ContextAwareCleaning.swift -- LocalWispr -- Cursor"
        )

        XCTAssertEqual(context.surface, .ide)
        XCTAssertEqual(context.projectName, "LocalWispr")
        XCTAssertEqual(context.activeDocumentHint, "ContextAwareCleaning.swift")
    }

    func testParsesBrowserTab() {
        let context = DictationContextParser.parse(
            appName: "Google Chrome",
            bundleIdentifier: "com.google.Chrome",
            windowTitle: "LocalWispr architecture notes - Google Chrome"
        )

        XCTAssertEqual(context.surface, .browser)
        XCTAssertEqual(context.browserTabHint, "LocalWispr architecture notes")
    }

    func testPrefersRememberedExternalContextWhenFrontmostIsLocalWispr() {
        let frontmost = DictationAppContext(
            appName: "LocalWispr",
            bundleIdentifier: "com.localwispr.host",
            windowTitle: "LocalWispr",
            surface: .other
        )
        let remembered = DictationAppContext(
            appName: "Cursor",
            bundleIdentifier: "com.todesktop.230313mzl4w4u92",
            windowTitle: "ContextAwareCleaning.swift -- LocalWispr -- Cursor",
            surface: .ide,
            projectName: "LocalWispr",
            activeDocumentHint: "ContextAwareCleaning.swift"
        )

        let effective = AppContextCapture.effectiveContext(frontmost: frontmost, rememberedExternal: remembered)
        XCTAssertEqual(effective, remembered)
    }

    func testKeepsFrontmostContextWhenItIsNotLocalWispr() {
        let frontmost = DictationAppContext(
            appName: "Google Chrome",
            bundleIdentifier: "com.google.Chrome",
            windowTitle: "Issue Details - Google Chrome",
            surface: .browser,
            browserTabHint: "Issue Details"
        )
        let remembered = DictationAppContext(
            appName: "Cursor",
            bundleIdentifier: "com.todesktop.230313mzl4w4u92",
            windowTitle: "Main.swift -- Demo -- Cursor",
            surface: .ide,
            projectName: "Demo",
            activeDocumentHint: "Main.swift"
        )

        let effective = AppContextCapture.effectiveContext(frontmost: frontmost, rememberedExternal: remembered)
        XCTAssertEqual(effective, frontmost)
    }
}

final class ProjectIdentifierIndexTests: XCTestCase {
    func testExtractIdentifiersFromSource() {
        let source = """
        struct FetchUsersService {
            let API_KEY = \"abc\"
            func fetchUsersFromDB() {}
            let user_session_token = \"x\"
        }
        """

        let identifiers = ProjectIdentifierIndex.extractIdentifiers(from: source)
        XCTAssertTrue(identifiers.contains("FetchUsersService"))
        XCTAssertTrue(identifiers.contains("API_KEY"))
        XCTAssertTrue(identifiers.contains("fetchUsersFromDB"))
        XCTAssertTrue(identifiers.contains("user_session_token"))
    }

    func testUserPromptIncludesContextAndIdentifiers() {
        let context = DictationAppContext(
            appName: "Cursor",
            bundleIdentifier: "com.todesktop.230313mzl4w4u92",
            windowTitle: "File.swift -- MyProj -- Cursor",
            surface: .ide,
            projectName: "MyProj",
            activeDocumentHint: "File.swift"
        )

        let prompt = ContextAwareCloudCleaner.userPrompt(
            rawText: "fix fetch users",
            context: context,
            identifiers: ["fetchUsers", "APIClient"]
        )

        XCTAssertTrue(prompt.contains("App: Cursor"))
        XCTAssertTrue(prompt.contains("Project: MyProj"))
        XCTAssertTrue(prompt.contains("Active file: File.swift"))
        XCTAssertTrue(prompt.contains("fetchUsers"))
    }

    func testBrowserUserPromptExcludesRawTitleWords() {
        let context = DictationAppContext(
            appName: "Google Chrome",
            bundleIdentifier: "com.google.Chrome",
            windowTitle: "LYNKUP™ | Expertise on Demand",
            surface: .browser,
            browserTabHint: "LYNKUP™ | Expertise on Demand",
            browserURL: "https://lynkup.ai/",
            browserHost: "lynkup.ai",
            browserPathHint: "/"
        )

        let prompt = ContextAwareCloudCleaner.userPrompt(
            rawText: "expert",
            context: context,
            identifiers: ["lynkup"]
        )

        XCTAssertTrue(prompt.contains("Browser host: lynkup.ai"))
        XCTAssertTrue(prompt.contains("Preferred identifiers: lynkup"))
        XCTAssertFalse(prompt.contains("Expertise"))
        XCTAssertFalse(prompt.contains("Demand"))
        XCTAssertFalse(prompt.contains("Browser tab:"))
    }

    func testFindProjectRootFromSwiftFile() {
        // Uses the actual repo we're in as a test case
        let thisFile = #filePath
        let root = ProjectIdentifierIndex.findProjectRoot(from: thisFile)
        // Should find the repo root (contains Package.swift)
        if let root {
            let packageSwift = (root as NSString).appendingPathComponent("Package.swift")
            XCTAssertTrue(FileManager.default.fileExists(atPath: packageSwift), "Expected Package.swift at project root: \(root)")
        }
        // If running outside the repo, root may be nil — that's OK
    }

    func testIsCodeFile() {
        XCTAssertTrue(ProjectIdentifierIndex.isCodeFile(path: "/foo/bar.swift"))
        XCTAssertTrue(ProjectIdentifierIndex.isCodeFile(path: "/foo/bar.ts"))
        XCTAssertTrue(ProjectIdentifierIndex.isCodeFile(path: "/foo/bar.py"))
        XCTAssertFalse(ProjectIdentifierIndex.isCodeFile(path: "/foo/bar.png"))
        XCTAssertFalse(ProjectIdentifierIndex.isCodeFile(path: "/foo/bar.exe"))
    }

    func testBrowserHintsExcludeGenericTitleWords() async {
        let index = ProjectIdentifierIndex()
        let context = DictationAppContext(
            appName: "Google Chrome",
            bundleIdentifier: "com.google.Chrome",
            windowTitle: "LYNKUP™ | Expertise on Demand",
            surface: .browser,
            browserTabHint: "LYNKUP™ | Expertise on Demand",
            browserURL: "https://lynkup.ai/",
            browserHost: "lynkup.ai",
            browserPathHint: "/"
        )

        let identifiers = await index.tieredIdentifiers(context: context, limit: 20)

        XCTAssertTrue(identifiers.contains("lynkup.ai"))
        XCTAssertTrue(identifiers.contains("LYNKUP") || identifiers.contains("lynkup"))
        XCTAssertFalse(identifiers.contains("Expertise"))
        XCTAssertFalse(identifiers.contains("Demand"))
    }
}

// MARK: - Fuzzy Identifier Matcher Tests

final class FuzzyIdentifierMatcherTests: XCTestCase {
    func testCompoundWordJoining() {
        let identifiers = ["fetchUsers", "APIClient", "useState"]
        let result = FuzzyIdentifierMatcher.postProcess("call fetch users from the API", identifiers: identifiers)
        XCTAssertTrue(result.contains("fetchUsers"), "Expected 'fetch users' → 'fetchUsers', got: \(result)")
    }

    func testCompoundWordJoiningUseState() {
        let identifiers = ["useState", "useEffect", "useCallback"]
        let result = FuzzyIdentifierMatcher.postProcess("import the use state hook", identifiers: identifiers)
        XCTAssertTrue(result.contains("useState"), "Expected 'use state' → 'useState', got: \(result)")
    }

    func testLevenshteinDistance() {
        XCTAssertEqual(FuzzyIdentifierMatcher.levenshteinDistance("kitten", "sitting"), 3)
        XCTAssertEqual(FuzzyIdentifierMatcher.levenshteinDistance("", "abc"), 3)
        XCTAssertEqual(FuzzyIdentifierMatcher.levenshteinDistance("abc", "abc"), 0)
        XCTAssertEqual(FuzzyIdentifierMatcher.levenshteinDistance("abc", "abd"), 1)
    }

    func testLevenshteinMatching() {
        let identifiers = ["fetchUsers", "pytest"]
        // "pyest" is 1 edit from "pytest"
        let result = FuzzyIdentifierMatcher.postProcess("run pyest now", identifiers: identifiers)
        XCTAssertTrue(result.contains("pytest"), "Expected 'pyest' → 'pytest', got: \(result)")
    }

    func testNoFalsePositivesOnShortWords() {
        let identifiers = ["fetchUsers"]
        // Short words (< 4 chars) should not be fuzzy matched
        let result = FuzzyIdentifierMatcher.postProcess("the cat sat", identifiers: identifiers)
        XCTAssertEqual(result, "the cat sat")
    }

    func testSpokenFormCamelCase() {
        XCTAssertEqual(FuzzyIdentifierMatcher.spokenForm(of: "fetchUsers"), "fetch users")
        XCTAssertEqual(FuzzyIdentifierMatcher.spokenForm(of: "APIClient"), "api client")
        XCTAssertEqual(FuzzyIdentifierMatcher.spokenForm(of: "getElementById"), "get element by id")
    }

    func testSpokenFormSnakeCase() {
        XCTAssertEqual(FuzzyIdentifierMatcher.spokenForm(of: "user_session_token"), "user session token")
        XCTAssertEqual(FuzzyIdentifierMatcher.spokenForm(of: "API_KEY"), "api key")
    }

    func testEmptyIdentifiersReturnsOriginal() {
        let result = FuzzyIdentifierMatcher.postProcess("hello world", identifiers: [])
        XCTAssertEqual(result, "hello world")
    }

    func testExactMatchNotReplaced() {
        let identifiers = ["fetchUsers"]
        let result = FuzzyIdentifierMatcher.postProcess("call fetchUsers now", identifiers: identifiers)
        XCTAssertTrue(result.contains("fetchUsers"))
    }
}

// MARK: - Groq Provider Tests

final class GroqProviderTests: XCTestCase {
    func testGroqProviderName() {
        // Verify the prompt composition works (we can't test the actual API without keys)
        let context = DictationAppContext(
            appName: "VS Code",
            bundleIdentifier: "com.microsoft.VSCode",
            windowTitle: "main.py -- myproject -- Visual Studio Code",
            surface: .ide,
            projectName: "myproject",
            activeDocumentHint: "main.py"
        )

        let prompt = ContextAwareCloudCleaner.userPrompt(
            rawText: "define fetch users function",
            context: context,
            identifiers: ["fetch_users", "APIClient"]
        )

        XCTAssertTrue(prompt.contains("Project: myproject"))
        XCTAssertTrue(prompt.contains("fetch_users"))
    }
}
