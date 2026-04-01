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
}
