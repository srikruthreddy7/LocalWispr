import FoundationModels
import XCTest
@testable import LocalWispr

final class TextCleanerTests: XCTestCase {
    func testSanitizeModelOutputStripsAssistantPreamble() {
        let input = """
        Sure! Here is the cleaned text:
        - Test one
        - Two
        """

        let cleaned = TextCleaner.sanitizeModelOutput(input)
        XCTAssertEqual(cleaned, "- Test one\n- Two")
    }

    func testSanitizeModelOutputStripsCodeFenceAndLabel() {
        let input = """
        ```text
        Cleaned text: Hello there.
        ```
        """

        let cleaned = TextCleaner.sanitizeModelOutput(input)
        XCTAssertEqual(cleaned, "Hello there.")
    }

    func testFormattingDirectiveConvertsOrdinalSpeechToBullets() {
        let raw = """
        Put this in the bullet points. The first one would be the test. The second one would be test two. The third one would be test three. The fourth one would be this is a wispr flow alternative.
        """

        let cleanedParagraph = """
        Put this in bullet points. The first is the test, the second is test two, the third is test three, and the fourth is this is a Wispr Flow alternative.
        """

        let final = TextCleaner.applyFormattingDirectives(rawTranscription: raw, cleanedOutput: cleanedParagraph)
        XCTAssertEqual(
            final,
            """
            - Test
            - Test two
            - Test three
            - This is a Wispr Flow alternative
            """
        )
    }

    func testFormattingDirectiveConvertsUserReportedPhraseToBullets() {
        let raw = """
        Put this in the bullet points. The first one would be the test. The second one would be test to the third one would be test three. The fourth one would be this is at whisper flow, alternative, how is it working?
        """
        let cleanedParagraph = """
        Put this in bullet points. The first is the test, the second is test two, the third is test three, and the fourth is this is a Wispr Flow alternative. How is it working?
        """

        let final = TextCleaner.applyFormattingDirectives(rawTranscription: raw, cleanedOutput: cleanedParagraph)
        XCTAssertEqual(
            final,
            """
            - Test
            - Test two
            - Test three
            - This is a Wispr Flow alternative. How is it working?
            """
        )
    }

    func testFastCleanRemovesOnlyHardFillersAndAddsPunctuation() {
        let cleaned = TextCleaner.fastClean(rawTranscription: "oh hey um actually this is a test")
        XCTAssertEqual(cleaned, "Actually this is a test.")
    }

    func testFastCleanPreservesLeadingOkayWhenItCarriesTone() {
        let cleaned = TextCleaner.fastClean(rawTranscription: "okay so is it median or mean")
        XCTAssertEqual(cleaned, "Okay so is it median or mean?")
    }

    func testSimpleDictationDoesNotRequireLanguageModel() {
        XCTAssertFalse(TextCleaner.shouldUseLanguageModel(for: "hey can you take out a job in this town"))
    }

    func testExplicitRewriteDirectiveUsesLanguageModelPath() {
        XCTAssertTrue(TextCleaner.shouldUseLanguageModel(for: "rewrite this as a formal email to recruiting"))
    }

    func testGenerationErrorMappingForRequiredCases() {
        let context = LanguageModelSession.GenerationError.Context(debugDescription: "debug")

        switch TextCleaner.mapGenerationError(.exceededContextWindowSize(context)) {
        case .exceededContextWindowSize:
            break
        default:
            XCTFail("Expected exceededContextWindowSize mapping")
        }

        switch TextCleaner.mapGenerationError(.assetsUnavailable(context)) {
        case .assetsUnavailable:
            break
        default:
            XCTFail("Expected assetsUnavailable mapping")
        }

        switch TextCleaner.mapGenerationError(.guardrailViolation(context)) {
        case .guardrailViolation:
            break
        default:
            XCTFail("Expected guardrailViolation mapping")
        }
    }

    func testGenerationErrorMappingFallsBackToGenericFailure() {
        let context = LanguageModelSession.GenerationError.Context(debugDescription: "debug")
        let mapped = TextCleaner.mapGenerationError(.unsupportedGuide(context))

        guard case .generationFailure(let details) = mapped else {
            return XCTFail("Expected generic generationFailure mapping")
        }

        XCTAssertFalse(details.isEmpty)
    }
}
