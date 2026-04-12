import Testing
@testable import LocalWispr

struct TranscriptResolutionPolicyTests {
    @Test
    func emptyTranscriptIsTreatedAsMissing() {
        #expect(TranscriptResolutionPolicy.normalizedTranscript(nil) == nil)
        #expect(TranscriptResolutionPolicy.normalizedTranscript("") == nil)
        #expect(TranscriptResolutionPolicy.normalizedTranscript("   \n  ") == nil)
        #expect(TranscriptResolutionPolicy.normalizedTranscript(" hello ") == "hello")
    }

    @Test
    func fallsBackToLatestPartialTranscript() {
        #expect(TranscriptResolutionPolicy.resolvedTranscript(primary: " hello ", fallback: "backup") == "hello")
        #expect(TranscriptResolutionPolicy.resolvedTranscript(primary: "   ", fallback: " partial result ") == "partial result")
        #expect(TranscriptResolutionPolicy.resolvedTranscript(primary: nil, fallback: nil) == nil)
    }

    @Test
    func prefersStrongerAlternativeTranscript() {
        #expect(
            TranscriptResolutionPolicy.preferredTranscript(
                primary: "Test software",
                alternative: "Auto test software"
            ) == "Auto test software"
        )
        #expect(
            TranscriptResolutionPolicy.preferredTranscript(
                primary: "Chrome is being controlled by auto soft",
                alternative: "Chrome is being controlled by auto"
            ) == "Chrome is being controlled by auto soft"
        )
    }

    @Test
    func batchVerificationIsDisabled() {
        #expect(
            TranscriptResolutionPolicy.shouldAttemptBatchVerification(
                liveTranscript: "Test software",
                recordingDurationMilliseconds: 12_000
            ) == false
        )
        #expect(
            TranscriptResolutionPolicy.shouldAttemptBatchVerification(
                liveTranscript: "Chrome is being controlled by auto soft",
                recordingDurationMilliseconds: 4_000
            ) == false
        )
    }

    @Test
    func dictationKeepsPreferredModeOnly() {
        #expect(
            TranscriptResolutionPolicy.fallbackModes(after: .dictationLong) ==
            [.dictationLong]
        )
    }

    @Test
    func speechTranscriptionKeepsPreferredModeOnly() {
        #expect(
            TranscriptResolutionPolicy.fallbackModes(after: .speechTranscription) ==
            [.speechTranscription]
        )
    }
}
