import Testing
@testable import LocalWispr

struct CloudSpeechFallbackPolicyTests {
    @Test
    func disabledByDefault() {
        #expect(CloudSpeechFallbackPolicy.isEnabled(environment: [:]) == false)
        #expect(CloudSpeechFallbackPolicy.isEnabled(environment: ["LOCALWISPR_ENABLE_CLOUD_STT": "0"]) == false)
        #expect(CloudSpeechFallbackPolicy.isEnabled(environment: ["LOCALWISPR_ENABLE_CLOUD_STT": "false"]) == false)
    }

    @Test
    func acceptsTruthyValues() {
        #expect(CloudSpeechFallbackPolicy.isEnabled(environment: ["LOCALWISPR_ENABLE_CLOUD_STT": "1"]) == true)
        #expect(CloudSpeechFallbackPolicy.isEnabled(environment: ["LOCALWISPR_ENABLE_CLOUD_STT": "true"]) == true)
        #expect(CloudSpeechFallbackPolicy.isEnabled(environment: ["LOCALWISPR_ENABLE_CLOUD_STT": " yes "]) == true)
    }
}
