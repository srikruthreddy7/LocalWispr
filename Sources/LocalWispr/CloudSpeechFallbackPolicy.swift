import Foundation

enum CloudSpeechFallbackPolicy {
    static func isEnabled(environment: [String: String]) -> Bool {
        guard let rawValue = environment["LOCALWISPR_ENABLE_CLOUD_STT"] else {
            return false
        }

        switch rawValue.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
        case "1", "true", "yes", "on":
            return true
        default:
            return false
        }
    }
}
