import Testing
@testable import LocalWispr

struct PermissionCapabilitiesTests {
    @Test
    func dictationDoesNotRequireAccessibility() {
        let capabilities = PermissionCapabilities(
            accessibilityGranted: false,
            microphoneGranted: true,
            speechGranted: true
        )

        #expect(capabilities.canDictate)
        #expect(!capabilities.canInsertIntoOtherApps)
        #expect(!capabilities.allGranted)
    }

    @Test
    func allGrantedOnlyWhenEveryPermissionIsAvailable() {
        let capabilities = PermissionCapabilities(
            accessibilityGranted: true,
            microphoneGranted: true,
            speechGranted: true
        )

        #expect(capabilities.canDictate)
        #expect(capabilities.canInsertIntoOtherApps)
        #expect(capabilities.allGranted)
    }
}
