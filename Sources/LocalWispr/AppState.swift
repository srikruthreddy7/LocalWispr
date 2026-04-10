import AppKit
import ApplicationServices
import AVFoundation
import CoreAudio
import Foundation
import OSLog
import ServiceManagement
import Speech

@MainActor
public final class AppState: ObservableObject {
    private static let logger = Logger(subsystem: "LocalWispr", category: "StopPath")
    private enum DictationTimeoutError: LocalizedError {
        case liveFinalization
        case batchFallback

        var errorDescription: String? {
            switch self {
            case .liveFinalization:
                return "Live finalization timed out"
            case .batchFallback:
                return "Batch transcription timed out"
            }
        }
    }

    private enum TranscriptResolutionError: LocalizedError {
        case emptyLiveTranscript
        case emptyBatchTranscript(TranscriberMode)

        var errorDescription: String? {
            switch self {
            case .emptyLiveTranscript:
                return "Live transcription returned no text"
            case .emptyBatchTranscript(let mode):
                return "\(mode.title) fallback returned no text"
            }
        }
    }

    private enum AudioSignalError: LocalizedError {
        case lowInputLevel(String)

        var errorDescription: String? {
            switch self {
            case .lowInputLevel(let details):
                return details
            }
        }
    }

    public static let shared = AppState()

    @Published public private(set) var state: DictationState = .idle
    @Published public private(set) var statusLine: String = "Idle"
    @Published public private(set) var modelPreparationProgress: Double?

    @Published public private(set) var lastRawTranscription: String = ""
    @Published public private(set) var lastCleanedText: String = ""
    @Published public private(set) var lastLatency: PipelineLatency?
    @Published public private(set) var latencyStats: StopLatencyStats?
    @Published public private(set) var lastStopPathDetails: StopPathDetails?
    @Published public private(set) var transcriptHistory: [TranscriptRecord] = []
    @Published public var selectedTranscriptRecordID: TranscriptRecord.ID?
    @Published public var controlPanelSection: ControlPanelSection = .dashboard

    @Published public var transcriberMode: TranscriberMode = .dictationLong

    @Published public private(set) var launchAtLoginEnabled: Bool = false
    @Published public private(set) var launchAtLoginSupported: Bool = false
    @Published public private(set) var launchAtLoginStatusMessage: String = ""
    @Published public private(set) var accessibilityPermissionGranted: Bool = false
    @Published public private(set) var microphonePermissionGranted: Bool = false
    @Published public private(set) var speechPermissionGranted: Bool = false
    @Published public private(set) var availableInputDevices: [AudioInputDeviceOption] = []
    @Published public private(set) var currentInputDeviceName: String = "Unknown Input"
    @Published public private(set) var liveInputLevelDescription: String = "Idle"
    @Published public private(set) var latestCapturedAudioURL: URL?
    @Published public private(set) var latestDebugCaptureDirectoryURL: URL?
    @Published public private(set) var latestCapturedAudioIsPlaying: Bool = false
    @Published public private(set) var audioPreviewRecordingActive: Bool = false
    @Published public var preferredInputDeviceID: UInt32? {
        didSet {
            guard preferredInputDeviceID != oldValue else { return }
            Self.savePreferredInputDeviceID(preferredInputDeviceID)
            audioCapture.preferredInputDeviceID = preferredInputDeviceID
            refreshInputDevices()
        }
    }

    @Published public private(set) var hotkeyRegistrationStatus: HotkeyRegistrationStatus = .pending
    @Published public private(set) var awaitingShortcutVerification: Bool = false
    @Published public private(set) var unavailableBindings: Set<GlobalHotkeyBinding> = []

    /// Global shortcut; persisted when not using an injected `HotkeyMonitoring` (tests).
    @Published public var globalHotkeyBinding: GlobalHotkeyBinding {
        didSet {
            guard hotkeyConfigManaged else { return }
            guard globalHotkeyBinding != oldValue else { return }
            Self.saveGlobalHotkeyBinding(globalHotkeyBinding)
            reconfigureHotkeyMonitor()
        }
    }

    @Published public var globalHotkeyInteractionMode: GlobalHotkeyInteractionMode {
        didSet {
            guard hotkeyConfigManaged else { return }
            guard globalHotkeyInteractionMode != oldValue else { return }
            Self.saveGlobalHotkeyInteractionMode(globalHotkeyInteractionMode)
            reconfigureHotkeyMonitor()
        }
    }

    private var hotkeyMonitor: HotkeyMonitoring
    private let hotkeyConfigManaged: Bool
    private let audioCapture: AudioCapturing
    private let transcriber: Transcriber
    private let pipeline: Pipeline
    private let transcriptHistoryStore: TranscriptHistoryStore
    private let projectIndex: ProjectIdentifierIndex

    private let clock = ContinuousClock()
    private var listeningStartedAt: ContinuousClock.Instant?
    private var liveTranscriptionSession: (any LiveTranscriptionSession)?
    private var activeSessionMode: TranscriberMode = .dictationLong
    private var activeDictationContext: DictationAppContext?
    private var activeContextualStrings: [String] = []
    private var audioPreviewPlayer: AVAudioPlayer?
    private var pendingDebugCaptureAudioURL: URL?
    private var stopLatencyHistory: [Int] = []

    private var hasBootstrapped = false
    private var isProcessing = false
    private var isStartingDictation = false

    private static let globalHotkeyBindingDefaultsKey = "LocalWispr.globalHotkeyBinding"
    private static let globalHotkeyInteractionModeDefaultsKey = "LocalWispr.globalHotkeyInteractionMode"
    private static let preferredInputDeviceDefaultsKey = "LocalWispr.preferredInputDeviceID"

    private var hasShownAccessibilityAlert = false
    private var shortcutVerificationWorkItem: DispatchWorkItem?

    public init(
        hotkeyMonitor: HotkeyMonitoring? = nil,
        audioCapture: AudioCapturing? = nil,
        transcriber: Transcriber? = nil,
        cleaner: (any Cleaning)? = nil,
        inserter: (any Inserting)? = nil,
        transcriptHistoryStore: TranscriptHistoryStore? = nil
    ) {
        let resolvedTranscriber = transcriber ?? Transcriber()
        let preferredInputDeviceID = Self.loadPreferredInputDeviceID()

        if let hotkeyMonitor {
            self.hotkeyMonitor = hotkeyMonitor
            self.hotkeyConfigManaged = false
            self._globalHotkeyBinding = Published(initialValue: .rightCommandDoubleTap)
            self._globalHotkeyInteractionMode = Published(initialValue: .toggle)
        } else {
            let binding = Self.loadGlobalHotkeyBinding()
            let interactionMode = Self.loadGlobalHotkeyInteractionMode(defaultBinding: binding)
            self.hotkeyMonitor = HotkeyFactory.makeMonitor(for: binding, interactionMode: interactionMode)
            self.hotkeyConfigManaged = true
            self._globalHotkeyBinding = Published(initialValue: binding)
            self._globalHotkeyInteractionMode = Published(initialValue: interactionMode)
        }
        self._preferredInputDeviceID = Published(initialValue: preferredInputDeviceID)

        let sharedProjectIndex = ProjectIdentifierIndex()
        self.projectIndex = sharedProjectIndex
        self.audioCapture = audioCapture ?? AudioCapture()
        self.audioCapture.preferredInputDeviceID = preferredInputDeviceID
        self.transcriber = resolvedTranscriber
        self.pipeline = Pipeline(
            cleaner: cleaner ?? AdaptiveTextCleaner(projectIndex: sharedProjectIndex),
            inserter: inserter ?? TextInserter()
        )
        self.transcriptHistoryStore = transcriptHistoryStore ?? TranscriptHistoryStore()

        resolvedTranscriber.onModelPreparationProgress = { [weak self] progress in
            Task { @MainActor in
                self?.modelPreparationProgress = progress.fractionCompleted
            }
        }
    }

    public func bootstrap() {
        guard !hasBootstrapped else { return }
        hasBootstrapped = true
        DebugLog.write("[AppState] bootstrap build=debug preferredInput=\(preferredInputDeviceID.map(String.init) ?? "system-default")")

        configureHotkeyCallbacks(for: hotkeyMonitor)

        refreshLaunchAtLoginStatus()

        accessibilityPermissionGranted = checkAccessibilityPermission(promptIfNeeded: false)
        refreshInputDevices()

        Task {
            await loadTranscriptHistory()
            await prewarmModels()
            await refreshAudioAndSpeechPermissions(promptIfNeeded: false, surfaceFailures: false)
            reconfigureHotkeyMonitor()
        }
    }

    public var allPermissionsGranted: Bool {
        permissionCapabilities.allGranted
    }

    public var dictationPermissionsGranted: Bool {
        permissionCapabilities.canDictate
    }

    public var canInsertIntoOtherApps: Bool {
        permissionCapabilities.canInsertIntoOtherApps
    }

    public var isBusy: Bool {
        switch state {
        case .finalizingTranscript, .cleaning, .inserting:
            return true
        default:
            return false
        }
    }

    public func requestAllPermissions() {
        accessibilityPermissionGranted = checkAccessibilityPermission(promptIfNeeded: true)
        if !accessibilityPermissionGranted {
            presentAccessibilityAlert()
        }

        Task {
            await refreshAudioAndSpeechPermissions(promptIfNeeded: true, surfaceFailures: true)
            reconfigureHotkeyMonitor()
        }
    }

    /// Call when the app becomes active (e.g. after editing Accessibility in System Settings) to refresh trust and shortcut registration.
    public func refreshAccessibilityAndHotkey() {
        accessibilityPermissionGranted = checkAccessibilityPermission(promptIfNeeded: false)
        reconfigureHotkeyMonitor()
    }

    public func openAccessibilitySettings() {
        if let url = URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility") {
            NSWorkspace.shared.open(url)
        }
    }

    public func openMicrophoneSettings() {
        if let url = URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone") {
            NSWorkspace.shared.open(url)
        }
    }

    public func openSpeechSettings() {
        if let url = URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_SpeechRecognition") {
            NSWorkspace.shared.open(url)
        }
    }

    public func toggleDictation() {
        confirmShortcutVerified()
        Task {
            guard !audioPreviewRecordingActive else { return }
            if state == .listening {
                await stopDictation()
            } else {
                await startDictation()
            }
        }
    }

    public func toggleAudioPreviewCapture() {
        confirmShortcutVerified()
        Task {
            if audioPreviewRecordingActive {
                await stopAudioPreviewCapture()
            } else {
                await startAudioPreviewCapture()
            }
        }
    }

    private func startDictationFromHotkey() {
        confirmShortcutVerified()
        Task {
            guard state != .listening, state != .recordingAudio else { return }
            await startDictation()
        }
    }

    private func stopDictationFromHotkey() {
        confirmShortcutVerified()
        Task {
            if state == .listening {
                await stopDictation()
                return
            }

            // Press-and-hold can release before start transitions to `.listening`.
            try? await Task.sleep(nanoseconds: 250_000_000)
            if state == .listening {
                await stopDictation()
            }
        }
    }

    private func startAudioPreviewCapture() async {
        guard !isProcessing, !audioPreviewRecordingActive, state != .listening else { return }
        guard await ensurePermissionsForDictation() else { return }

        refreshInputDeviceName()
        liveInputLevelDescription = "Armed"
        stopLatestCapturedAudioPlayback()

        audioCapture.onBufferCaptured = { buffer in
            if let levelSummary = Self.inputLevelSummary(for: buffer) {
                Task { @MainActor [weak self] in
                    self?.liveInputLevelDescription = levelSummary
                }
            }
        }

        do {
            try audioCapture.start()
            playStartSound()
            audioPreviewRecordingActive = true
            statusLine = "Recording audio only"
            transition(to: .recordingAudio)
        } catch {
            audioCapture.onBufferCaptured = nil
            setError("Unable to start audio recording: \(error.localizedDescription)")
        }
    }

    private func stopAudioPreviewCapture() async {
        guard audioPreviewRecordingActive else { return }

        let buffers: [AVAudioPCMBuffer]
        do {
            buffers = try audioCapture.stopAndDrain()
        } catch {
            audioCapture.onBufferCaptured = nil
            audioPreviewRecordingActive = false
            setError("Unable to stop audio recording: \(error.localizedDescription)")
            return
        }

        audioCapture.onBufferCaptured = nil
        playStopSound()
        pendingDebugCaptureAudioURL = dumpCapturedAudioIfEnabled(buffers)
        audioPreviewRecordingActive = false

        if let audioURL = pendingDebugCaptureAudioURL {
            persistAudioOnlyDebugCaptureArtifacts(audioURL: audioURL)
            statusLine = "Audio recording saved"
        } else {
            statusLine = "Audio recording unavailable"
        }

        transition(to: .idle)
    }

    private func configureHotkeyCallbacks(for monitor: HotkeyMonitoring) {
        monitor.onToggleRequested = { [weak self] in
            self?.toggleDictation()
        }
        monitor.onStartRequested = { [weak self] in
            self?.startDictationFromHotkey()
        }
        monitor.onStopRequested = { [weak self] in
            self?.stopDictationFromHotkey()
        }
    }

    public func showControlPanel() {
        NotificationCenter.default.post(name: .localWisprShowControlPanel, object: nil)
    }

    public func setLaunchAtLogin(_ enabled: Bool) {
        guard launchAtLoginSupported else { return }

        do {
            if enabled {
                try SMAppService.mainApp.register()
            } else {
                try SMAppService.mainApp.unregister()
            }
            refreshLaunchAtLoginStatus()
        } catch {
            launchAtLoginStatusMessage = "Launch at login update failed: \(error.localizedDescription)"
            refreshLaunchAtLoginStatus()
        }
    }

    public func quit() {
        NSApplication.shared.terminate(nil)
    }

    public var hasLastRawTranscription: Bool {
        !lastRawTranscription.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    public var hasLastCleanedText: Bool {
        !lastCleanedText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    public var selectedTranscriptRecord: TranscriptRecord? {
        guard let selectedTranscriptRecordID else {
            return transcriptHistory.first
        }

        return transcriptHistory.first(where: { $0.id == selectedTranscriptRecordID }) ?? transcriptHistory.first
    }

    public func copyLastTranscription() {
        let text = lastRawTranscription.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }
        copyToPasteboard(text)
    }

    public func copyLastCleanedText() {
        let text = lastCleanedText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }
        copyToPasteboard(text)
    }

    public func copyLatestResult() {
        let cleaned = lastCleanedText.trimmingCharacters(in: .whitespacesAndNewlines)
        if !cleaned.isEmpty {
            copyToPasteboard(cleaned)
            return
        }

        let raw = lastRawTranscription.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !raw.isEmpty else { return }
        copyToPasteboard(raw)
    }

    private static func loadGlobalHotkeyBinding() -> GlobalHotkeyBinding {
        guard let raw = UserDefaults.standard.string(forKey: Self.globalHotkeyBindingDefaultsKey),
              let value = GlobalHotkeyBinding(rawValue: raw) else {
            return .controlOptionSpace
        }
        return value
    }

    private static func saveGlobalHotkeyBinding(_ binding: GlobalHotkeyBinding) {
        UserDefaults.standard.set(binding.rawValue, forKey: Self.globalHotkeyBindingDefaultsKey)
    }

    private static func loadGlobalHotkeyInteractionMode(defaultBinding: GlobalHotkeyBinding) -> GlobalHotkeyInteractionMode {
        if let raw = UserDefaults.standard.string(forKey: Self.globalHotkeyInteractionModeDefaultsKey),
           let value = GlobalHotkeyInteractionMode(rawValue: raw) {
            return value
        }

        // Keep legacy behavior for existing command-key users if no explicit mode was persisted.
        if defaultBinding == .rightCommandDoubleTap || defaultBinding == .leftCommandDoubleTap {
            return .doubleTap
        }

        return .toggle
    }

    private static func saveGlobalHotkeyInteractionMode(_ mode: GlobalHotkeyInteractionMode) {
        UserDefaults.standard.set(mode.rawValue, forKey: Self.globalHotkeyInteractionModeDefaultsKey)
    }

    private static func loadPreferredInputDeviceID() -> UInt32? {
        guard UserDefaults.standard.object(forKey: Self.preferredInputDeviceDefaultsKey) != nil else {
            return nil
        }
        return UInt32(UserDefaults.standard.integer(forKey: Self.preferredInputDeviceDefaultsKey))
    }

    private static func savePreferredInputDeviceID(_ deviceID: UInt32?) {
        if let deviceID {
            UserDefaults.standard.set(Int(deviceID), forKey: Self.preferredInputDeviceDefaultsKey)
        } else {
            UserDefaults.standard.removeObject(forKey: Self.preferredInputDeviceDefaultsKey)
        }
    }

    public func selectInputDevice(_ deviceID: UInt32?) {
        preferredInputDeviceID = deviceID
    }

    public func refreshInputDevices() {
        availableInputDevices = Self.availableInputDeviceOptions()
        refreshInputDeviceName()
    }

    private func reconfigureHotkeyMonitor() {
        guard hotkeyConfigManaged else { return }

        hotkeyMonitor.onToggleRequested = nil
        hotkeyMonitor.onStartRequested = nil
        hotkeyMonitor.onStopRequested = nil
        hotkeyMonitor.stop()
        awaitingShortcutVerification = false
        shortcutVerificationWorkItem?.cancel()
        shortcutVerificationWorkItem = nil
        unavailableBindings = []

        guard globalHotkeyBinding != .none else {
            hotkeyMonitor = HotkeyFactory.makeMonitor(for: .none, interactionMode: globalHotkeyInteractionMode)
            accessibilityPermissionGranted = checkAccessibilityPermission(promptIfNeeded: false)
            hotkeyRegistrationStatus = .inactive("Shortcut disabled in settings")
            return
        }

        accessibilityPermissionGranted = checkAccessibilityPermission(promptIfNeeded: false)
        guard accessibilityPermissionGranted else {
            hotkeyMonitor = HotkeyFactory.makeMonitor(for: .none, interactionMode: globalHotkeyInteractionMode)
            hotkeyRegistrationStatus = .failed("Accessibility permission is required for global shortcut capture")
            return
        }

        let monitor = HotkeyFactory.makeMonitor(
            for: globalHotkeyBinding,
            interactionMode: globalHotkeyInteractionMode
        )

        do {
            try monitor.start()
            hotkeyMonitor = monitor
            configureHotkeyCallbacks(for: hotkeyMonitor)
            hotkeyRegistrationStatus = .listening("\(globalHotkeyBinding.menuTitle) (\(globalHotkeyInteractionMode.statusSuffix))")
            beginShortcutVerification()
        } catch {
            monitor.stop()
            hotkeyMonitor = HotkeyFactory.makeMonitor(for: .none, interactionMode: globalHotkeyInteractionMode)
            let message = error.localizedDescription.trimmingCharacters(in: .whitespacesAndNewlines)
            hotkeyRegistrationStatus = .failed(message.isEmpty ? "Unable to register shortcut" : message)
        }
    }

    private func beginShortcutVerification() {
        awaitingShortcutVerification = true

        let workItem = DispatchWorkItem { [weak self] in
            Task { @MainActor in
                self?.awaitingShortcutVerification = false
            }
        }
        shortcutVerificationWorkItem = workItem
        DispatchQueue.main.asyncAfter(deadline: .now() + 8, execute: workItem)
    }

    /// Called when the shortcut actually fires — confirms it's working.
    private func confirmShortcutVerified() {
        if awaitingShortcutVerification {
            awaitingShortcutVerification = false
            shortcutVerificationWorkItem?.cancel()
            shortcutVerificationWorkItem = nil
        }
    }

    private func checkAccessibilityPermission(promptIfNeeded: Bool) -> Bool {
        if promptIfNeeded {
            let promptKey = "AXTrustedCheckOptionPrompt" as CFString
            let options = [promptKey: true] as CFDictionary
            return AXIsProcessTrustedWithOptions(options)
        }
        return AXIsProcessTrusted()
    }

    private func presentAccessibilityAlert() {
        guard !hasShownAccessibilityAlert else { return }
        hasShownAccessibilityAlert = true

        let alert = NSAlert()
        alert.messageText = "Accessibility Access Required"
        alert.informativeText = "LocalWispr needs Accessibility access to register global shortcuts and paste text into other apps."
        alert.alertStyle = .warning
        alert.addButton(withTitle: "Open Settings")
        alert.addButton(withTitle: "Later")

        NSApplication.shared.activate(ignoringOtherApps: true)
        let response = alert.runModal()

        if response == .alertFirstButtonReturn,
           let url = URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility") {
            NSWorkspace.shared.open(url)
        }
    }

    private var permissionCapabilities: PermissionCapabilities {
        PermissionCapabilities(
            accessibilityGranted: accessibilityPermissionGranted,
            microphoneGranted: microphonePermissionGranted,
            speechGranted: speechPermissionGranted
        )
    }

    private func refreshAudioAndSpeechPermissions(promptIfNeeded: Bool, surfaceFailures: Bool) async {
        microphonePermissionGranted = await requestMicrophoneAccessIfNeeded(promptIfNeeded: promptIfNeeded)
        speechPermissionGranted = await requestSpeechAuthorizationIfNeeded(promptIfNeeded: promptIfNeeded)
        refreshInputDevices()

        guard surfaceFailures else { return }

        if !microphonePermissionGranted {
            setError("Microphone permission denied. Click Grant Permissions.")
            return
        }

        if !speechPermissionGranted {
            setError("Speech recognition permission denied. Click Grant Permissions.")
            return
        }

        if state == .error("Microphone permission denied. Click Grant Permissions.")
            || state == .error("Speech recognition permission denied. Click Grant Permissions.") {
            transition(to: .idle)
        }
    }

    private func requestMicrophoneAccessIfNeeded(promptIfNeeded: Bool) async -> Bool {
        switch Self.microphoneAuthorizationStatus() {
        case .authorized:
            return true
        case .notDetermined:
            return promptIfNeeded ? await Self.requestMicrophoneAccess() : false
        case .denied, .restricted:
            return false
        @unknown default:
            return false
        }
    }

    private func requestSpeechAuthorizationIfNeeded(promptIfNeeded: Bool) async -> Bool {
        switch Self.speechAuthorizationStatus() {
        case .authorized:
            return true
        case .notDetermined:
            let status = promptIfNeeded ? await Self.requestSpeechAuthorization() : .notDetermined
            return status == .authorized
        case .denied, .restricted:
            return false
        @unknown default:
            return false
        }
    }

    private nonisolated static func microphoneAuthorizationStatus() -> AVAuthorizationStatus {
        AVCaptureDevice.authorizationStatus(for: .audio)
    }

    private nonisolated static func speechAuthorizationStatus() -> SFSpeechRecognizerAuthorizationStatus {
        SFSpeechRecognizer.authorizationStatus()
    }

    private nonisolated static func requestMicrophoneAccess() async -> Bool {
        await withCheckedContinuation { continuation in
            AVCaptureDevice.requestAccess(for: .audio) { granted in
                continuation.resume(returning: granted)
            }
        }
    }

    private nonisolated static func requestSpeechAuthorization() async -> SFSpeechRecognizerAuthorizationStatus {
        await withCheckedContinuation { continuation in
            SFSpeechRecognizer.requestAuthorization { authorizationStatus in
                continuation.resume(returning: authorizationStatus)
            }
        }
    }

    private func ensurePermissionsForDictation() async -> Bool {
        accessibilityPermissionGranted = checkAccessibilityPermission(promptIfNeeded: false)

        if !dictationPermissionsGranted {
            await refreshAudioAndSpeechPermissions(promptIfNeeded: true, surfaceFailures: true)
        }

        if !dictationPermissionsGranted {
            return false
        }

        return true
    }

    private func prewarmModels() async {
        let format = audioCapture.outputAudioFormat
        await transcriber.prewarm(mode: .dictationLong, locale: .current, audioFormat: format)
        await transcriber.prewarm(mode: .speechTranscription, locale: .current, audioFormat: format)
    }

    private func startDictation() async {
        guard !isProcessing,
              !isStartingDictation,
              !audioPreviewRecordingActive,
              state != .listening,
              liveTranscriptionSession == nil else {
            DebugLog.write("[AppState] startDictation skipped: isProcessing=\(isProcessing) isStarting=\(isStartingDictation) state=\(state) hasLiveSession=\(liveTranscriptionSession != nil)")
            return
        }

        isStartingDictation = true
        defer { isStartingDictation = false }

        guard await ensurePermissionsForDictation() else { return }

        let mode = transcriberMode
        let contextEnabled = ContextUsagePolicy.isEnabled(environment: DotEnv.merged())
        DebugLog.write("[AppState] startDictation: mode=\(mode)")
        refreshInputDeviceName()
        liveInputLevelDescription = "Armed"
        activeDictationContext = nil
        activeContextualStrings = []

        let contextualStrings: [String]
        if contextEnabled {
            let appContext = AppContextCapture.captureForDictationStart()
            activeDictationContext = appContext
            await SessionContextStore.shared.set(appContext)
            DebugLog.write("[AppState] context: \(Self.debugContextSummary(appContext))")
            contextualStrings = await projectIndex.tieredIdentifiers(context: appContext, limit: 100)
            DebugLog.write("[AppState] contextual hints: count=\(contextualStrings.count) top=\(Array(contextualStrings.prefix(12)))")
        } else {
            await SessionContextStore.shared.clear()
            DebugLog.write("[AppState] context disabled for live dictation")
            contextualStrings = []
        }

        do {
            let session = try await transcriber.startSession(
                mode: mode,
                locale: .current,
                audioFormat: audioCapture.outputAudioFormat,
                contextualStrings: contextualStrings
            )

            playStartSound()
            try await Task.sleep(nanoseconds: 250_000_000)

            audioCapture.onBufferCaptured = { buffer in
                session.append(buffer)
                if let levelSummary = Self.inputLevelSummary(for: buffer) {
                    Task { @MainActor [weak self] in
                        self?.liveInputLevelDescription = levelSummary
                    }
                }
            }

            do {
                try audioCapture.start()
            } catch {
                audioCapture.onBufferCaptured = nil
                await session.cancel()
                throw error
            }

            liveTranscriptionSession = session
            activeSessionMode = mode
            activeContextualStrings = contextualStrings
            listeningStartedAt = clock.now

            modelPreparationProgress = nil
            transition(to: .listening)
        } catch {
            await SessionContextStore.shared.clear()
            activeDictationContext = nil
            activeContextualStrings = []
            setError("Unable to start live transcription: \(error.localizedDescription)")
        }
    }

    private func stopDictation() async {
        DebugLog.write("[AppState] stopDictation called")
        guard !isProcessing else { DebugLog.write("[AppState] already processing, skipping"); return }

        let stopPressedAt = clock.now
        let recordingDurationMilliseconds = listeningStartedAt.map { durationToMilliseconds($0.duration(to: stopPressedAt)) }
        listeningStartedAt = nil

        let activeSession = liveTranscriptionSession
        liveTranscriptionSession = nil
        let sessionContextualStrings = activeContextualStrings
        activeContextualStrings = []

        let modeUsedForSession = activeSessionMode
        var resolvedMode = modeUsedForSession

        let buffers: [AVAudioPCMBuffer]
        let drainStartedAt = clock.now
        do {
            buffers = try audioCapture.stopAndDrain()
        } catch {
            audioCapture.onBufferCaptured = nil
            if let activeSession {
                await activeSession.cancel()
            }
            await SessionContextStore.shared.clear()
            activeDictationContext = nil
            setError("Unable to stop microphone capture: \(error.localizedDescription)")
            return
        }
        let drainMilliseconds = durationToMilliseconds(drainStartedAt.duration(to: clock.now))
        audioCapture.onBufferCaptured = nil
        playStopSound()
        pendingDebugCaptureAudioURL = dumpCapturedAudioIfEnabled(buffers)

        isProcessing = true
        modelPreparationProgress = nil
        transition(to: .finalizingTranscript)

        let transcriptStartedAt = clock.now
        var rawText: String?
        var transcriptionError: Error?
        var liveFinalizationMilliseconds: Int?
        var batchFallbackMilliseconds: Int?
        var transcriptSource: TranscriptResolutionSource = .unavailable
        let lowSignalDiagnosis = lowSignalMessage(for: buffers)

        if let activeSession {
            let liveFinalizationStartedAt = clock.now
            do {
                let liveTranscript = try await runWithTimeout(seconds: 4) {
                    try await activeSession.finish()
                } onTimeout: {
                    DictationTimeoutError.liveFinalization
                }
                liveFinalizationMilliseconds = durationToMilliseconds(liveFinalizationStartedAt.duration(to: clock.now))
                rawText = TranscriptResolutionPolicy.normalizedTranscript(liveTranscript)
                if rawText != nil {
                    transcriptSource = .liveFinalization
                } else {
                    transcriptionError = TranscriptResolutionError.emptyLiveTranscript
                }
            } catch {
                liveFinalizationMilliseconds = durationToMilliseconds(liveFinalizationStartedAt.duration(to: clock.now))
                transcriptionError = error
                await activeSession.cancel()
            }
        }

        let shouldAttemptBatchVerification = TranscriptResolutionPolicy.shouldAttemptBatchVerification(
            liveTranscript: rawText,
            recordingDurationMilliseconds: recordingDurationMilliseconds
        )
        let verifyingExistingLiveTranscript = rawText != nil && shouldAttemptBatchVerification

        if (!buffers.isEmpty) && (rawText == nil || shouldAttemptBatchVerification) {
            let fallbackStartedAt = clock.now
            for fallbackMode in TranscriptResolutionPolicy.fallbackModes(after: modeUsedForSession) {
                do {
                    let fallbackTranscript = try await runWithTimeout(seconds: 10) {
                        try await self.transcriber.transcribe(
                            buffers: buffers,
                            mode: fallbackMode,
                            locale: .current,
                            contextualStrings: sessionContextualStrings
                        )
                    } onTimeout: {
                        DictationTimeoutError.batchFallback
                    }

                    if let normalizedTranscript = TranscriptResolutionPolicy.normalizedTranscript(fallbackTranscript) {
                        let existingTranscript = rawText
                        let preferredTranscript = TranscriptResolutionPolicy.preferredTranscript(
                            primary: existingTranscript,
                            alternative: normalizedTranscript
                        )

                        if existingTranscript == nil || preferredTranscript != existingTranscript {
                            rawText = preferredTranscript
                            resolvedMode = fallbackMode
                            transcriptSource = .batchFallback
                        }

                        if rawText != nil && !verifyingExistingLiveTranscript {
                            break
                        }
                    }

                    transcriptionError = TranscriptResolutionError.emptyBatchTranscript(fallbackMode)
                } catch {
                    transcriptionError = error
                }
            }
            batchFallbackMilliseconds = durationToMilliseconds(fallbackStartedAt.duration(to: clock.now))
        }

        if let lowSignalDiagnosis {
            DebugLog.write("[AppState] low signal detected: \(lowSignalDiagnosis)")
            rawText = nil
            transcriptionError = AudioSignalError.lowInputLevel(lowSignalDiagnosis)
            transcriptSource = .unavailable
        }

        lastRawTranscription = rawText ?? ""

        let stopToTranscriptMilliseconds = durationToMilliseconds(transcriptStartedAt.duration(to: clock.now))
        let stopPathDetails = StopPathDetails(
            source: transcriptSource,
            bufferCount: buffers.count,
            drainMilliseconds: drainMilliseconds,
            liveFinalizationMilliseconds: liveFinalizationMilliseconds,
            batchFallbackMilliseconds: batchFallbackMilliseconds,
            liveFailureDescription: transcriptionError?.localizedDescription
        )
        lastStopPathDetails = stopPathDetails
        Self.logger.info(
            "Stop path source=\(stopPathDetails.source.rawValue, privacy: .public) buffers=\(stopPathDetails.bufferCount, privacy: .public) drain=\(stopPathDetails.drainMilliseconds, privacy: .public)ms live=\(stopPathDetails.liveFinalizationMilliseconds ?? -1, privacy: .public)ms batch=\(stopPathDetails.batchFallbackMilliseconds ?? -1, privacy: .public)ms liveFailure=\(stopPathDetails.liveFailureDescription ?? "none", privacy: .public)"
        )

        DebugLog.write("[AppState] transcription done: rawText=\(rawText?.prefix(100) ?? "nil") source=\(transcriptSource) mode=\(resolvedMode)")

        guard let rawText else {
            isProcessing = false
            await SessionContextStore.shared.clear()
            activeDictationContext = nil
            let latency = PipelineLatency(
                stopToTranscriptMilliseconds: stopToTranscriptMilliseconds,
                cleanupMilliseconds: 0,
                insertionMilliseconds: 0,
                recordingDurationMilliseconds: recordingDurationMilliseconds
            )
            handlePipelineResult(
                .failed(
                    raw: nil,
                    cleaned: nil,
                    error: "Transcription failed: \(transcriptionError?.localizedDescription ?? "Unknown error")",
                    latency: latency
                ),
                modeUsedForSession: modeUsedForSession,
                stopPathDetails: stopPathDetails
            )
            return
        }

        let result = await pipeline.process(
            rawText: rawText,
            stopToTranscriptMilliseconds: stopToTranscriptMilliseconds,
            recordingDurationMilliseconds: recordingDurationMilliseconds
        ) { [weak self] stage in
            Task { @MainActor in
                self?.applyPipelineStage(stage)
            }
        }

        DebugLog.write("[AppState] pipeline result: \(result)")
        isProcessing = false
        await SessionContextStore.shared.clear()
        activeDictationContext = nil
        handlePipelineResult(result, modeUsedForSession: resolvedMode, stopPathDetails: stopPathDetails)
    }

    private func runWithTimeout<T: Sendable>(
        seconds: Double,
        operation: @escaping @Sendable () async throws -> T,
        onTimeout: @escaping @Sendable () -> Error
    ) async throws -> T {
        try await withThrowingTaskGroup(of: T.self) { group in
            group.addTask {
                try await operation()
            }
            group.addTask {
                let duration = UInt64(max(0, seconds) * 1_000_000_000)
                try await Task.sleep(nanoseconds: duration)
                throw onTimeout()
            }

            defer {
                group.cancelAll()
            }

            guard let firstFinished = try await group.next() else {
                throw onTimeout()
            }
            return firstFinished
        }
    }

    private func dumpCapturedAudioIfEnabled(_ buffers: [AVAudioPCMBuffer]) -> URL? {
        #if DEBUG
        let shouldDumpAudio = true
        #else
        let environment = DotEnv.merged()
        let rawValue = environment["LOCALWISPR_DUMP_AUDIO"]?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let shouldDumpAudio = rawValue.map { ["1", "true", "yes", "on"].contains($0) } ?? false
        #endif
        guard shouldDumpAudio else { return nil }
        guard let firstBuffer = buffers.first else { return nil }

        let formatter = DateFormatter()
        formatter.dateFormat = "yyyyMMdd-HHmmss"
        let stamp = formatter.string(from: Date())
        let directoryURL = URL(fileURLWithPath: "/tmp/localwispr-debug-captures", isDirectory: true)
            .appendingPathComponent("session-\(stamp)", isDirectory: true)
        let url = directoryURL.appendingPathComponent("audio.wav")

        do {
            try FileManager.default.createDirectory(at: directoryURL, withIntermediateDirectories: true)
            let file = try AVAudioFile(
                forWriting: url,
                settings: firstBuffer.format.settings,
                commonFormat: firstBuffer.format.commonFormat,
                interleaved: firstBuffer.format.isInterleaved
            )
            for buffer in buffers where buffer.frameLength > 0 {
                try file.write(from: buffer)
            }
            DebugLog.write("[AppState] dumped captured audio to \(url.path)")
            latestCapturedAudioURL = url
            latestDebugCaptureDirectoryURL = directoryURL
            latestCapturedAudioIsPlaying = false
            audioPreviewPlayer?.stop()
            audioPreviewPlayer = nil
            return url
        } catch {
            DebugLog.write("[AppState] failed to dump captured audio: \(error.localizedDescription)")
            return nil
        }
    }

    private func applyPipelineStage(_ stage: PipelineStage) {
        switch stage {
        case .cleaning:
            transition(to: .cleaning)
        case .inserting:
            transition(to: .inserting)
        }
    }

    private func handlePipelineResult(
        _ result: PipelineResult,
        modeUsedForSession: TranscriberMode,
        stopPathDetails: StopPathDetails?
    ) {
        persistDebugCaptureArtifactsIfNeeded(
            for: result,
            modeUsedForSession: modeUsedForSession,
            stopPathDetails: stopPathDetails
        )

        switch result {
        case .noSpeech(let latency):
            lastLatency = latency
            recordLatency(latency)
            transition(to: .noSpeech)
            statusLine = "No speech detected | \(latency.formattedSummary)"
        case .inserted(let raw, let cleaned, let warning, let latency):
            lastLatency = latency
            recordLatency(latency)
            lastRawTranscription = raw
            lastCleanedText = cleaned
            transition(to: .idle)
            if let warning, !warning.isEmpty {
                statusLine = "\(warning) | \(latency.formattedSummary)"
            } else {
                statusLine = latency.formattedSummary
            }
        case .failed(let raw, let cleaned, let error, let latency):
            lastLatency = latency
            recordLatency(latency)
            if let raw {
                lastRawTranscription = raw
            }
            if let cleaned {
                lastCleanedText = cleaned
            }
            setError("\(error) | \(latency.formattedSummary)")
        }

        persistTranscriptRecord(
            from: result,
            modeUsedForSession: modeUsedForSession,
            localeIdentifier: Locale.current.identifier,
            stopPathDetails: stopPathDetails
        )
    }

    private func loadTranscriptHistory() async {
        do {
            let records = try await transcriptHistoryStore.loadRecords()
            applyTranscriptHistory(records)
        } catch {
            Self.logger.error("Transcript history load failed: \(error.localizedDescription, privacy: .public)")
        }
    }

    private func persistDebugCaptureArtifactsIfNeeded(
        for result: PipelineResult,
        modeUsedForSession: TranscriberMode,
        stopPathDetails: StopPathDetails?
    ) {
        guard let audioURL = pendingDebugCaptureAudioURL else { return }
        pendingDebugCaptureAudioURL = nil

        let directoryURL = audioURL.deletingLastPathComponent()

        do {
            let metadata = DebugCaptureMetadata(
                createdAt: Date(),
                mode: modeUsedForSession.rawValue,
                inputDeviceName: currentInputDeviceName,
                inputLevel: liveInputLevelDescription,
                audioPath: audioURL.path,
                startContext: activeDictationContext.map(Self.debugContextPayload(from:)),
                contextualHints: Array(activeContextualStrings.prefix(40)),
                result: Self.debugResultPayload(from: result),
                stopPath: stopPathDetails
            )

            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            encoder.dateEncodingStrategy = .iso8601
            try encoder.encode(metadata).write(
                to: directoryURL.appendingPathComponent("session.json"),
                options: .atomic
            )

            switch result {
            case .inserted(let raw, let cleaned, _, _):
                try raw.write(to: directoryURL.appendingPathComponent("raw.txt"), atomically: true, encoding: .utf8)
                try cleaned.write(to: directoryURL.appendingPathComponent("cleaned.txt"), atomically: true, encoding: .utf8)
            case .failed(let raw, let cleaned, let error, _):
                if let raw {
                    try raw.write(to: directoryURL.appendingPathComponent("raw.txt"), atomically: true, encoding: .utf8)
                }
                if let cleaned {
                    try cleaned.write(to: directoryURL.appendingPathComponent("cleaned.txt"), atomically: true, encoding: .utf8)
                }
                try error.write(to: directoryURL.appendingPathComponent("error.txt"), atomically: true, encoding: .utf8)
            case .noSpeech:
                break
            }

            DebugLog.write("[AppState] wrote debug capture artifacts to \(directoryURL.path)")
        } catch {
            DebugLog.write("[AppState] failed to write debug capture artifacts: \(error.localizedDescription)")
        }
    }

    private func persistAudioOnlyDebugCaptureArtifacts(audioURL: URL) {
        pendingDebugCaptureAudioURL = nil
        let directoryURL = audioURL.deletingLastPathComponent()

        do {
            let metadata = DebugCaptureMetadata(
                createdAt: Date(),
                mode: "audio-only",
                inputDeviceName: currentInputDeviceName,
                inputLevel: liveInputLevelDescription,
                audioPath: audioURL.path,
                startContext: nil,
                contextualHints: [],
                result: DebugCaptureResultPayload(
                    kind: "audioOnly",
                    rawText: nil,
                    cleanedText: nil,
                    error: nil,
                    warning: nil,
                    latency: PipelineLatency(
                        stopToTranscriptMilliseconds: 0,
                        cleanupMilliseconds: 0,
                        insertionMilliseconds: 0,
                        recordingDurationMilliseconds: nil
                    )
                ),
                stopPath: nil
            )

            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            encoder.dateEncodingStrategy = .iso8601
            try encoder.encode(metadata).write(
                to: directoryURL.appendingPathComponent("session.json"),
                options: .atomic
            )

            DebugLog.write("[AppState] wrote audio-only debug capture artifacts to \(directoryURL.path)")
        } catch {
            DebugLog.write("[AppState] failed to write audio-only debug capture artifacts: \(error.localizedDescription)")
        }
    }

    private static func debugResultPayload(from result: PipelineResult) -> DebugCaptureResultPayload {
        switch result {
        case .noSpeech(let latency):
            return DebugCaptureResultPayload(
                kind: "noSpeech",
                rawText: nil,
                cleanedText: nil,
                error: nil,
                warning: nil,
                latency: latency
            )
        case .inserted(let raw, let cleaned, let warning, let latency):
            return DebugCaptureResultPayload(
                kind: "inserted",
                rawText: raw,
                cleanedText: cleaned,
                error: nil,
                warning: warning,
                latency: latency
            )
        case .failed(let raw, let cleaned, let error, let latency):
            return DebugCaptureResultPayload(
                kind: "failed",
                rawText: raw,
                cleanedText: cleaned,
                error: error,
                warning: nil,
                latency: latency
            )
        }
    }

    private func persistTranscriptRecord(
        from result: PipelineResult,
        modeUsedForSession: TranscriberMode,
        localeIdentifier: String,
        stopPathDetails: StopPathDetails?
    ) {
        guard let record = makeTranscriptRecord(
            from: result,
            modeUsedForSession: modeUsedForSession,
            localeIdentifier: localeIdentifier,
            stopPathDetails: stopPathDetails
        ) else {
            return
        }

        Task { [weak self] in
            guard let self else { return }

            do {
                let updatedRecords = try await self.transcriptHistoryStore.append(record)
                await MainActor.run {
                    self.applyTranscriptHistory(updatedRecords)
                }
            } catch {
                Self.logger.error("Transcript history append failed: \(error.localizedDescription, privacy: .public)")
            }
        }
    }

    private func makeTranscriptRecord(
        from result: PipelineResult,
        modeUsedForSession: TranscriberMode,
        localeIdentifier: String,
        stopPathDetails: StopPathDetails?
    ) -> TranscriptRecord? {
        switch result {
        case .noSpeech:
            return nil
        case .inserted(let raw, let cleaned, let warning, let latency):
            return TranscriptRecord(
                transcriberMode: modeUsedForSession,
                localeIdentifier: localeIdentifier,
                rawText: raw,
                cleanedText: cleaned,
                outcome: .inserted,
                cleanupWarning: warning,
                latency: latency,
                stopPath: stopPathDetails
            )
        case .failed(let raw, let cleaned, let error, let latency):
            let outcome: TranscriptRecordOutcome = error.hasPrefix("Text insertion failed:")
                ? .insertionFailed
                : .transcriptionFailed

            return TranscriptRecord(
                transcriberMode: modeUsedForSession,
                localeIdentifier: localeIdentifier,
                rawText: raw ?? "",
                cleanedText: cleaned ?? "",
                outcome: outcome,
                errorMessage: error,
                latency: latency,
                stopPath: stopPathDetails
            )
        }
    }

    private func applyTranscriptHistory(_ records: [TranscriptRecord]) {
        transcriptHistory = records
        rebuildLatencyStats(from: records)

        if let selectedTranscriptRecordID,
           records.contains(where: { $0.id == selectedTranscriptRecordID }) {
            return
        }

        self.selectedTranscriptRecordID = records.first?.id
    }

    private func recordLatency(_ latency: PipelineLatency) {
        let total = latency.totalStopToInsertMilliseconds
        guard total > 0 else { return }

        stopLatencyHistory.append(total)
        if stopLatencyHistory.count > 20 {
            stopLatencyHistory.removeFirst(stopLatencyHistory.count - 20)
        }

        refreshLatencyStats()
    }

    private func rebuildLatencyStats(from records: [TranscriptRecord]) {
        stopLatencyHistory = Self.recentLatencyTotals(from: records)

        refreshLatencyStats()
    }

    private func refreshLatencyStats() {
        latencyStats = StopLatencyStats.fromRecentTotals(stopLatencyHistory)
    }

    nonisolated static func recentLatencyTotals(from records: [TranscriptRecord], limit: Int = 20) -> [Int] {
        Array(
            records
                .sorted(by: { $0.createdAt > $1.createdAt })
                .compactMap { record in
                    guard let latency = record.latency else { return nil }
                    let total = latency.totalStopToInsertMilliseconds
                    return total > 0 ? total : nil
                }
                .prefix(limit)
        )
    }

    private func durationToMilliseconds(_ duration: Duration) -> Int {
        let components = duration.components
        let milliseconds = (Double(components.seconds) * 1_000.0) + (Double(components.attoseconds) / 1_000_000_000_000_000.0)
        return max(0, Int(milliseconds.rounded()))
    }

    private func refreshInputDeviceName() {
        if let preferredInputDeviceID,
           let selectedOption = availableInputDevices.first(where: { $0.deviceID == preferredInputDeviceID }) {
            currentInputDeviceName = selectedOption.name
            return
        }

        currentInputDeviceName = Self.defaultInputDeviceName() ?? "Unknown Input"
    }

    private nonisolated static func defaultInputDeviceID() -> AudioDeviceID? {
        var deviceID = AudioDeviceID(0)
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDefaultInputDevice,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        var size = UInt32(MemoryLayout<AudioDeviceID>.size)

        let deviceStatus = AudioObjectGetPropertyData(
            AudioObjectID(kAudioObjectSystemObject),
            &address,
            0,
            nil,
            &size,
            &deviceID
        )
        guard deviceStatus == noErr, deviceID != kAudioObjectUnknown else {
            return nil
        }
        return deviceID
    }

    private nonisolated static func defaultInputDeviceName() -> String? {
        guard let deviceID = defaultInputDeviceID() else {
            return nil
        }
        return audioObjectStringProperty(selector: kAudioObjectPropertyName, objectID: deviceID)
    }

    private nonisolated static func availableInputDeviceOptions() -> [AudioInputDeviceOption] {
        let defaultDeviceID = defaultInputDeviceID()
        let devices = availableInputDeviceIDs()
        let sortedDevices = devices.sorted { lhs, rhs in
            let leftName = audioObjectStringProperty(selector: kAudioObjectPropertyName, objectID: lhs) ?? ""
            let rightName = audioObjectStringProperty(selector: kAudioObjectPropertyName, objectID: rhs) ?? ""
            return leftName.localizedCaseInsensitiveCompare(rightName) == .orderedAscending
        }

        var options: [AudioInputDeviceOption] = []
        let defaultName = defaultInputDeviceName() ?? "Unknown Input"
        options.append(
            AudioInputDeviceOption(
                deviceID: nil,
                name: defaultName,
                detail: "Uses the current macOS default input device.",
                isSystemDefault: true
            )
        )

        for deviceID in sortedDevices {
            guard let name = audioObjectStringProperty(selector: kAudioObjectPropertyName, objectID: deviceID) else {
                continue
            }

            let manufacturer = audioObjectStringProperty(selector: kAudioObjectPropertyManufacturer, objectID: deviceID)
            let detail: String
            if deviceID == defaultDeviceID {
                if let manufacturer, !manufacturer.isEmpty {
                    detail = "Default input • \(manufacturer)"
                } else {
                    detail = "Default input"
                }
            } else if let manufacturer, !manufacturer.isEmpty {
                detail = manufacturer
            } else {
                detail = "Available input device"
            }

            options.append(
                AudioInputDeviceOption(
                    deviceID: deviceID,
                    name: name,
                    detail: detail,
                    isSystemDefault: false
                )
            )
        }
        return options
    }

    private nonisolated static func availableInputDeviceIDs() -> [AudioDeviceID] {
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDevices,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        var size: UInt32 = 0
        let sizeStatus = AudioObjectGetPropertyDataSize(
            AudioObjectID(kAudioObjectSystemObject),
            &address,
            0,
            nil,
            &size
        )
        guard sizeStatus == noErr, size > 0 else {
            return []
        }

        let count = Int(size) / MemoryLayout<AudioDeviceID>.size
        var deviceIDs = Array(repeating: AudioDeviceID(0), count: count)
        let dataStatus = AudioObjectGetPropertyData(
            AudioObjectID(kAudioObjectSystemObject),
            &address,
            0,
            nil,
            &size,
            &deviceIDs
        )
        guard dataStatus == noErr else {
            return []
        }

        return deviceIDs.filter { inputChannelCount(for: $0) > 0 }
    }

    private nonisolated static func inputChannelCount(for deviceID: AudioDeviceID) -> Int {
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioDevicePropertyStreamConfiguration,
            mScope: kAudioObjectPropertyScopeInput,
            mElement: kAudioObjectPropertyElementMain
        )
        var size: UInt32 = 0
        let sizeStatus = AudioObjectGetPropertyDataSize(deviceID, &address, 0, nil, &size)
        guard sizeStatus == noErr, size > 0 else {
            return 0
        }

        let rawPointer = UnsafeMutableRawPointer.allocate(
            byteCount: Int(size),
            alignment: MemoryLayout<AudioBufferList>.alignment
        )
        defer { rawPointer.deallocate() }

        let dataStatus = AudioObjectGetPropertyData(deviceID, &address, 0, nil, &size, rawPointer)
        guard dataStatus == noErr else {
            return 0
        }

        let bufferListPointer = rawPointer.assumingMemoryBound(to: AudioBufferList.self)
        let bufferList = UnsafeMutableAudioBufferListPointer(bufferListPointer)
        return bufferList.reduce(0) { $0 + Int($1.mNumberChannels) }
    }

    private nonisolated static func audioObjectStringProperty(
        selector: AudioObjectPropertySelector,
        objectID: AudioObjectID
    ) -> String? {
        var value: CFString = "" as CFString
        var address = AudioObjectPropertyAddress(
            mSelector: selector,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        var size = UInt32(MemoryLayout<CFString>.size)
        let status = AudioObjectGetPropertyData(objectID, &address, 0, nil, &size, &value)
        guard status == noErr else {
            return nil
        }
        let result = value as String
        return result.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? nil : result
    }

    private nonisolated static func inputLevelSummary(for buffer: AVAudioPCMBuffer) -> String? {
        guard let peak = peakLevel(for: buffer) else { return nil }

        let clamped = max(0.0, min(peak, 1.0))
        let percent = clamped * 100.0
        if percent < 10.0 {
            return String(format: "Peak %.1f%%", percent)
        }
        return "Peak \(Int(percent.rounded()))%"
    }

    private func lowSignalMessage(for buffers: [AVAudioPCMBuffer]) -> String? {
        let peak = buffers.compactMap(Self.peakLevel(for:)).max() ?? 0.0
        guard peak < 0.01 else {
            return nil
        }
        return String(
            format: "Mic signal too low on %@. Peak %.1f%%. Choose a different input device or raise the input level.",
            currentInputDeviceName,
            peak * 100.0
        )
    }

    private nonisolated static func peakLevel(for buffer: AVAudioPCMBuffer) -> Double? {
        switch buffer.format.commonFormat {
        case .pcmFormatFloat32:
            guard let channels = buffer.floatChannelData else { return nil }
            let frameCount = Int(buffer.frameLength)
            guard frameCount > 0 else { return 0.0 }
            let samples = UnsafeBufferPointer(start: channels[0], count: frameCount)
            return samples.reduce(0.0) { current, sample in
                max(current, abs(Double(sample)))
            }

        case .pcmFormatInt16:
            guard let channels = buffer.int16ChannelData else { return nil }
            let frameCount = Int(buffer.frameLength)
            guard frameCount > 0 else { return 0.0 }
            let samples = UnsafeBufferPointer(start: channels[0], count: frameCount)
            return samples.reduce(0.0) { current, sample in
                max(current, abs(Double(sample) / Double(Int16.max)))
            }

        default:
            return nil
        }
    }

    private func playStartSound() {
        if let sound = NSSound(named: NSSound.Name("Tink")) {
            sound.play()
        } else {
            NSSound.beep()
        }
    }

    private func playStopSound() {
        if let sound = NSSound(named: NSSound.Name("Pop")) {
            sound.play()
        } else {
            NSSound.beep()
        }
    }

    private func refreshLaunchAtLoginStatus() {
#if DEBUG
        launchAtLoginSupported = false
        launchAtLoginEnabled = false
        launchAtLoginStatusMessage = "Unavailable in debug builds"
#else
        guard Bundle.main.bundleURL.pathExtension == "app" else {
            launchAtLoginSupported = false
            launchAtLoginEnabled = false
            launchAtLoginStatusMessage = "Unavailable outside app bundle"
            return
        }

        launchAtLoginSupported = true

        let status = SMAppService.mainApp.status
        switch status {
        case .enabled:
            launchAtLoginEnabled = true
            launchAtLoginStatusMessage = "Enabled"
        case .requiresApproval:
            launchAtLoginEnabled = true
            launchAtLoginStatusMessage = "Requires approval in System Settings"
        case .notRegistered:
            launchAtLoginEnabled = false
            launchAtLoginStatusMessage = "Disabled"
        case .notFound:
            launchAtLoginEnabled = false
            launchAtLoginStatusMessage = "Not available"
        @unknown default:
            launchAtLoginEnabled = false
            launchAtLoginStatusMessage = "Unknown status"
        }
#endif
    }

    private func transition(to newState: DictationState) {
        state = newState

        switch newState {
        case .idle:
            statusLine = "Idle"
        case .listening:
            statusLine = "Listening..."
        case .recordingAudio:
            statusLine = "Recording audio..."
        case .finalizingTranscript:
            statusLine = "Finalizing transcript..."
        case .cleaning:
            if let progress = modelPreparationProgress, progress < 1 {
                statusLine = String(format: "Cleaning up... (%.0f%%)", progress * 100)
            } else {
                statusLine = "Cleaning up..."
            }
        case .inserting:
            statusLine = "Inserting text..."
        case .noSpeech:
            statusLine = "No speech detected"
        case .error(let message):
            statusLine = message
        }
    }

    private func setError(_ message: String) {
        transition(to: .error(message))
    }

    public var hasLatestCapturedAudio: Bool {
        latestCapturedAudioURL != nil
    }

    public func toggleLatestCapturedAudioPlayback() {
        if latestCapturedAudioIsPlaying {
            stopLatestCapturedAudioPlayback()
        } else {
            playLatestCapturedAudio()
        }
    }

    public func playLatestCapturedAudio() {
        guard let latestCapturedAudioURL else { return }

        do {
            audioPreviewPlayer?.stop()
            let player = try AVAudioPlayer(contentsOf: latestCapturedAudioURL)
            audioPreviewPlayer = player
            player.play()
            latestCapturedAudioIsPlaying = true

            let playbackDuration = player.duration
            Task { @MainActor [weak self] in
                guard playbackDuration > 0 else { return }
                try? await Task.sleep(nanoseconds: UInt64(playbackDuration * 1_000_000_000))
                guard let self, self.audioPreviewPlayer === player else { return }
                self.latestCapturedAudioIsPlaying = player.isPlaying
            }
        } catch {
            DebugLog.write("[AppState] failed to play latest captured audio: \(error.localizedDescription)")
            latestCapturedAudioIsPlaying = false
            audioPreviewPlayer = nil
        }
    }

    public func stopLatestCapturedAudioPlayback() {
        audioPreviewPlayer?.stop()
        latestCapturedAudioIsPlaying = false
    }

    public func revealLatestDebugCaptureInFinder() {
        guard let latestDebugCaptureDirectoryURL else { return }
        NSWorkspace.shared.activateFileViewerSelecting([latestDebugCaptureDirectoryURL])
    }

    private func copyToPasteboard(_ text: String) {
        let pasteboard = NSPasteboard.general
        pasteboard.clearContents()
        pasteboard.setString(text, forType: .string)
    }

    private struct DebugCaptureMetadata: Encodable {
        let createdAt: Date
        let mode: String
        let inputDeviceName: String
        let inputLevel: String
        let audioPath: String
        let startContext: DebugContextPayload?
        let contextualHints: [String]
        let result: DebugCaptureResultPayload
        let stopPath: StopPathDetails?
    }

    private struct DebugCaptureResultPayload: Encodable {
        let kind: String
        let rawText: String?
        let cleanedText: String?
        let error: String?
        let warning: String?
        let latency: PipelineLatency
    }

    private struct DebugContextPayload: Encodable {
        let appName: String
        let bundleIdentifier: String
        let surface: String
        let windowTitle: String
        let projectName: String?
        let projectPathHint: String?
        let activeDocumentHint: String?
        let browserTabHint: String?
        let browserURL: String?
        let browserHost: String?
        let browserPathHint: String?
    }

    private static func debugContextPayload(from context: DictationAppContext) -> DebugContextPayload {
        DebugContextPayload(
            appName: context.appName,
            bundleIdentifier: context.bundleIdentifier,
            surface: context.surface.rawValue,
            windowTitle: context.windowTitle,
            projectName: context.projectName,
            projectPathHint: context.projectPathHint,
            activeDocumentHint: context.activeDocumentHint,
            browserTabHint: context.browserTabHint,
            browserURL: context.browserURL,
            browserHost: context.browserHost,
            browserPathHint: context.browserPathHint
        )
    }

    private static func debugContextSummary(_ context: DictationAppContext) -> String {
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

public extension Notification.Name {
    static let localWisprShowControlPanel = Notification.Name("LocalWisprShowControlPanel")
}
