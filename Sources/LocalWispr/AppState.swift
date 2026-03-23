import AppKit
import ApplicationServices
import AVFoundation
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

    @Published public private(set) var hotkeyRegistrationStatus: HotkeyRegistrationStatus = .pending

    /// Global shortcut; persisted when not using an injected `HotkeyMonitoring` (tests).
    @Published public var globalHotkeyBinding: GlobalHotkeyBinding {
        didSet {
            guard hotkeyConfigManaged else { return }
            guard globalHotkeyBinding != oldValue else { return }
            Self.saveGlobalHotkeyBinding(globalHotkeyBinding)
            reconfigureHotkeyMonitor()
        }
    }

    private var hotkeyMonitor: HotkeyMonitoring
    private let hotkeyConfigManaged: Bool
    private let audioCapture: AudioCapturing
    private let transcriber: Transcriber
    private let pipeline: Pipeline
    private let transcriptHistoryStore: TranscriptHistoryStore

    private let clock = ContinuousClock()
    private var listeningStartedAt: ContinuousClock.Instant?
    private var liveTranscriptionSession: (any LiveTranscriptionSession)?
    private var activeSessionMode: TranscriberMode = .dictationLong
    private var stopLatencyHistory: [Int] = []

    private var hasBootstrapped = false
    private var isProcessing = false

    private static let globalHotkeyBindingDefaultsKey = "LocalWispr.globalHotkeyBinding"

    private var hasShownAccessibilityAlert = false

    public init(
        hotkeyMonitor: HotkeyMonitoring? = nil,
        audioCapture: AudioCapturing? = nil,
        transcriber: Transcriber? = nil,
        cleaner: (any Cleaning)? = nil,
        inserter: (any Inserting)? = nil,
        transcriptHistoryStore: TranscriptHistoryStore? = nil
    ) {
        let resolvedTranscriber = transcriber ?? Transcriber()

        if let hotkeyMonitor {
            self.hotkeyMonitor = hotkeyMonitor
            self.hotkeyConfigManaged = false
            self._globalHotkeyBinding = Published(initialValue: .rightCommandDoubleTap)
        } else {
            let binding = Self.loadGlobalHotkeyBinding()
            self.hotkeyMonitor = HotkeyFactory.makeMonitor(for: binding)
            self.hotkeyConfigManaged = true
            self._globalHotkeyBinding = Published(initialValue: binding)
        }

        self.audioCapture = audioCapture ?? AudioCapture()
        self.transcriber = resolvedTranscriber
        self.pipeline = Pipeline(
            cleaner: cleaner ?? TextCleaner(),
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

        hotkeyMonitor.onToggleRequested = { [weak self] in
            self?.toggleDictation()
        }

        refreshLaunchAtLoginStatus()

        accessibilityPermissionGranted = checkAccessibilityPermission(promptIfNeeded: true)
        if !accessibilityPermissionGranted {
            statusLine = "Accessibility permission required for global hotkey"
            presentAccessibilityAlert()
        }

        Task {
            await loadTranscriptHistory()
            await prewarmModels()
            await refreshAudioAndSpeechPermissions()
            reconfigureHotkeyMonitor()
        }
    }

    public var allPermissionsGranted: Bool {
        accessibilityPermissionGranted && microphonePermissionGranted && speechPermissionGranted
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
            await refreshAudioAndSpeechPermissions()
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
        Task {
            if state == .listening {
                await stopDictation()
            } else {
                await startDictation()
            }
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

    private func reconfigureHotkeyMonitor() {
        guard hotkeyConfigManaged else { return }

        hotkeyMonitor.onToggleRequested = nil
        hotkeyMonitor.stop()

        hotkeyMonitor = HotkeyFactory.makeMonitor(for: globalHotkeyBinding)
        hotkeyMonitor.onToggleRequested = { [weak self] in
            self?.toggleDictation()
        }

        guard globalHotkeyBinding != .none else {
            accessibilityPermissionGranted = checkAccessibilityPermission(promptIfNeeded: false)
            hotkeyRegistrationStatus = .inactive("Shortcut disabled in settings")
            return
        }

        accessibilityPermissionGranted = checkAccessibilityPermission(promptIfNeeded: false)

        if globalHotkeyBinding.requiresAccessibility && !accessibilityPermissionGranted {
            hotkeyRegistrationStatus = .failed("Accessibility access required for \(globalHotkeyBinding.menuTitle)")
            return
        }

        do {
            try hotkeyMonitor.start()
            hotkeyRegistrationStatus = .listening(globalHotkeyBinding.menuTitle)
        } catch {
            hotkeyRegistrationStatus = .failed(error.localizedDescription)
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

    private func refreshAudioAndSpeechPermissions() async {
        microphonePermissionGranted = await requestMicrophoneAccessIfNeeded()
        speechPermissionGranted = await requestSpeechAuthorizationIfNeeded()

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

    private func requestMicrophoneAccessIfNeeded() async -> Bool {
        await Self.requestMicrophoneAccess()
    }

    private func requestSpeechAuthorizationIfNeeded() async -> Bool {
        let status = await Self.requestSpeechAuthorization()
        return status == .authorized
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
        if !accessibilityPermissionGranted {
            setError("Accessibility permission is required. Click Grant Permissions.")
            presentAccessibilityAlert()
            return false
        }

        if !microphonePermissionGranted || !speechPermissionGranted {
            await refreshAudioAndSpeechPermissions()
        }

        if !microphonePermissionGranted || !speechPermissionGranted {
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
        guard !isProcessing else { return }
        guard await ensurePermissionsForDictation() else { return }

        let mode = transcriberMode

        do {
            let session = try await transcriber.startSession(
                mode: mode,
                locale: .current,
                audioFormat: audioCapture.outputAudioFormat
            )

            audioCapture.onBufferCaptured = { buffer in
                session.append(buffer)
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
            listeningStartedAt = clock.now

            modelPreparationProgress = nil
            playStartSound()
            transition(to: .listening)
        } catch {
            setError("Unable to start live transcription: \(error.localizedDescription)")
        }
    }

    private func stopDictation() async {
        guard !isProcessing else { return }

        playStopSound()

        let stopPressedAt = clock.now
        let recordingDurationMilliseconds = listeningStartedAt.map { durationToMilliseconds($0.duration(to: stopPressedAt)) }
        listeningStartedAt = nil

        let activeSession = liveTranscriptionSession
        liveTranscriptionSession = nil

        let modeUsedForSession = activeSessionMode

        let buffers: [AVAudioPCMBuffer]
        let drainStartedAt = clock.now
        do {
            buffers = try audioCapture.stopAndDrain()
        } catch {
            audioCapture.onBufferCaptured = nil
            if let activeSession {
                await activeSession.cancel()
            }
            setError("Unable to stop microphone capture: \(error.localizedDescription)")
            return
        }
        let drainMilliseconds = durationToMilliseconds(drainStartedAt.duration(to: clock.now))
        audioCapture.onBufferCaptured = nil

        isProcessing = true
        modelPreparationProgress = nil
        transition(to: .finalizingTranscript)

        let transcriptStartedAt = clock.now
        var rawText: String?
        var transcriptionError: Error?
        var liveFinalizationMilliseconds: Int?
        var batchFallbackMilliseconds: Int?
        var transcriptSource: TranscriptResolutionSource = .unavailable

        if let activeSession {
            let liveFinalizationStartedAt = clock.now
            do {
                rawText = try await runWithTimeout(seconds: 4) {
                    try await activeSession.finish()
                } onTimeout: {
                    DictationTimeoutError.liveFinalization
                }
                liveFinalizationMilliseconds = durationToMilliseconds(liveFinalizationStartedAt.duration(to: clock.now))
                transcriptSource = .liveFinalization
            } catch {
                liveFinalizationMilliseconds = durationToMilliseconds(liveFinalizationStartedAt.duration(to: clock.now))
                transcriptionError = error
                await activeSession.cancel()
            }
        }

        if rawText == nil && !buffers.isEmpty {
            let fallbackStartedAt = clock.now
            do {
                rawText = try await runWithTimeout(seconds: 10) {
                    try await self.transcriber.transcribe(buffers: buffers, mode: modeUsedForSession, locale: .current)
                } onTimeout: {
                    DictationTimeoutError.batchFallback
                }
                batchFallbackMilliseconds = durationToMilliseconds(fallbackStartedAt.duration(to: clock.now))
                transcriptSource = .batchFallback
            } catch {
                batchFallbackMilliseconds = durationToMilliseconds(fallbackStartedAt.duration(to: clock.now))
                transcriptionError = error
            }
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

        guard let rawText else {
            isProcessing = false
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

        isProcessing = false
        handlePipelineResult(result, modeUsedForSession: modeUsedForSession, stopPathDetails: stopPathDetails)
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

    private func copyToPasteboard(_ text: String) {
        let pasteboard = NSPasteboard.general
        pasteboard.clearContents()
        pasteboard.setString(text, forType: .string)
    }
}

public extension Notification.Name {
    static let localWisprShowControlPanel = Notification.Name("LocalWisprShowControlPanel")
}
