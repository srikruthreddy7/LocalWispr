import AppKit
import SwiftUI

public struct ControlPanelView: View {
    @ObservedObject var appState: AppState
    @Namespace private var glassNamespace

    public init(appState: AppState) {
        self.appState = appState
    }

    public var body: some View {
        GeometryReader { proxy in
            // Use the allocated height (window content), not the tallest child’s intrinsic height.
            // Short windows + tall dashboard content used to let the HStack grow with content while
            // Settings/History stayed short — sidebar height then tracked the wrong axis.
            let contentHeight = max(proxy.size.height - 48, 0)

            ZStack {
                background

                HStack(alignment: .top, spacing: 20) {
                    sidebar
                        .frame(width: 244, height: contentHeight, alignment: .top)

                    activeSectionView
                        .frame(maxWidth: .infinity, maxHeight: contentHeight, alignment: .topLeading)
                }
                .padding(24)
            }
            .frame(width: proxy.size.width, height: proxy.size.height, alignment: .topLeading)
        }
        // Must match ControlPanelWindowController window minSize (1260×800); 820 here broke layout
        // when the window was resized near the minimum (SwiftUI vs AppKit height mismatch).
        .frame(minWidth: 1260, minHeight: 800)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .animation(.snappy(duration: 0.38, extraBounce: 0.06), value: appState.state)
        .animation(.snappy(duration: 0.28, extraBounce: 0.02), value: appState.lastLatency)
        .animation(.snappy(duration: 0.28, extraBounce: 0.02), value: appState.hotkeyRegistrationStatus)
        .animation(.snappy(duration: 0.28, extraBounce: 0.02), value: appState.globalHotkeyBinding)
    }

    // MARK: - Background

    private var background: some View {
        ZStack {
            LinearGradient(
                colors: [
                    AppTheme.backgroundTop,
                    Color(red: 0.022, green: 0.024, blue: 0.031),
                    AppTheme.backgroundBottom
                ],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )

            GridBackdrop()
                .opacity(0.04)

            RadialGradient(
                colors: [
                    Color.white.opacity(0.08),
                    Color.white.opacity(0.015),
                    .clear
                ],
                center: .topLeading,
                startRadius: 10,
                endRadius: 520
            )
            .offset(x: -180, y: -260)

            RadialGradient(
                colors: [
                    Color.white.opacity(0.05),
                    .clear
                ],
                center: .bottomTrailing,
                startRadius: 20,
                endRadius: 420
            )
            .offset(x: 260, y: 200)
        }
        .ignoresSafeArea()
    }

    // MARK: - Sidebar

    private var sidebar: some View {
        LiquidCard(cornerRadius: 30, padding: 20, fillHeight: true) {
            VStack(alignment: .leading, spacing: 18) {
                VStack(alignment: .leading, spacing: 8) {
                    Text("LocalWispr")
                        .font(AppTheme.sans(28, weight: .semibold))
                        .foregroundStyle(.white)

                    Text("Choose a workspace inside the app.")
                        .font(AppTheme.sans(14, weight: .regular))
                        .foregroundStyle(AppTheme.secondaryText)
                        .lineSpacing(2)
                }

                VStack(spacing: 10) {
                    ForEach(ControlPanelSection.allCases) { section in
                        sidebarButton(for: section)
                    }
                }

                Divider()
                    .overlay(AppTheme.softStroke)

                VStack(alignment: .leading, spacing: 8) {
                    Text("CURRENT SHORTCUT")
                        .font(AppTheme.mono(10, weight: .medium))
                        .tracking(1.2)
                        .foregroundStyle(AppTheme.tertiaryText)

                    Text(appState.globalHotkeyBinding.menuTitle)
                        .font(AppTheme.sans(14, weight: .medium))
                        .foregroundStyle(.white)

                    Text(appState.hotkeyRegistrationStatus.detailLine)
                        .font(AppTheme.sans(12, weight: .regular))
                        .foregroundStyle(AppTheme.secondaryText)
                        .lineSpacing(2)
                }

                Spacer(minLength: 0)
            }
            .frame(maxHeight: .infinity, alignment: .topLeading)
        }
        .frame(maxHeight: .infinity, alignment: .top)
    }

    private func sidebarButton(for section: ControlPanelSection) -> some View {
        let isSelected = appState.controlPanelSection == section

        return Button {
            withAnimation(.snappy(duration: 0.25)) {
                appState.controlPanelSection = section
            }
            if section == .history, appState.selectedTranscriptRecordID == nil {
                appState.selectedTranscriptRecordID = appState.transcriptHistory.first?.id
            }
        } label: {
            HStack(spacing: 12) {
                Image(systemName: section.systemImageName)
                    .font(.system(size: 14, weight: .semibold))
                    .frame(width: 20)

                VStack(alignment: .leading, spacing: 3) {
                    Text(section.title)
                        .font(AppTheme.sans(15, weight: .medium))

                    Text(section.subtitle)
                        .font(AppTheme.sans(12, weight: .regular))
                        .foregroundStyle(isSelected ? Color.white.opacity(0.55) : AppTheme.secondaryText)
                }

                Spacer(minLength: 0)
            }
            .foregroundStyle(.white)
            .padding(.horizontal, 16)
            .padding(.vertical, 14)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(isSelected ? Color.white.opacity(0.14) : Color.clear)
            .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
            .contentShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
        }
        .buttonStyle(.plain)
    }

    // MARK: - Section Router

    @ViewBuilder
    private var activeSectionView: some View {
        switch appState.controlPanelSection {
        case .dashboard:
            ScrollView(.vertical, showsIndicators: false) {
                dashboardView
            }
        case .history:
            TranscriptHistoryView(appState: appState)
                .padding(4)
        case .settings:
            settingsView
        }
    }

    // MARK: - Dashboard (2×2 Grid)

    private let gridGap: CGFloat = 16
    private let cellHeight: CGFloat = 340

    private var dashboardView: some View {
        VStack(spacing: gridGap) {
            heroCard

            HStack(alignment: .top, spacing: gridGap) {
                commandDeck
                    .frame(maxWidth: .infinity, minHeight: cellHeight, maxHeight: cellHeight)

                performanceCard
                    .frame(maxWidth: .infinity, minHeight: cellHeight, maxHeight: cellHeight)
            }

            HStack(alignment: .top, spacing: gridGap) {
                transcriptDeck
                    .frame(maxWidth: .infinity, minHeight: cellHeight, maxHeight: cellHeight)

                permissionsCard
                    .frame(maxWidth: .infinity, minHeight: cellHeight, maxHeight: cellHeight)
            }
        }
        .padding(4)
    }

    // MARK: - Hero

    private var heroCard: some View {
        LiquidCard(cornerRadius: 32, padding: 24) {
            HStack(alignment: .top, spacing: 24) {
                VStack(alignment: .leading, spacing: 12) {
                    VStack(alignment: .leading, spacing: 10) {
                        Text("LocalWispr")
                            .font(AppTheme.sans(54, weight: .semibold))
                            .foregroundStyle(.white)

                        Text("Minimal local dictation with Apple speech analysis and Apple Intelligence cleanup.")
                            .font(AppTheme.sans(16, weight: .regular))
                            .foregroundStyle(AppTheme.secondaryText)
                            .lineSpacing(2)
                    }

                    if let alertLine = alertLine {
                        Text(alertLine)
                            .font(AppTheme.sans(14, weight: .medium))
                            .foregroundStyle(alertTint)
                    }
                }

                Spacer(minLength: 16)

                HStack(alignment: .top, spacing: 10) {
                    HeroMetricChip(
                        title: "LAST STOP",
                        value: lastStopSummary,
                        detail: lastStopCaption
                    )

                    if let stats = appState.latencyStats {
                        HeroMetricChip(
                            title: "AVG LATENCY",
                            value: "\(stats.averageMilliseconds)ms",
                            detail: averageLatencyCaption(stats)
                        )
                    }
                }
                .frame(width: 300)
            }
        }
    }

    // MARK: - Command Deck

    private var commandDeck: some View {
        LiquidCard(cornerRadius: 28, padding: 24) {
            VStack(alignment: .leading, spacing: 18) {
                DeckHeader(title: "Command Deck", subtitle: "Control capture and the latest reusable result.")

                commandButtons

                commandInfoStrip

                Spacer(minLength: 0)
            }
        }
    }

    private var commandButtons: some View {
        GlassEffectContainer(spacing: 12) {
            HStack(spacing: 12) {
                Button {
                    deferAction { appState.toggleDictation() }
                } label: {
                    HStack(spacing: 10) {
                        Circle()
                            .fill(appState.state == .listening ? AppTheme.danger : Color.black.opacity(0.78))
                            .frame(width: 9, height: 9)
                        Text(actionButtonTitle)
                            .font(AppTheme.sans(16, weight: .semibold))
                    }
                    .foregroundStyle(Color.black.opacity(0.84))
                    .padding(.horizontal, 20)
                    .padding(.vertical, 14)
                }
                .buttonStyle(.solidProminent)
                .disabled((!appState.allPermissionsGranted && appState.state != .listening) || appState.isBusy)

                Button {
                    deferAction { appState.copyLatestResult() }
                } label: {
                    Text("Copy Latest Result")
                        .font(AppTheme.sans(15, weight: .medium))
                        .foregroundStyle(.white)
                        .padding(.horizontal, 18)
                        .padding(.vertical, 14)
                }
                .buttonStyle(.subtleGlass)
                .disabled(!appState.hasLastRawTranscription && !appState.hasLastCleanedText)
            }
        }
    }

    private var commandInfoStrip: some View {
        GlassEffectContainer(spacing: 12) {
            HStack(spacing: 12) {
                InfoGlassTile(title: "STATE", value: stateLabel)
                    .glassEffectID("state-tile", in: glassNamespace)

                InfoGlassTile(title: "PERMISSIONS", value: appState.allPermissionsGranted ? "Ready" : "Needs Attention")
                    .glassEffectID("permission-tile", in: glassNamespace)

                InfoGlassTile(title: "HOTKEY", value: hotkeyTileValue)
                    .glassEffectID("hotkey-tile", in: glassNamespace)
            }
        }
    }

    // MARK: - Performance Card

    private var performanceCard: some View {
        LiquidCard(cornerRadius: 28, padding: 20) {
            VStack(alignment: .leading, spacing: 16) {
                Text("PERFORMANCE")
                    .font(AppTheme.mono(11, weight: .medium))
                    .tracking(1.6)
                    .foregroundStyle(AppTheme.tertiaryText)

                HStack(spacing: 10) {
                    MetricCell(title: "TRANSCRIBE", value: metricText(appState.lastLatency?.stopToTranscriptMilliseconds))
                    MetricCell(title: "CLEANUP", value: metricText(appState.lastLatency?.cleanupMilliseconds))
                }

                HStack(spacing: 10) {
                    MetricCell(title: "INSERT", value: metricText(appState.lastLatency?.insertionMilliseconds))
                    MetricCell(title: "P50 / P95", value: percentileText)
                }

                if appState.lastLatency == nil {
                    Text("No completed runs yet.")
                        .font(AppTheme.sans(14, weight: .regular))
                        .foregroundStyle(AppTheme.tertiaryText)
                }

                Spacer(minLength: 0)
            }
        }
    }

    // MARK: - Transcript Deck (Raw + Cleaned side by side)

    private var transcriptDeck: some View {
        LiquidCard(cornerRadius: 28, padding: 20) {
            VStack(alignment: .leading, spacing: 14) {
                DeckHeader(title: "Latest Result", subtitle: "Always retrievable, even when insertion misses the focused field.")

                HStack(spacing: 14) {
                    TranscriptMiniPane(
                        title: "Raw transcription",
                        text: appState.lastRawTranscription,
                        placeholder: "The raw transcript will appear here after each dictation session.",
                        buttonTitle: "Copy Raw",
                        action: { deferAction { appState.copyLastTranscription() } },
                        enabled: appState.hasLastRawTranscription
                    )

                    TranscriptMiniPane(
                        title: "Cleaned output",
                        text: appState.lastCleanedText,
                        placeholder: "The cleaned version will appear here once the language model finishes.",
                        buttonTitle: "Copy Cleaned",
                        action: { deferAction { appState.copyLastCleanedText() } },
                        enabled: appState.hasLastCleanedText
                    )
                }
            }
        }
    }

    // MARK: - Permissions Card

    private var permissionsCard: some View {
        LiquidCard(cornerRadius: 28, padding: 20) {
            VStack(alignment: .leading, spacing: 16) {
                DeckHeader(title: "Permissions", subtitle: "System capabilities required for global capture and voice input.")

                PermissionRow(title: "Accessibility", granted: appState.accessibilityPermissionGranted)
                PermissionRow(title: "Microphone", granted: appState.microphonePermissionGranted)
                PermissionRow(title: "Speech", granted: appState.speechPermissionGranted)

                Button {
                    deferAction { appState.requestAllPermissions() }
                } label: {
                    Text("Grant All Permissions")
                        .font(AppTheme.sans(14, weight: .medium))
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 12)
                }
                .buttonStyle(.solidProminent)

                Spacer(minLength: 0)
            }
        }
    }

    // MARK: - Settings View

    private var settingsView: some View {
        ScrollView(.vertical, showsIndicators: false) {
            VStack(spacing: 20) {
                LiquidCard(cornerRadius: 28, padding: 24) {
                    VStack(alignment: .leading, spacing: 22) {
                        DeckHeader(title: "Transcriber", subtitle: "Choose the speech recognition mode.")

                        modeSelector
                    }
                }

                LiquidCard(cornerRadius: 28, padding: 24) {
                    VStack(alignment: .leading, spacing: 22) {
                        DeckHeader(title: "Global Shortcut", subtitle: "Select and configure the system-wide hotkey for dictation.")

                        globalShortcutSection
                    }
                }

                LiquidCard(cornerRadius: 28, padding: 24) {
                    VStack(alignment: .leading, spacing: 22) {
                        DeckHeader(title: "Application", subtitle: "Basic app lifecycle controls.")

                        Toggle(
                            "Launch at Login",
                            isOn: Binding(
                                get: { appState.launchAtLoginEnabled },
                                set: { newValue in
                                    deferAction { appState.setLaunchAtLogin(newValue) }
                                }
                            )
                        )
                        .toggleStyle(.switch)
                        .disabled(!appState.launchAtLoginSupported)
                        .font(AppTheme.sans(15, weight: .medium))
                        .foregroundStyle(.white)

                        if !appState.launchAtLoginStatusMessage.isEmpty {
                            Text(appState.launchAtLoginStatusMessage)
                                .font(AppTheme.sans(13, weight: .regular))
                                .foregroundStyle(AppTheme.tertiaryText)
                        }

                        Divider()
                            .overlay(AppTheme.softStroke)

                        GlassEffectContainer(spacing: 10) {
                            HStack(spacing: 10) {
                                Button {
                                    deferAction { appState.requestAllPermissions() }
                                } label: {
                                    Text("Refresh Permissions")
                                        .font(AppTheme.sans(14, weight: .medium))
                                        .padding(.horizontal, 16)
                                        .padding(.vertical, 12)
                                }
                                .buttonStyle(.subtleGlass)

                                Button {
                                    deferAction { appState.quit() }
                                } label: {
                                    Text("Quit")
                                        .font(AppTheme.sans(14, weight: .medium))
                                        .padding(.horizontal, 16)
                                        .padding(.vertical, 12)
                                }
                                .buttonStyle(.subtleGlass)
                            }
                        }
                    }
                }
            }
            .padding(4)
        }
    }

    private var modeSelector: some View {
        GlassEffectContainer(spacing: 10) {
            HStack(spacing: 10) {
                ForEach(TranscriberMode.allCases) { mode in
                    modeButton(for: mode)
                }
            }
        }
    }

    private var globalShortcutSection: some View {
        GlassEffectContainer(spacing: 12) {
            VStack(alignment: .leading, spacing: 10) {
                Menu {
                    ForEach(GlobalHotkeyBinding.allCases) { binding in
                        Button {
                            deferAction { appState.globalHotkeyBinding = binding }
                        } label: {
                            if binding == appState.globalHotkeyBinding {
                                Label(binding.menuTitle, systemImage: "checkmark")
                            } else {
                                Text(binding.menuTitle)
                            }
                        }
                    }
                } label: {
                    HStack(spacing: 12) {
                        VStack(alignment: .leading, spacing: 4) {
                            Text("Shortcut")
                                .font(AppTheme.mono(10, weight: .medium))
                                .tracking(1.2)
                                .foregroundStyle(AppTheme.tertiaryText)

                            Text(appState.globalHotkeyBinding.menuTitle)
                                .font(AppTheme.sans(15, weight: .medium))
                                .foregroundStyle(.white)
                        }

                        Spacer()

                        Image(systemName: "chevron.up.chevron.down")
                            .font(.system(size: 12, weight: .semibold))
                            .foregroundStyle(AppTheme.secondaryText)
                    }
                    .padding(.horizontal, 16)
                    .padding(.vertical, 14)
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
                .buttonStyle(.subtleGlass)
                .accessibilityLabel("Global shortcut")

                Text(appState.hotkeyRegistrationStatus.detailLine)
                    .font(AppTheme.sans(13, weight: .regular))
                    .foregroundStyle(AppTheme.secondaryText)
                    .lineSpacing(2)
            }
        }
    }

    @ViewBuilder
    private func modeButton(for mode: TranscriberMode) -> some View {
        if mode == appState.transcriberMode {
            Button {
                deferAction { appState.transcriberMode = mode }
            } label: {
                Text(mode.title)
                    .font(AppTheme.sans(15, weight: .medium))
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 12)
            }
            .buttonStyle(.solidProminent)
        } else {
            Button {
                deferAction { appState.transcriberMode = mode }
            } label: {
                Text(mode.title)
                    .font(AppTheme.sans(15, weight: .medium))
                    .foregroundStyle(Color.white.opacity(0.86))
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 12)
            }
            .buttonStyle(.subtleGlass)
        }
    }

    // MARK: - Computed Properties

    private var hotkeyTileValue: String {
        switch appState.hotkeyRegistrationStatus {
        case .listening:
            return "Registered"
        case .failed:
            return "Not Registered"
        case .inactive:
            return "Off"
        case .pending:
            return "Checking…"
        }
    }

    private var actionButtonTitle: String {
        appState.state == .listening ? "Stop Dictation" : "Start Dictation"
    }

    private var stateLabel: String {
        switch appState.state {
        case .idle:
            return "Idle"
        case .listening:
            return "Listening"
        case .finalizingTranscript:
            return "Finalizing"
        case .cleaning:
            return "Cleaning"
        case .inserting:
            return "Inserting"
        case .noSpeech:
            return "No Speech"
        case .error:
            return "Needs Attention"
        }
    }

    private var alertLine: String? {
        switch appState.state {
        case .error(let message):
            return message
        case .noSpeech:
            return "Nothing meaningful was detected in the last capture."
        default:
            return nil
        }
    }

    private var alertTint: Color {
        switch appState.state {
        case .error:
            return AppTheme.warning
        case .noSpeech:
            return AppTheme.secondaryText
        default:
            return AppTheme.secondaryText
        }
    }

    private var lastStopSummary: String {
        if let latency = appState.lastLatency {
            return "\(latency.totalStopToInsertMilliseconds)ms"
        }
        return "—"
    }

    private var lastStopCaption: String {
        guard appState.lastLatency != nil else {
            return "No completed run yet"
        }
        return "Most recent session"
    }

    private func averageLatencyCaption(_ stats: StopLatencyStats) -> String {
        let n = stats.sampleCount
        let noun = n == 1 ? "session" : "sessions"
        return "Average of last \(n) completed \(noun)"
    }

    private func metricText(_ value: Int?) -> String {
        guard let value else { return "—" }
        return "\(value)ms"
    }

    private var percentileText: String {
        guard let stats = appState.latencyStats else { return "—" }
        return "\(stats.p50Milliseconds) / \(stats.p95Milliseconds)"
    }

    private func deferAction(_ action: @escaping @MainActor () -> Void) {
        DispatchQueue.main.async {
            action()
        }
    }
}

// MARK: - Reusable Private Views

private struct LiquidCard<Content: View>: View {
    let cornerRadius: CGFloat
    let padding: CGFloat
    let fillHeight: Bool
    @ViewBuilder let content: Content

    init(
        cornerRadius: CGFloat,
        padding: CGFloat,
        fillHeight: Bool = false,
        @ViewBuilder content: () -> Content
    ) {
        self.cornerRadius = cornerRadius
        self.padding = padding
        self.fillHeight = fillHeight
        self.content = content()
    }

    var body: some View {
        let frameAlignment: Alignment = fillHeight ? .topLeading : .leading

        content
            .padding(padding)
            .frame(maxWidth: .infinity, maxHeight: fillHeight ? .infinity : nil, alignment: frameAlignment)
            .background(AppTheme.panelFill)
            .glassEffect(.regular, in: RoundedRectangle(cornerRadius: cornerRadius, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                    .strokeBorder(
                        LinearGradient(
                            colors: [
                                Color.white.opacity(0.18),
                                Color.white.opacity(0.03)
                            ],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        ),
                        lineWidth: 1
                    )
            )
            .shadow(color: Color.black.opacity(0.28), radius: 28, y: 18)
    }
}

private struct GridBackdrop: View {
    var body: some View {
        GeometryReader { proxy in
            Path { path in
                let width = proxy.size.width
                let height = proxy.size.height
                let spacing: CGFloat = 56

                var x: CGFloat = 0
                while x <= width {
                    path.move(to: CGPoint(x: x, y: 0))
                    path.addLine(to: CGPoint(x: x, y: height))
                    x += spacing
                }

                var y: CGFloat = 0
                while y <= height {
                    path.move(to: CGPoint(x: 0, y: y))
                    path.addLine(to: CGPoint(x: width, y: y))
                    y += spacing
                }
            }
            .stroke(Color.white.opacity(0.12), lineWidth: 0.55)
        }
    }
}

private struct DeckHeader: View {
    let title: String
    let subtitle: String

    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            Text(title.uppercased())
                .font(AppTheme.mono(11, weight: .medium))
                .tracking(1.6)
                .foregroundStyle(AppTheme.tertiaryText)

            Text(subtitle)
                .font(AppTheme.sans(14, weight: .regular))
                .foregroundStyle(AppTheme.secondaryText)
                .lineSpacing(2)
        }
    }
}

private struct HeroMetricChip: View {
    let title: String
    let value: String
    let detail: String

    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            Text(title)
                .font(AppTheme.mono(9, weight: .medium))
                .tracking(1.0)
                .foregroundStyle(AppTheme.tertiaryText)

            Text(value)
                .font(AppTheme.sans(20, weight: .semibold))
                .foregroundStyle(.white)
                .minimumScaleFactor(0.85)
                .lineLimit(1)

            Text(detail)
                .font(AppTheme.sans(11, weight: .regular))
                .foregroundStyle(AppTheme.secondaryText)
                .lineSpacing(2)
                .lineLimit(2)
                .fixedSize(horizontal: false, vertical: true)
        }
        .frame(maxWidth: .infinity, minHeight: 82, alignment: .topLeading)
        .padding(12)
        .background(Color.white.opacity(0.03))
        .glassEffect(.regular, in: RoundedRectangle(cornerRadius: 16, style: .continuous))
    }
}

private struct InfoGlassTile: View {
    let title: String
    let value: String

    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            Text(title)
                .font(AppTheme.mono(10, weight: .medium))
                .tracking(1.2)
                .foregroundStyle(AppTheme.tertiaryText)

            Text(value)
                .font(AppTheme.sans(15, weight: .medium))
                .foregroundStyle(.white)
                .lineLimit(1)
                .minimumScaleFactor(0.7)
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.white.opacity(0.025))
        .glassEffect(.regular, in: RoundedRectangle(cornerRadius: 18, style: .continuous))
    }
}

private struct MetricCell: View {
    let title: String
    let value: String

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title)
                .font(AppTheme.mono(10, weight: .medium))
                .tracking(1.2)
                .foregroundStyle(AppTheme.tertiaryText)

            Text(value)
                .font(AppTheme.sans(20, weight: .semibold))
                .foregroundStyle(.white)
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.white.opacity(0.028))
        .glassEffect(.regular, in: RoundedRectangle(cornerRadius: 20, style: .continuous))
    }
}

private struct PermissionRow: View {
    let title: String
    let granted: Bool

    var body: some View {
        HStack(spacing: 10) {
            Circle()
                .fill(granted ? AppTheme.success : AppTheme.warning)
                .frame(width: 9, height: 9)

            Text(title)
                .font(AppTheme.sans(15, weight: .medium))
                .foregroundStyle(.white)

            Spacer()

            Text(granted ? "Granted" : "Missing")
                .font(AppTheme.mono(11, weight: .medium))
                .foregroundStyle(AppTheme.secondaryText)
        }
    }
}

/// Compact transcript pane that fits inside a grid cell; scrolls only when text overflows.
private struct TranscriptMiniPane: View {
    let title: String
    let text: String
    let placeholder: String
    let buttonTitle: String
    let action: () -> Void
    let enabled: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text(title)
                    .font(AppTheme.sans(15, weight: .semibold))
                    .foregroundStyle(.white)
                    .lineLimit(1)

                Spacer(minLength: 4)

                Button(action: action) {
                    Text(buttonTitle)
                        .font(AppTheme.sans(12, weight: .medium))
                        .padding(.horizontal, 12)
                        .padding(.vertical, 8)
                }
                .buttonStyle(.subtleGlass)
                .disabled(!enabled)
            }

            ScrollView(.vertical, showsIndicators: false) {
                Text(displayText)
                    .font(AppTheme.sans(14, weight: .regular))
                    .foregroundStyle(displayColor)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .multilineTextAlignment(.leading)
                    .textSelection(.enabled)
                    .lineSpacing(2)
            }
            .frame(maxHeight: .infinity)
        }
        .padding(14)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .background(Color.white.opacity(0.025))
        .clipShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
    }

    private var displayText: String {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? placeholder : trimmed
    }

    private var displayColor: Color {
        text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? AppTheme.tertiaryText : .white
    }
}

// MARK: - Custom Button Styles (immune to inactive-window dimming)

/// Solid bright button — replaces `.glassProminent` so it never dims when the window loses focus.
private struct SolidProminentButtonStyle: ButtonStyle {
    @Environment(\.isEnabled) private var isEnabled

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .background(
                RoundedRectangle(cornerRadius: 14, style: .continuous)
                    .fill(Color.white.opacity(isEnabled ? (configuration.isPressed ? 0.72 : 0.88) : 0.28))
            )
            .foregroundStyle(Color.black.opacity(isEnabled ? 0.84 : 0.4))
            .opacity(isEnabled ? 1 : 0.6)
            .contentShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
    }
}

/// Subtle translucent button — replaces `.glass` so it never dims when the window loses focus.
private struct SubtleGlassButtonStyle: ButtonStyle {
    @Environment(\.isEnabled) private var isEnabled

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .background(
                RoundedRectangle(cornerRadius: 14, style: .continuous)
                    .fill(Color.white.opacity(configuration.isPressed ? 0.14 : 0.07))
            )
            .overlay(
                RoundedRectangle(cornerRadius: 14, style: .continuous)
                    .strokeBorder(Color.white.opacity(0.10), lineWidth: 1)
            )
            .foregroundStyle(Color.white.opacity(isEnabled ? 0.86 : 0.35))
            .opacity(isEnabled ? 1 : 0.55)
            .contentShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
    }
}

extension ButtonStyle where Self == SolidProminentButtonStyle {
    static var solidProminent: SolidProminentButtonStyle { SolidProminentButtonStyle() }
}

extension ButtonStyle where Self == SubtleGlassButtonStyle {
    static var subtleGlass: SubtleGlassButtonStyle { SubtleGlassButtonStyle() }
}
