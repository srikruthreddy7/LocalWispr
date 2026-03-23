import SwiftUI

public struct MenuBarView: View {
    @ObservedObject var appState: AppState

    public init(appState: AppState) {
        self.appState = appState
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .center, spacing: 10) {
                MenuBarStatusIconView(state: appState.state)
                VStack(alignment: .leading, spacing: 2) {
                    Text("LocalWispr")
                        .font(.headline)
                    Text(appState.statusLine)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(2)

                    Text(appState.hotkeyRegistrationStatus.menuBarLine)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .lineLimit(2)
                }
            }

            if let latency = appState.lastLatency {
                Text(latency.formattedSummary)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }

            if !appState.allPermissionsGranted {
                permissionSummary
            }

            Divider()

            Button(appState.state == .listening ? "Stop Dictation" : "Start Dictation") {
                DispatchQueue.main.async {
                    appState.toggleDictation()
                }
            }
            .disabled((!appState.allPermissionsGranted && appState.state != .listening) || appState.isBusy)

            Button("Open Control Panel") {
                DispatchQueue.main.async {
                    appState.showControlPanel()
                }
            }

            if appState.hasLastRawTranscription || appState.hasLastCleanedText {
                Button("Copy Latest Result") {
                    DispatchQueue.main.async {
                        appState.copyLatestResult()
                    }
                }
            }

            Picker("Mode", selection: $appState.transcriberMode) {
                ForEach(TranscriberMode.allCases) { mode in
                    Text(mode.title).tag(mode)
                }
            }

            Menu("Shortcut: \(appState.globalHotkeyBinding.menuTitle)") {
                ForEach(GlobalHotkeyBinding.allCases) { binding in
                    Button {
                        DispatchQueue.main.async {
                            appState.globalHotkeyBinding = binding
                        }
                    } label: {
                        if binding == appState.globalHotkeyBinding {
                            Label(binding.menuTitle, systemImage: "checkmark")
                        } else {
                            Text(binding.menuTitle)
                        }
                    }
                }
            }

            Divider()

            Button("Quit") {
                appState.quit()
            }
        }
        .frame(width: 280)
        .padding(12)
    }

    @ViewBuilder
    private var permissionSummary: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Permissions Required")
                .font(.caption)
                .foregroundStyle(.secondary)

            Text(permissionLine(title: "Accessibility", granted: appState.accessibilityPermissionGranted))
                .font(.caption2)
            Text(permissionLine(title: "Microphone", granted: appState.microphonePermissionGranted))
                .font(.caption2)
            Text(permissionLine(title: "Speech", granted: appState.speechPermissionGranted))
                .font(.caption2)

            Button("Grant Permissions") {
                DispatchQueue.main.async {
                    appState.requestAllPermissions()
                }
            }
        }
        .padding(10)
        .background(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .fill(Color.white.opacity(0.06))
        )
    }

    private func permissionLine(title: String, granted: Bool) -> String {
        "\(title): \(granted ? "Granted" : "Missing")"
    }
}

public struct MenuBarStatusIconView: View {
    let state: DictationState

    @State private var pulsing = false

    public init(state: DictationState) {
        self.state = state
    }

    public var body: some View {
        Group {
            switch state {
            case .idle, .noSpeech, .error:
                Image(systemName: "mic")
            case .listening:
                Circle()
                    .fill(Color.red)
                    .frame(width: 10, height: 10)
                    .opacity(pulsing ? 0.25 : 1.0)
                    .onAppear {
                        withAnimation(.easeInOut(duration: 0.8).repeatForever(autoreverses: true)) {
                            pulsing = true
                        }
                    }
                    .onDisappear {
                        pulsing = false
                    }
            case .finalizingTranscript, .cleaning, .inserting:
                ProgressView()
                    .controlSize(.small)
                    .frame(width: 14, height: 14)
            }
        }
        .frame(width: 16, height: 16)
    }
}
