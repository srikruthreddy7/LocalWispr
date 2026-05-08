import SwiftUI

public struct BenchmarkRecorderView: View {
    @ObservedObject var appState: AppState

    @State private var category: BenchmarkClipCategory = .dictation
    @State private var promptText = "Say the sentence exactly, then stop recording."
    @State private var referenceText = ""

    public init(appState: AppState) {
        self.appState = appState
    }

    public var body: some View {
        ScrollView(.vertical, showsIndicators: false) {
            VStack(alignment: .leading, spacing: 18) {
                header

                HStack(alignment: .top, spacing: 16) {
                    recorderPanel
                        .frame(maxWidth: .infinity, minHeight: 420, alignment: .topLeading)

                    corpusPanel
                        .frame(maxWidth: .infinity, minHeight: 420, alignment: .topLeading)
                }

                checklistPanel
            }
        }
        .onAppear {
            appState.reloadBenchmarkCorpus()
        }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("API Benchmark Corpus")
                .font(AppTheme.sans(42, weight: .semibold))
                .foregroundStyle(.white)

            Text("Record once, save exact references, replay the same clips through every REST and WebSocket provider.")
                .font(AppTheme.sans(16, weight: .regular))
                .foregroundStyle(AppTheme.secondaryText)
                .lineSpacing(2)
        }
        .padding(.bottom, 2)
    }

    private var recorderPanel: some View {
        BenchmarkPanel {
            VStack(alignment: .leading, spacing: 18) {
                BenchmarkSectionTitle(
                    title: "Recorder",
                    subtitle: "Create clips that can be replayed through Groq, Fireworks, Together, Cerebras, or any local server."
                )

                categoryPicker

                fieldBlock(title: "Prompt", helper: "Optional. Use this to tell yourself what to say.") {
                    TextEditor(text: $promptText)
                        .benchmarkTextEditor(minHeight: 68)
                }

                fieldBlock(title: "Reference", helper: "Required. Type the exact words spoken in the recording.") {
                    TextEditor(text: $referenceText)
                        .benchmarkTextEditor(minHeight: 118)
                }

                recordingControls

                Text(appState.benchmarkCorpusStatusLine)
                    .font(AppTheme.sans(13, weight: .regular))
                    .foregroundStyle(AppTheme.secondaryText)
                    .lineSpacing(2)
            }
        }
    }

    private var categoryPicker: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("CATEGORY")
                .font(AppTheme.mono(10, weight: .medium))
                .tracking(1.2)
                .foregroundStyle(AppTheme.tertiaryText)

            Picker("Category", selection: $category) {
                ForEach(BenchmarkClipCategory.allCases) { option in
                    Text(option.title).tag(option)
                }
            }
            .pickerStyle(.segmented)
        }
    }

    private var recordingControls: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 10) {
                Button {
                    appState.toggleAudioPreviewCapture()
                } label: {
                    Label(appState.audioPreviewRecordingActive ? "Stop Recording" : "Record Clip", systemImage: appState.audioPreviewRecordingActive ? "stop.fill" : "record.circle")
                        .font(AppTheme.sans(14, weight: .semibold))
                        .padding(.horizontal, 16)
                        .padding(.vertical, 12)
                }
                .buttonStyle(BenchmarkPrimaryButtonStyle(isRecording: appState.audioPreviewRecordingActive))
                .disabled(appState.isBusy || appState.state == .listening)

                Button {
                    appState.toggleLatestCapturedAudioPlayback()
                } label: {
                    Text(appState.latestCapturedAudioIsPlaying ? "Stop Audio" : "Play Last")
                        .font(AppTheme.sans(14, weight: .medium))
                        .padding(.horizontal, 16)
                        .padding(.vertical, 12)
                }
                .buttonStyle(BenchmarkSecondaryButtonStyle())
                .disabled(!appState.hasLatestCapturedAudio)

                Button {
                    appState.recoverLatestDebugAudioCapture()
                } label: {
                    Text("Recover Latest")
                        .font(AppTheme.sans(14, weight: .medium))
                        .padding(.horizontal, 16)
                        .padding(.vertical, 12)
                }
                .buttonStyle(BenchmarkSecondaryButtonStyle())

                Button {
                    appState.saveLatestCaptureToBenchmarkCorpus(
                        category: category,
                        promptText: promptText,
                        referenceText: referenceText
                    )
                } label: {
                    Text("Save to Corpus")
                        .font(AppTheme.sans(14, weight: .semibold))
                        .padding(.horizontal, 16)
                        .padding(.vertical, 12)
                }
                .buttonStyle(BenchmarkSecondaryButtonStyle())
                .disabled(!canAttemptSave)
            }

            if let latestCapturedAudioURL = appState.latestCapturedAudioURL {
                Text(latestCapturedAudioURL.path)
                    .font(AppTheme.mono(11, weight: .regular))
                    .foregroundStyle(AppTheme.tertiaryText)
                    .lineLimit(2)
                    .textSelection(.enabled)
            } else {
                Text("No clip recorded in this session yet.")
                    .font(AppTheme.sans(13, weight: .regular))
                    .foregroundStyle(AppTheme.tertiaryText)
            }
        }
    }

    private var corpusPanel: some View {
        BenchmarkPanel {
            VStack(alignment: .leading, spacing: 18) {
                BenchmarkSectionTitle(
                    title: "Saved Corpus",
                    subtitle: "This folder is the source of truth for provider benchmarks."
                )

                HStack(spacing: 10) {
                    Button {
                        appState.revealBenchmarkCorpusInFinder()
                    } label: {
                        Text("Open Corpus Folder")
                            .font(AppTheme.sans(14, weight: .medium))
                            .padding(.horizontal, 16)
                            .padding(.vertical, 12)
                    }
                    .buttonStyle(BenchmarkSecondaryButtonStyle())

                    Button {
                        appState.copyBenchmarkCorpusPath()
                    } label: {
                        Text("Copy Path")
                            .font(AppTheme.sans(14, weight: .medium))
                            .padding(.horizontal, 16)
                            .padding(.vertical, 12)
                    }
                    .buttonStyle(BenchmarkSecondaryButtonStyle())
                }

                Text(appState.benchmarkCorpusDirectoryURL.path)
                    .font(AppTheme.mono(11, weight: .regular))
                    .foregroundStyle(AppTheme.tertiaryText)
                    .textSelection(.enabled)
                    .lineLimit(2)

                Divider()
                    .overlay(AppTheme.softStroke)

                recentClips
            }
        }
    }

    private var recentClips: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("RECENT CLIPS")
                .font(AppTheme.mono(10, weight: .medium))
                .tracking(1.2)
                .foregroundStyle(AppTheme.tertiaryText)

            if appState.benchmarkCorpusEntries.isEmpty {
                Text("Save a few short clips first. We need a stable corpus before ranking API providers.")
                    .font(AppTheme.sans(14, weight: .regular))
                    .foregroundStyle(AppTheme.secondaryText)
                    .lineSpacing(2)
            } else {
                VStack(spacing: 8) {
                    ForEach(appState.benchmarkCorpusEntries.prefix(8)) { entry in
                        BenchmarkClipRow(entry: entry)
                    }
                }
            }
        }
    }

    private var checklistPanel: some View {
        BenchmarkPanel {
            VStack(alignment: .leading, spacing: 16) {
                BenchmarkSectionTitle(
                    title: "Corpus Targets",
                    subtitle: "These targets match the API benchmark plan. Record the hard Indian-accent cases first."
                )

                LazyVGrid(columns: [GridItem(.adaptive(minimum: 220), spacing: 12)], spacing: 12) {
                    ForEach(BenchmarkClipCategory.allCases) { option in
                        let count = appState.benchmarkCorpusEntries.filter { $0.category == option }.count
                        BenchmarkTargetTile(category: option, count: count)
                    }
                }
            }
        }
    }

    private var canAttemptSave: Bool {
        appState.hasLatestCapturedAudio
            && !appState.audioPreviewRecordingActive
    }

    @ViewBuilder
    private func fieldBlock<Content: View>(
        title: String,
        helper: String,
        @ViewBuilder content: () -> Content
    ) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .firstTextBaseline) {
                Text(title.uppercased())
                    .font(AppTheme.mono(10, weight: .medium))
                    .tracking(1.2)
                    .foregroundStyle(AppTheme.tertiaryText)

                Spacer()

                Text(helper)
                    .font(AppTheme.sans(12, weight: .regular))
                    .foregroundStyle(AppTheme.tertiaryText)
            }

            content()
        }
    }
}

private struct BenchmarkPanel<Content: View>: View {
    @ViewBuilder let content: Content

    var body: some View {
        content
            .padding(22)
            .frame(maxWidth: .infinity, alignment: .topLeading)
            .background(AppTheme.panelFill)
            .glassEffect(.regular, in: RoundedRectangle(cornerRadius: 24, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 24, style: .continuous)
                    .strokeBorder(AppTheme.panelStroke, lineWidth: 1)
            )
    }
}

private struct BenchmarkSectionTitle: View {
    let title: String
    let subtitle: String

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
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

private struct BenchmarkClipRow: View {
    let entry: BenchmarkClipManifestEntry

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            VStack(alignment: .leading, spacing: 4) {
                Text(entry.category.title)
                    .font(AppTheme.sans(13, weight: .semibold))
                    .foregroundStyle(.white)

                Text(entry.referenceText)
                    .font(AppTheme.sans(12, weight: .regular))
                    .foregroundStyle(AppTheme.secondaryText)
                    .lineLimit(2)
            }

            Spacer(minLength: 8)

            Text(durationText)
                .font(AppTheme.mono(11, weight: .medium))
                .foregroundStyle(AppTheme.tertiaryText)
        }
        .padding(12)
        .background(Color.white.opacity(0.035))
        .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
    }

    private var durationText: String {
        guard let durationMilliseconds = entry.durationMilliseconds else {
            return "—"
        }
        return String(format: "%.1fs", Double(durationMilliseconds) / 1_000.0)
    }
}

private struct BenchmarkTargetTile: View {
    let category: BenchmarkClipCategory
    let count: Int

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text(category.title)
                    .font(AppTheme.sans(14, weight: .semibold))
                    .foregroundStyle(.white)
                    .lineLimit(1)
                    .minimumScaleFactor(0.8)

                Spacer()

                Text("\(count)/\(category.targetCount)")
                    .font(AppTheme.mono(12, weight: .medium))
                    .foregroundStyle(AppTheme.secondaryText)
            }

            ProgressView(value: min(Double(count), Double(category.targetCount)), total: Double(category.targetCount))
                .tint(.white.opacity(0.82))
        }
        .padding(14)
        .background(Color.white.opacity(0.035))
        .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
    }
}

private struct BenchmarkPrimaryButtonStyle: ButtonStyle {
    let isRecording: Bool
    @Environment(\.isEnabled) private var isEnabled

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .foregroundStyle(Color.black.opacity(isEnabled ? 0.86 : 0.38))
            .background(
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .fill(isRecording ? AppTheme.danger.opacity(0.9) : Color.white.opacity(configuration.isPressed ? 0.72 : 0.9))
            )
            .opacity(isEnabled ? 1 : 0.56)
            .contentShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
    }
}

private struct BenchmarkSecondaryButtonStyle: ButtonStyle {
    @Environment(\.isEnabled) private var isEnabled

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .foregroundStyle(Color.white.opacity(isEnabled ? 0.88 : 0.34))
            .background(
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .fill(Color.white.opacity(configuration.isPressed ? 0.13 : 0.065))
            )
            .overlay(
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .strokeBorder(Color.white.opacity(0.10), lineWidth: 1)
            )
            .opacity(isEnabled ? 1 : 0.58)
            .contentShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
    }
}

private extension View {
    func benchmarkTextEditor(minHeight: CGFloat) -> some View {
        self
            .font(AppTheme.sans(14, weight: .regular))
            .foregroundStyle(.white)
            .scrollContentBackground(.hidden)
            .padding(10)
            .frame(minHeight: minHeight)
            .background(Color.white.opacity(0.04))
            .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .strokeBorder(Color.white.opacity(0.10), lineWidth: 1)
            )
    }
}
