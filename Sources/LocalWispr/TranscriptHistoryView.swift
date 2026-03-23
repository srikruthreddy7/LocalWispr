import SwiftUI

public struct TranscriptHistoryView: View {
    @ObservedObject var appState: AppState

    public init(appState: AppState) {
        self.appState = appState
    }

    public var body: some View {
        Group {
            if appState.transcriptHistory.isEmpty {
                emptyState
            } else {
                HStack(alignment: .top, spacing: 20) {
                    recordsList
                        .frame(width: 340)

                    recordDetail
                        .frame(maxWidth: .infinity, alignment: .topLeading)
                }
            }
        }
    }

    private var emptyState: some View {
        HistoryCard(cornerRadius: 30, padding: 28) {
            VStack(alignment: .leading, spacing: 16) {
                Text("History")
                    .font(AppTheme.sans(34, weight: .semibold))
                    .foregroundStyle(.white)

                Text("Completed dictation sessions will appear here once you start using LocalWispr. Both raw and cleaned text are stored locally on this Mac.")
                    .font(AppTheme.sans(15, weight: .regular))
                    .foregroundStyle(AppTheme.secondaryText)
                    .lineSpacing(3)

                Text("Nothing saved yet.")
                    .font(AppTheme.mono(12, weight: .medium))
                    .foregroundStyle(AppTheme.tertiaryText)
            }
            .frame(maxWidth: .infinity, minHeight: 420, alignment: .leading)
        }
    }

    private var recordsList: some View {
        HistoryCard(cornerRadius: 30, padding: 20) {
            VStack(alignment: .leading, spacing: 16) {
                VStack(alignment: .leading, spacing: 6) {
                    Text("History".uppercased())
                        .font(AppTheme.mono(11, weight: .medium))
                        .tracking(1.6)
                        .foregroundStyle(AppTheme.tertiaryText)

                    Text("\(appState.transcriptHistory.count) local sessions")
                        .font(AppTheme.sans(18, weight: .semibold))
                        .foregroundStyle(.white)

                    Text("Newest first. Stored only on this Mac.")
                        .font(AppTheme.sans(13, weight: .regular))
                        .foregroundStyle(AppTheme.secondaryText)
                }

                ScrollView(.vertical, showsIndicators: false) {
                    LazyVStack(spacing: 10) {
                        ForEach(appState.transcriptHistory) { record in
                            recordRow(for: record)
                        }
                    }
                    .padding(.vertical, 2)
                }
            }
        }
    }

    private func recordRow(for record: TranscriptRecord) -> some View {
        let isSelected = appState.selectedTranscriptRecordID == record.id

        return Button {
            appState.selectedTranscriptRecordID = record.id
        } label: {
            VStack(alignment: .leading, spacing: 10) {
                HStack(alignment: .top, spacing: 12) {
                    VStack(alignment: .leading, spacing: 4) {
                        Text(record.statusSummary.uppercased())
                            .font(AppTheme.mono(10, weight: .medium))
                            .tracking(1.2)
                            .foregroundStyle(isSelected ? Color.black.opacity(0.68) : AppTheme.tertiaryText)

                        Text(record.transcriberMode.title)
                            .font(AppTheme.sans(14, weight: .medium))
                            .foregroundStyle(isSelected ? Color.black.opacity(0.82) : .white)
                    }

                    Spacer(minLength: 10)

                    Text(record.createdAt.formatted(date: .abbreviated, time: .shortened))
                        .font(AppTheme.mono(10, weight: .regular))
                        .foregroundStyle(isSelected ? Color.black.opacity(0.62) : AppTheme.tertiaryText)
                        .multilineTextAlignment(.trailing)
                }

                Text(record.preferredDisplayText)
                    .font(AppTheme.sans(14, weight: .regular))
                    .foregroundStyle(isSelected ? Color.black.opacity(0.82) : AppTheme.secondaryText)
                    .lineLimit(3)
                    .multilineTextAlignment(.leading)
            }
            .padding(16)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(
                Group {
                    if isSelected {
                        RoundedRectangle(cornerRadius: 22, style: .continuous)
                            .fill(Color.white.opacity(0.9))
                    } else {
                        RoundedRectangle(cornerRadius: 22, style: .continuous)
                            .fill(Color.white.opacity(0.03))
                            .overlay(
                                RoundedRectangle(cornerRadius: 22, style: .continuous)
                                    .strokeBorder(Color.white.opacity(0.05), lineWidth: 1)
                            )
                    }
                }
            )
        }
        .buttonStyle(.plain)
    }

    private var recordDetail: some View {
        HistoryCard(cornerRadius: 30, padding: 24) {
            if let record = appState.selectedTranscriptRecord {
                ScrollView(.vertical, showsIndicators: false) {
                    VStack(alignment: .leading, spacing: 18) {
                        VStack(alignment: .leading, spacing: 8) {
                            Text("Session Detail")
                                .font(AppTheme.sans(30, weight: .semibold))
                                .foregroundStyle(.white)

                            Text(record.createdAt.formatted(date: .complete, time: .standard))
                                .font(AppTheme.sans(14, weight: .regular))
                                .foregroundStyle(AppTheme.secondaryText)
                        }

                        detailMetaGrid(for: record)

                        if let errorMessage = record.errorMessage, !errorMessage.isEmpty {
                            HistoryMessageCard(title: "Issue", text: errorMessage)
                        } else if let cleanupWarning = record.cleanupWarning, !cleanupWarning.isEmpty {
                            HistoryMessageCard(title: "Warning", text: cleanupWarning)
                        }

                        HStack(alignment: .top, spacing: 18) {
                            HistoryTextPane(
                                title: "Raw Transcript",
                                subtitle: "Verbatim speech output captured for this session.",
                                text: record.rawText,
                                placeholder: "No raw transcript was stored for this session."
                            )

                            HistoryTextPane(
                                title: "Cleaned Transcript",
                                subtitle: "Final cleaned output that LocalWispr produced.",
                                text: record.cleanedText,
                                placeholder: "No cleaned transcript was stored for this session."
                            )
                        }
                    }
                }
            } else {
                VStack(alignment: .leading, spacing: 12) {
                    Text("Select a session")
                        .font(AppTheme.sans(24, weight: .semibold))
                        .foregroundStyle(.white)

                    Text("Choose a saved transcript from the left to inspect raw text, cleaned text, and session metadata.")
                        .font(AppTheme.sans(14, weight: .regular))
                        .foregroundStyle(AppTheme.secondaryText)
                }
                .frame(maxWidth: .infinity, minHeight: 420, alignment: .leading)
            }
        }
    }

    private func detailMetaGrid(for record: TranscriptRecord) -> some View {
        LazyVGrid(columns: [
            GridItem(.flexible(minimum: 140)),
            GridItem(.flexible(minimum: 140)),
            GridItem(.flexible(minimum: 140)),
            GridItem(.flexible(minimum: 140))
        ], alignment: .leading, spacing: 12) {
            HistoryStatChip(title: "STATUS", value: record.statusSummary)
            HistoryStatChip(title: "MODE", value: record.transcriberMode.title)
            HistoryStatChip(title: "LOCALE", value: record.localeIdentifier)
            HistoryStatChip(title: "WORDS", value: "\(maxWordCount(for: record))")

            if let latency = record.latency {
                HistoryStatChip(title: "TOTAL", value: "\(latency.totalStopToInsertMilliseconds)ms")
                HistoryStatChip(title: "TRANSCRIBE", value: "\(latency.stopToTranscriptMilliseconds)ms")
                HistoryStatChip(title: "CLEANUP", value: "\(latency.cleanupMilliseconds)ms")
                HistoryStatChip(title: "INSERT", value: "\(latency.insertionMilliseconds)ms")
            }
        }
    }

    private func maxWordCount(for record: TranscriptRecord) -> Int {
        let rawCount = record.rawText.split(whereSeparator: \.isWhitespace).count
        let cleanedCount = record.cleanedText.split(whereSeparator: \.isWhitespace).count
        return max(rawCount, cleanedCount)
    }
}

private struct HistoryCard<Content: View>: View {
    let cornerRadius: CGFloat
    let padding: CGFloat
    @ViewBuilder let content: Content

    var body: some View {
        content
            .padding(padding)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(AppTheme.panelFill)
            .glassEffect(.regular, in: RoundedRectangle(cornerRadius: cornerRadius, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                    .strokeBorder(Color.white.opacity(0.08), lineWidth: 1)
            )
            .shadow(color: Color.black.opacity(0.22), radius: 24, y: 16)
    }
}

private struct HistoryStatChip: View {
    let title: String
    let value: String

    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            Text(title)
                .font(AppTheme.mono(10, weight: .medium))
                .tracking(1.2)
                .foregroundStyle(AppTheme.tertiaryText)

            Text(value)
                .font(AppTheme.sans(14, weight: .medium))
                .foregroundStyle(.white)
                .lineLimit(2)
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.white.opacity(0.028))
        .glassEffect(.regular, in: RoundedRectangle(cornerRadius: 18, style: .continuous))
    }
}

private struct HistoryMessageCard: View {
    let title: String
    let text: String

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title.uppercased())
                .font(AppTheme.mono(10, weight: .medium))
                .tracking(1.2)
                .foregroundStyle(AppTheme.tertiaryText)

            Text(text)
                .font(AppTheme.sans(14, weight: .regular))
                .foregroundStyle(AppTheme.secondaryText)
                .lineSpacing(2)
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.white.opacity(0.03))
        .glassEffect(.regular, in: RoundedRectangle(cornerRadius: 20, style: .continuous))
    }
}

private struct HistoryTextPane: View {
    let title: String
    let subtitle: String
    let text: String
    let placeholder: String

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            VStack(alignment: .leading, spacing: 5) {
                Text(title)
                    .font(AppTheme.sans(22, weight: .semibold))
                    .foregroundStyle(.white)

                Text(subtitle)
                    .font(AppTheme.sans(13, weight: .regular))
                    .foregroundStyle(AppTheme.secondaryText)
                    .lineSpacing(2)
            }

            ScrollView(.vertical, showsIndicators: false) {
                Text(displayText)
                    .font(AppTheme.sans(15, weight: .regular))
                    .foregroundStyle(displayColor)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .multilineTextAlignment(.leading)
                    .lineSpacing(3)
                    .textSelection(.enabled)
                    .padding(.bottom, 8)
            }
            .frame(minHeight: 320, maxHeight: 420)
        }
        .padding(20)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.white.opacity(0.028))
        .glassEffect(.regular, in: RoundedRectangle(cornerRadius: 24, style: .continuous))
    }

    private var displayText: String {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? placeholder : trimmed
    }

    private var displayColor: Color {
        text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? AppTheme.tertiaryText : .white
    }
}
