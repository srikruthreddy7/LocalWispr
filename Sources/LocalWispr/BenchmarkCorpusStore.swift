import AVFoundation
import Foundation

public enum BenchmarkClipCategory: String, CaseIterable, Identifiable, Sendable, Codable {
    case shortCommand
    case dictation
    case longForm
    case developer
    case noisyRoom
    case hardAccent

    public var id: String { rawValue }

    public var title: String {
        switch self {
        case .shortCommand:
            return "Short Command"
        case .dictation:
            return "Normal Dictation"
        case .longForm:
            return "Long Paragraph"
        case .developer:
            return "Developer / Code"
        case .noisyRoom:
            return "Noisy Room"
        case .hardAccent:
            return "Accent Hard Case"
        }
    }

    public var targetCount: Int {
        switch self {
        case .shortCommand:
            return 50
        case .dictation:
            return 50
        case .longForm:
            return 30
        case .developer:
            return 30
        case .noisyRoom:
            return 20
        case .hardAccent:
            return 30
        }
    }

    public var folderName: String {
        rawValue
    }
}

public struct BenchmarkClipManifestEntry: Identifiable, Sendable, Equatable, Codable {
    public let id: UUID
    public let createdAt: Date
    public let category: BenchmarkClipCategory
    public let promptText: String
    public let referenceText: String
    public let audioFilename: String
    public let audioPath: String
    public let durationMilliseconds: Int?
    public let inputDeviceName: String

    public init(
        id: UUID = UUID(),
        createdAt: Date = Date(),
        category: BenchmarkClipCategory,
        promptText: String,
        referenceText: String,
        audioFilename: String,
        audioPath: String,
        durationMilliseconds: Int?,
        inputDeviceName: String
    ) {
        self.id = id
        self.createdAt = createdAt
        self.category = category
        self.promptText = promptText
        self.referenceText = referenceText
        self.audioFilename = audioFilename
        self.audioPath = audioPath
        self.durationMilliseconds = durationMilliseconds
        self.inputDeviceName = inputDeviceName
    }
}

public final class BenchmarkCorpusStore: Sendable {
    public let rootURL: URL

    private var manifestURL: URL {
        rootURL.appendingPathComponent("manifest.json")
    }

    public init(rootURL: URL = BenchmarkCorpusStore.defaultRootURL()) {
        self.rootURL = rootURL
    }

    public static func defaultRootURL() -> URL {
        let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first
        return (documents ?? URL(fileURLWithPath: NSHomeDirectory(), isDirectory: true))
            .appendingPathComponent("LocalWispr", isDirectory: true)
            .appendingPathComponent("BenchmarkCorpus", isDirectory: true)
    }

    public func loadManifest() throws -> [BenchmarkClipManifestEntry] {
        guard FileManager.default.fileExists(atPath: manifestURL.path) else {
            return []
        }

        let data = try Data(contentsOf: manifestURL)
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return try decoder.decode([BenchmarkClipManifestEntry].self, from: data)
            .sorted { $0.createdAt > $1.createdAt }
    }

    public func saveClip(
        audioURL: URL,
        category: BenchmarkClipCategory,
        promptText: String,
        referenceText: String,
        inputDeviceName: String
    ) throws -> BenchmarkClipManifestEntry {
        let trimmedReference = referenceText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedReference.isEmpty else {
            throw BenchmarkCorpusStoreError.emptyReference
        }

        let fm = FileManager.default
        let clipsDirectory = rootURL
            .appendingPathComponent("clips", isDirectory: true)
            .appendingPathComponent(category.folderName, isDirectory: true)
        try fm.createDirectory(at: clipsDirectory, withIntermediateDirectories: true)

        let id = UUID()
        let stamp = Self.filenameDateFormatter.string(from: Date())
        let basename = "\(stamp)-\(id.uuidString.lowercased())"
        let destinationURL = clipsDirectory.appendingPathComponent("\(basename).wav")
        let referenceURL = clipsDirectory.appendingPathComponent("\(basename).txt")

        if fm.fileExists(atPath: destinationURL.path) {
            try fm.removeItem(at: destinationURL)
        }
        try fm.copyItem(at: audioURL, to: destinationURL)
        try trimmedReference.write(to: referenceURL, atomically: true, encoding: .utf8)

        let entry = BenchmarkClipManifestEntry(
            id: id,
            createdAt: Date(),
            category: category,
            promptText: promptText.trimmingCharacters(in: .whitespacesAndNewlines),
            referenceText: trimmedReference,
            audioFilename: destinationURL.lastPathComponent,
            audioPath: destinationURL.path,
            durationMilliseconds: Self.durationMilliseconds(for: destinationURL),
            inputDeviceName: inputDeviceName
        )

        var manifest = try loadManifest()
        manifest.insert(entry, at: 0)
        try writeManifest(manifest)
        return entry
    }

    private func writeManifest(_ entries: [BenchmarkClipManifestEntry]) throws {
        try FileManager.default.createDirectory(at: rootURL, withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        let data = try encoder.encode(entries.sorted { $0.createdAt > $1.createdAt })
        try data.write(to: manifestURL, options: [.atomic])
    }

    private static func durationMilliseconds(for audioURL: URL) -> Int? {
        guard let file = try? AVAudioFile(forReading: audioURL) else {
            return nil
        }

        let seconds = Double(file.length) / file.fileFormat.sampleRate
        guard seconds.isFinite, seconds > 0 else {
            return nil
        }

        return Int((seconds * 1_000.0).rounded())
    }

    private static let filenameDateFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = .current
        formatter.dateFormat = "yyyyMMdd-HHmmss"
        return formatter
    }()
}

public enum BenchmarkCorpusStoreError: LocalizedError, Equatable {
    case emptyReference

    public var errorDescription: String? {
        switch self {
        case .emptyReference:
            return "Add the exact words you said before saving this benchmark clip."
        }
    }
}
