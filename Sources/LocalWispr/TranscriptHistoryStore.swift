import Foundation

public actor TranscriptHistoryStore {
    private struct HistoryArchive: Codable {
        let version: Int
        let records: [TranscriptRecord]
    }

    private let fileManager: FileManager
    private let bundleIdentifier: String
    private let maxRecords: Int
    private let maxApproximateBytes: Int

    private let encoder: JSONEncoder
    private let decoder: JSONDecoder

    private var cachedRecords: [TranscriptRecord]?

    init(
        fileManager: FileManager = .default,
        bundleIdentifier: String = Bundle.main.bundleIdentifier ?? "LocalWispr",
        maxRecords: Int = 10_000,
        maxApproximateBytes: Int = 50 * 1_024 * 1_024
    ) {
        self.fileManager = fileManager
        self.bundleIdentifier = bundleIdentifier
        self.maxRecords = maxRecords
        self.maxApproximateBytes = maxApproximateBytes

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        self.encoder = encoder

        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        self.decoder = decoder
    }

    func loadRecords() throws -> [TranscriptRecord] {
        if let cachedRecords {
            return cachedRecords
        }

        let records = try readArchive().records
        cachedRecords = records
        return records
    }

    @discardableResult
    func append(_ record: TranscriptRecord) throws -> [TranscriptRecord] {
        var records = try loadRecords()
        records.insert(record, at: 0)
        records = try pruned(records)
        try writeArchive(records)
        cachedRecords = records
        return records
    }

    private func readArchive() throws -> HistoryArchive {
        let url = try historyFileURL()
        guard fileManager.fileExists(atPath: url.path) else {
            return HistoryArchive(version: 1, records: [])
        }

        let data = try Data(contentsOf: url)
        guard !data.isEmpty else {
            return HistoryArchive(version: 1, records: [])
        }

        let archive = try decoder.decode(HistoryArchive.self, from: data)
        let sortedRecords = archive.records.sorted(by: { $0.createdAt > $1.createdAt })
        return HistoryArchive(version: archive.version, records: sortedRecords)
    }

    private func writeArchive(_ records: [TranscriptRecord]) throws {
        let url = try historyFileURL()
        try fileManager.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)

        let archive = HistoryArchive(version: 1, records: records)
        let data = try encoder.encode(archive)
        try data.write(to: url, options: [.atomic])
    }

    private func pruned(_ inputRecords: [TranscriptRecord]) throws -> [TranscriptRecord] {
        var records = Array(inputRecords.prefix(maxRecords))

        while records.count > 1 {
            let archive = HistoryArchive(version: 1, records: records)
            let encoded = try encoder.encode(archive)
            if encoded.count <= maxApproximateBytes {
                break
            }
            records.removeLast()
        }

        return records
    }

    private func historyFileURL() throws -> URL {
        let appSupportURL = try fileManager.url(
            for: .applicationSupportDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        )

        return appSupportURL
            .appendingPathComponent(bundleIdentifier, isDirectory: true)
            .appendingPathComponent("history-v1.json", isDirectory: false)
    }
}
