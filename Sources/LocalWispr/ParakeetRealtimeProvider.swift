import AVFoundation
import Foundation

#if arch(arm64)
@preconcurrency import CoreML
import FluidAudio

final class ParakeetRealtimeProvider: @unchecked Sendable {
    let name = "Parakeet Flash (FluidAudio)"

    var isAvailable: Bool { true }
    private(set) var isReady: Bool = false

    private let chunkSize: StreamingChunkSize
    private var engine: StreamingEouAsrManager?
    private var streamedSampleCount: Int = 0

    init(chunkSize: StreamingChunkSize = .ms160) {
        self.chunkSize = chunkSize
    }

    func prepare(progressHandler: (@Sendable (Double) -> Void)? = nil) async throws {
        guard self.isReady == false else { return }

        let configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuAndNeuralEngine
        configuration.allowLowPrecisionAccumulationOnGPU = true

        let engine = StreamingEouAsrManager(configuration: configuration, chunkSize: self.chunkSize)
        try await engine.loadModelsFromHuggingFace(progressHandler: { progress in
            progressHandler?(max(0.0, min(1.0, progress.fractionCompleted)))
        })

        self.engine = engine
        self.streamedSampleCount = 0
        self.isReady = true
        DebugLog.write("[Parakeet] ready chunk=\(self.chunkSize.modelSubdirectory)")
    }

    func transcribeStreaming(_ samples: [Float]) async throws -> String {
        let engine = try self.requireEngine()
        let delta = try await self.consumeDelta(from: samples, engine: engine)
        if !delta.isEmpty {
            try await engine.appendAudio(self.createPCMBuffer(from: delta))
            try await engine.processBufferedAudio()
        }
        return await engine.getPartialTranscript().trimmingCharacters(in: .whitespacesAndNewlines)
    }

    func transcribeFinal(_ samples: [Float]) async throws -> String {
        let engine = try self.requireEngine()
        let delta = try await self.consumeDelta(from: samples, engine: engine)
        if !delta.isEmpty {
            try await engine.appendAudio(self.createPCMBuffer(from: delta))
            try await engine.processBufferedAudio()
        }

        let text = try await engine.finish().trimmingCharacters(in: .whitespacesAndNewlines)
        await engine.reset()
        self.streamedSampleCount = 0
        return text
    }

    func resetSession() async {
        if let engine = self.engine {
            await engine.reset()
        }
        self.streamedSampleCount = 0
    }

    private func requireEngine() throws -> StreamingEouAsrManager {
        guard let engine = self.engine else {
            throw NSError(
                domain: "ParakeetRealtimeProvider",
                code: -1,
                userInfo: [NSLocalizedDescriptionKey: "Parakeet engine is not initialized"]
            )
        }
        return engine
    }

    private func consumeDelta(from samples: [Float], engine: StreamingEouAsrManager) async throws -> [Float] {
        if samples.count < self.streamedSampleCount {
            await engine.reset()
            self.streamedSampleCount = 0
        }

        let delta = Array(samples.dropFirst(self.streamedSampleCount))
        self.streamedSampleCount = samples.count
        return delta
    }

    private func createPCMBuffer(from samples: [Float]) throws -> AVAudioPCMBuffer {
        guard
            let format = AVAudioFormat(
                commonFormat: .pcmFormatFloat32,
                sampleRate: 16_000,
                channels: 1,
                interleaved: false
            ),
            let buffer = AVAudioPCMBuffer(
                pcmFormat: format,
                frameCapacity: AVAudioFrameCount(samples.count)
            ),
            let channelData = buffer.floatChannelData
        else {
            throw NSError(
                domain: "ParakeetRealtimeProvider",
                code: -2,
                userInfo: [NSLocalizedDescriptionKey: "Failed to create audio buffer for Parakeet"]
            )
        }

        buffer.frameLength = AVAudioFrameCount(samples.count)
        samples.withUnsafeBufferPointer { samplePtr in
            guard let baseAddress = samplePtr.baseAddress else { return }
            channelData[0].update(from: baseAddress, count: samples.count)
        }
        return buffer
    }
}
#else
final class ParakeetRealtimeProvider: @unchecked Sendable {
    let name = "Parakeet Flash (FluidAudio)"
    var isAvailable: Bool { false }
    var isReady: Bool { false }

    func prepare(progressHandler: (@Sendable (Double) -> Void)? = nil) async throws {
        throw NSError(
            domain: "ParakeetRealtimeProvider",
            code: -1,
            userInfo: [NSLocalizedDescriptionKey: "Parakeet requires Apple Silicon"]
        )
    }

    func transcribeStreaming(_ samples: [Float]) async throws -> String {
        throw NSError(
            domain: "ParakeetRealtimeProvider",
            code: -1,
            userInfo: [NSLocalizedDescriptionKey: "Parakeet requires Apple Silicon"]
        )
    }

    func transcribeFinal(_ samples: [Float]) async throws -> String {
        throw NSError(
            domain: "ParakeetRealtimeProvider",
            code: -1,
            userInfo: [NSLocalizedDescriptionKey: "Parakeet requires Apple Silicon"]
        )
    }

    func resetSession() async {}
}
#endif
