@preconcurrency import AVFAudio
import AudioToolbox
import CoreAudio
import Foundation

public enum AudioCaptureError: LocalizedError {
    case engineAlreadyRunning
    case engineNotRunning
    case cannotCreateBuffer
    case conversionFailed(String)
    case deviceSelectionFailed(String)

    public var errorDescription: String? {
        switch self {
        case .engineAlreadyRunning:
            return "Audio capture is already running."
        case .engineNotRunning:
            return "Audio capture is not running."
        case .cannotCreateBuffer:
            return "Unable to allocate audio buffer."
        case .conversionFailed(let details):
            return "Unable to convert microphone audio: \(details)"
        case .deviceSelectionFailed(let details):
            return "Unable to use the selected microphone: \(details)"
        }
    }
}

public final class AudioCapture: @unchecked Sendable, AudioCapturing {
    private final class ConversionInputState: @unchecked Sendable {
        var didProvideInput = false
    }

    private let engine = AVAudioEngine()
    private let bufferQueue = DispatchQueue(label: "LocalWispr.AudioCapture.buffer")

    private var isRunning = false
    private var capturedBufferCount = 0
    private var bufferedAudio: [AVAudioPCMBuffer] = []
    private var capturedBufferHandler: (@Sendable (AVAudioPCMBuffer) -> Void)?

    private let targetFormat: AVAudioFormat = {
        AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: 16_000,
            channels: 1,
            interleaved: false
        )!
    }()

    public init() {}

    public var outputAudioFormat: AVAudioFormat {
        targetFormat
    }

    public var preferredInputDeviceID: UInt32?

    public var onBufferCaptured: (@Sendable (AVAudioPCMBuffer) -> Void)? {
        get {
            bufferQueue.sync { capturedBufferHandler }
        }
        set {
            bufferQueue.sync { capturedBufferHandler = newValue }
        }
    }

    public func start() throws {
        try bufferQueue.sync {
            if isRunning {
                throw AudioCaptureError.engineAlreadyRunning
            }

            bufferedAudio.removeAll(keepingCapacity: true)
            capturedBufferCount = 0

            let inputNode = engine.inputNode
            try self.configureInputDevice(for: inputNode)
            let inputFormat = inputNode.outputFormat(forBus: 0)
            DebugLog.write("[AudioCapture] inputDevice=\(preferredInputDeviceID.map(String.init) ?? "system-default") inputFormat=\(inputFormat) targetFormat=\(targetFormat)")

            inputNode.removeTap(onBus: 0)
            inputNode.installTap(onBus: 0, bufferSize: 1024, format: inputFormat) { [weak self] buffer, _ in
                self?.handleIncomingBuffer(buffer)
            }

            engine.prepare()
            do {
                try engine.start()
                isRunning = true
            } catch {
                inputNode.removeTap(onBus: 0)
                throw error
            }
        }
    }

    public func stopAndDrain() throws -> [AVAudioPCMBuffer] {
        try bufferQueue.sync {
            guard isRunning else {
                throw AudioCaptureError.engineNotRunning
            }

            engine.inputNode.removeTap(onBus: 0)
            engine.stop()
            isRunning = false

            let result = bufferedAudio
            bufferedAudio = []
            return result
        }
    }

    private func handleIncomingBuffer(_ inputBuffer: AVAudioPCMBuffer) {
        bufferQueue.async { [weak self] in
            guard let self else { return }

            do {
                let normalized = try self.normalizeToTargetFormat(inputBuffer)
                self.capturedBufferCount += 1
                if self.capturedBufferCount <= 3 || self.capturedBufferCount.isMultiple(of: 50) {
                    DebugLog.write(
                        "[AudioCapture] buffer #\(self.capturedBufferCount) raw=\(Self.describeLevel(inputBuffer)) normalized=\(Self.describeLevel(normalized))"
                    )
                }
                self.bufferedAudio.append(normalized)
                self.capturedBufferHandler?(normalized)
            } catch {
                // Drop malformed chunks; the pipeline still processes whatever was captured.
            }
        }
    }

    private func normalizeToTargetFormat(_ sourceBuffer: AVAudioPCMBuffer) throws -> AVAudioPCMBuffer {
        if sourceBuffer.format.sampleRate == targetFormat.sampleRate,
           sourceBuffer.format.channelCount == targetFormat.channelCount,
           sourceBuffer.format.commonFormat == targetFormat.commonFormat,
           sourceBuffer.format.isInterleaved == targetFormat.isInterleaved {
            return try copy(buffer: sourceBuffer)
        }

        guard let converter = AVAudioConverter(from: sourceBuffer.format, to: targetFormat) else {
            throw AudioCaptureError.conversionFailed("converter initialization failed")
        }

        let ratio = targetFormat.sampleRate / sourceBuffer.format.sampleRate
        let capacity = AVAudioFrameCount((Double(sourceBuffer.frameLength) * ratio).rounded(.up)) + 8

        guard let convertedBuffer = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: max(capacity, 64)) else {
            throw AudioCaptureError.cannotCreateBuffer
        }

        let inputState = ConversionInputState()
        var conversionError: NSError?

        let status = converter.convert(to: convertedBuffer, error: &conversionError) { _, outStatus in
            if inputState.didProvideInput {
                outStatus.pointee = .endOfStream
                return nil
            }

            inputState.didProvideInput = true
            outStatus.pointee = .haveData
            return sourceBuffer
        }

        if status == .error {
            let detail = conversionError?.localizedDescription ?? "unknown conversion error"
            throw AudioCaptureError.conversionFailed(detail)
        }

        if convertedBuffer.frameLength == 0 {
            throw AudioCaptureError.conversionFailed("converted frame length is zero")
        }

        return convertedBuffer
    }

    private func copy(buffer: AVAudioPCMBuffer) throws -> AVAudioPCMBuffer {
        guard let copy = AVAudioPCMBuffer(pcmFormat: buffer.format, frameCapacity: buffer.frameLength) else {
            throw AudioCaptureError.cannotCreateBuffer
        }

        copy.frameLength = buffer.frameLength

        let channelCount = Int(buffer.format.channelCount)
        let frameLength = Int(buffer.frameLength)
        let isInterleaved = buffer.format.isInterleaved

        switch buffer.format.commonFormat {
        case .pcmFormatFloat32:
            guard let source = buffer.floatChannelData, let destination = copy.floatChannelData else {
                throw AudioCaptureError.cannotCreateBuffer
            }

            if isInterleaved {
                memcpy(destination[0], source[0], frameLength * channelCount * MemoryLayout<Float>.size)
            } else {
                for channel in 0..<channelCount {
                    memcpy(destination[channel], source[channel], frameLength * MemoryLayout<Float>.size)
                }
            }
        case .pcmFormatInt16:
            guard let source = buffer.int16ChannelData, let destination = copy.int16ChannelData else {
                throw AudioCaptureError.cannotCreateBuffer
            }

            if isInterleaved {
                memcpy(destination[0], source[0], frameLength * channelCount * MemoryLayout<Int16>.size)
            } else {
                for channel in 0..<channelCount {
                    memcpy(destination[channel], source[channel], frameLength * MemoryLayout<Int16>.size)
                }
            }
        case .pcmFormatInt32:
            guard let source = buffer.int32ChannelData, let destination = copy.int32ChannelData else {
                throw AudioCaptureError.cannotCreateBuffer
            }

            if isInterleaved {
                memcpy(destination[0], source[0], frameLength * channelCount * MemoryLayout<Int32>.size)
            } else {
                for channel in 0..<channelCount {
                    memcpy(destination[channel], source[channel], frameLength * MemoryLayout<Int32>.size)
                }
            }
        default:
            throw AudioCaptureError.conversionFailed("unsupported sample format")
        }

        return copy
    }

    private func configureInputDevice(for inputNode: AVAudioInputNode) throws {
        guard let preferredInputDeviceID else { return }
        guard let audioUnit = inputNode.audioUnit else {
            throw AudioCaptureError.deviceSelectionFailed("audio unit unavailable")
        }

        var deviceID = AudioDeviceID(preferredInputDeviceID)
        let status = AudioUnitSetProperty(
            audioUnit,
            kAudioOutputUnitProperty_CurrentDevice,
            kAudioUnitScope_Global,
            0,
            &deviceID,
            UInt32(MemoryLayout<AudioDeviceID>.size)
        )

        guard status == noErr else {
            throw AudioCaptureError.deviceSelectionFailed("OSStatus \(status)")
        }
    }

    private static func describeLevel(_ buffer: AVAudioPCMBuffer) -> String {
        let frameCount = Int(buffer.frameLength)
        guard frameCount > 0 else { return "empty" }

        switch buffer.format.commonFormat {
        case .pcmFormatFloat32:
            guard let channels = buffer.floatChannelData else { return "unavailable" }
            let samples = UnsafeBufferPointer(start: channels[0], count: frameCount)
            let rms = sqrt(samples.reduce(0.0) { $0 + Double($1 * $1) } / Double(frameCount))
            let peak = samples.reduce(0.0) { max($0, Double(abs($1))) }
            return String(format: "float rms=%.5f peak=%.5f", rms, peak)

        case .pcmFormatInt16:
            guard let channels = buffer.int16ChannelData else { return "unavailable" }
            let samples = UnsafeBufferPointer(start: channels[0], count: frameCount)
            let scale = Double(Int16.max)
            let rms = sqrt(samples.reduce(0.0) { partial, sample in
                let normalized = Double(sample) / scale
                return partial + (normalized * normalized)
            } / Double(frameCount))
            let peak = samples.reduce(0.0) { partial, sample in
                max(partial, abs(Double(sample) / scale))
            }
            return String(format: "int16 rms=%.5f peak=%.5f", rms, peak)

        default:
            return "unsupported"
        }
    }
}
