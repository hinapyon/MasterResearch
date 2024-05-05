import Foundation
import CoreMotion

class MotionDataManager: ObservableObject {
    private var motionManager = CMMotionManager()
    private let updateInterval = 0.02 // 50Hz
    private var currentBuffer: [MotionData] = []
    private var backupBuffer: [MotionData] = []
    private var isUsingCurrentBuffer = true
    private var sessionStartTime: Date?
    private let saveQueue = DispatchQueue(label: "com.example.SaveMotionDataQueue", qos: .background)

    @Published var acceleration: (x: Double, y: Double, z: Double) = (0.0, 0.0, 0.0)
    @Published var gyro: (x: Double, y: Double, z: Double) = (0.0, 0.0, 0.0)

    var isDeviceMotionAvailable: Bool {
        motionManager.isDeviceMotionAvailable
    }

    func startUpdates() {
        if motionManager.isDeviceMotionAvailable {
            motionManager.deviceMotionUpdateInterval = updateInterval
            sessionStartTime = Date() // セッション開始時の時刻を記録
            let backgroundQueue = OperationQueue()
            backgroundQueue.qualityOfService = .userInitiated
            motionManager.startDeviceMotionUpdates(to: backgroundQueue) { [weak self] (motionData, error) in
                guard let self = self, let motion = motionData else { return }
                self.updateMotionData(motion)
            }
        }
    }

    func stopUpdates() {
        motionManager.stopDeviceMotionUpdates()
        let bufferToSave = isUsingCurrentBuffer ? currentBuffer : backupBuffer
        saveData(buffer: bufferToSave)
        if isUsingCurrentBuffer {
            currentBuffer.removeAll()
        } else {
            backupBuffer.removeAll()
        }
    }

    private func updateMotionData(_ motion: CMDeviceMotion) {
        let motionData = MotionData(
            timestamp: Date().timeIntervalSince1970,
            accelerationX: motion.userAcceleration.x,
            accelerationY: motion.userAcceleration.y,
            accelerationZ: motion.userAcceleration.z,
            gyroX: motion.rotationRate.x,
            gyroY: motion.rotationRate.y,
            gyroZ: motion.rotationRate.z
        )

        if isUsingCurrentBuffer {
            currentBuffer.append(motionData)
            if currentBuffer.count > 1000 {
                swapBuffers()
            }
        } else {
            backupBuffer.append(motionData)
        }
    }

    private func swapBuffers() {
        var bufferToSave = currentBuffer
        currentBuffer = backupBuffer
        backupBuffer = bufferToSave
        isUsingCurrentBuffer = !isUsingCurrentBuffer

        saveQueue.async {
            self.saveData(buffer: bufferToSave)
            DispatchQueue.main.async {
                bufferToSave.removeAll()
            }
        }
    }

    private func saveData(buffer: [MotionData]) {
        guard let url = createNewSessionFile() else { return }
        let encoder = JSONEncoder()
        do {
            let data = try encoder.encode(buffer)
            try data.write(to: url, options: .atomic)
            print("Data saved to \(url.path)")
        } catch {
            print("Failed to save data: \(error)")
        }
    }
    
    private func createNewSessionFile() -> URL? {
        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "yyyyMMdd_HHmmss"
        guard let startTime = sessionStartTime, let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else {
            return nil
        }
        let dateStr = dateFormatter.string(from: startTime)
        let fileName = "MotionData_\(dateStr).json"
        return documentsPath.appendingPathComponent(fileName)
    }
}
