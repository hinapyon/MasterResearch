import Foundation
import CoreMotion

class MotionDataManager: ObservableObject {
    private var motionManager = CMMotionManager()
    private let updateInterval = 0.02 // 50Hz
    private var motionBuffer: [MotionData] = []
    private let bufferQueue = DispatchQueue(label: "com.example.MotionBufferQueue", attributes: .concurrent)
    private let saveQueue = DispatchQueue(label: "com.example.SaveMotionDataQueue", qos: .background)
    private var sessionStartTime: Date?

    @Published var acceleration: (x: Double, y: Double, z: Double) = (0.0, 0.0, 0.0)
    @Published var gyro: (x: Double, y: Double, z: Double) = (0.0, 0.0, 0.0)

    var isDeviceMotionAvailable: Bool {
        motionManager.isDeviceMotionAvailable
    }

    func startUpdates() {
        if motionManager.isDeviceMotionAvailable {
            motionManager.deviceMotionUpdateInterval = updateInterval
            sessionStartTime = Date()
            let backgroundQueue = OperationQueue()
            backgroundQueue.qualityOfService = .userInitiated
            motionManager.startDeviceMotionUpdates(to: backgroundQueue) { [weak self] (motionData, error) in
                guard let self = self, let motion = motionData else { return }
                self.bufferQueue.async(flags: .barrier) {
                    self.updateMotionData(motion)
                }
            }
        }
    }

    func stopUpdates() {
        motionManager.stopDeviceMotionUpdates()
        saveQueue.async {
            self.saveData(buffer: self.motionBuffer)
            DispatchQueue.main.async {
                self.motionBuffer.removeAll()
            }
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
        motionBuffer.append(motionData)
    }

    private func saveData(buffer: [MotionData]) {
        guard let url = createNewSessionFile(), !buffer.isEmpty else { return }
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
