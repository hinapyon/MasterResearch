//
//  MotionDataManager.swift
//  MasterResearch Watch App
//
//  Created by Kawano Hinase on 2024/01/31.
//

import Foundation
import CoreMotion

class MotionDataManager: ObservableObject {
    private var motionManager = CMMotionManager()
    @Published var accelerationText = "X: 0.0, Y: 0.0, Z: 0.0"
    @Published var gyroText = "X: 0.0, Y: 0.0, Z: 0.0"
    private let updateInterval = 1.0 // 1Hz

    var isDeviceMotionAvailable: Bool {
        motionManager.isDeviceMotionAvailable
    }

    func startUpdates() {
        if motionManager.isDeviceMotionAvailable {
            motionManager.deviceMotionUpdateInterval = updateInterval
            motionManager.startDeviceMotionUpdates(to: OperationQueue.main) { [weak self] (motionData, error) in
                guard let self = self, let motion = motionData else { return }
                self.updateMotionData(motion)
            }
        }
    }

    func stopUpdates() {
        motionManager.stopDeviceMotionUpdates()
    }

    private func updateMotionData(_ motion: CMDeviceMotion) {
        let acceleration = motion.userAcceleration
        accelerationText = String(format: "X: %.2f, Y: %.2f, Z: %.2f", acceleration.x, acceleration.y, acceleration.z)
        let gyro = motion.rotationRate
        gyroText = String(format: "X: %.2f, Y: %.2f, Z: %.2f", gyro.x, gyro.y, gyro.z)

        // iPhoneにデータを送信
        let timestamp = Date().timeIntervalSince1970
        let data: [String: Any] = [
            "timestamp": timestamp,
            "accelerationX": acceleration.x,
            "accelerationY": acceleration.y,
            "accelerationZ": acceleration.z,
            "gyroX": gyro.x,
            "gyroY": gyro.y,
            "gyroZ": gyro.z
        ]

        WatchSessionManager.shared.sendMessage(data)
    }
}