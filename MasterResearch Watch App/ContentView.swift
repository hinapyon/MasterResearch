//
//  ContentView.swift
//  MasterResearch Watch App
//
//  Created by Kawano Hinase on 2024/01/09.
//

// ライブラリのインクルード
import SwiftUI  //AppleのUIフレームワーク
import CoreMotion   //モーションデータを利用するためのフレームワーク
import WatchConnectivity    //iPhoneと通信するためのフレームワーク

// モーションデータ管理クラス
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
        sendMotionDataToiPhone(motion)
    }
    
    // MotionDataManager内に追加
    func sendMotionDataToiPhone(_ motion: CMDeviceMotion) {
        guard WCSession.isSupported() else { return }
        let session = WCSession.default

        if session.isReachable {
            let timestamp = Date().timeIntervalSince1970
            let data: [String: Any] = [
                "timestamp": timestamp,
                "accelerationX": motion.userAcceleration.x,
                "accelerationY": motion.userAcceleration.y,
                "accelerationZ": motion.userAcceleration.z,
                "gyroX": motion.rotationRate.x,
                "gyroY": motion.rotationRate.y,
                "gyroZ": motion.rotationRate.z
            ]

            session.sendMessage(data, replyHandler: nil) { error in
                print("Error sending message: \(error.localizedDescription)")
            }
        }
    }
}

// メインのUI
struct ContentView: View {
    @ObservedObject private var motionData = MotionDataManager()
    @State private var isRecording = false

    var body: some View {
        VStack {
            if motionData.isDeviceMotionAvailable {
                Text("Acceleration")
                Text(motionData.accelerationText)
                    .padding()
                Text("Gyro")
                Text(motionData.gyroText)
                    .padding()

                Button(action: toggleRecording) {
                    Text(isRecording ? "Stop Recording" : "Start Recording")
                }
            } else {
                Text("Device Motion Not Available")
                    .foregroundColor(.red)
            }
        }
        .onAppear {
            // 初期化時にデバイスモーションの可用性をチェック
            self.isRecording = false
        }
    }

    func toggleRecording() {
        if isRecording {
            motionData.stopUpdates()
        } else {
            motionData.startUpdates()
        }
        isRecording.toggle()
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
