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

//  メインのUI
struct ContentView: View {
    @State private var isRecording = false  //記録中かどうかを判定
    @State private var accelerationText = "X: 0.0, Y: 0.0, Z: 0.0"
    @State private var gyroText = "X: 0.0, Y: 0.0, Z: 0.0"

    var body: some View {
        VStack {
            Text("Acceleration")
            Text(accelerationText)
                .padding()
            Text("Gyro")
            Text(gyroText)
                .padding()

            Button(action: {
                self.toggleRecording()
            }) {
                Text(isRecording ? "Stop Recording" : "Start Recording")
            }
        }
        .onAppear() {
            self.setupMotionManager()
        }
    }

    let motionManager = CMMotionManager()   //CoreMotionのデータを扱うインスタンス作成

    //  motionManagerのセットアップを行う
    func setupMotionManager() {
        if WCSession.isSupported() {
            WCSession.default.activate()
        } else {
            print("WatchConnectivity not supported on this device")
        }

        if motionManager.isAccelerometerAvailable && motionManager.isGyroAvailable { // 加速度とジャイロが利用可能かどうか
            motionManager.accelerometerUpdateInterval = 0.1 // 加速度データのサンプリング周波数
            motionManager.gyroUpdateInterval = 0.1 // ジャイロデータのサンプリング周波数
        } else {
            print("Accelerometer or Gyroscope not available") // 利用できない場合はエラーメッセージをコンソールに出力
        }
    }

    // 記録と停止を繰り返す
    func toggleRecording() {
        if isRecording {
            stopRecording()
        } else {
            startRecording()
        }
    }

    // 記録開始
    func startRecording() {
        motionManager.startAccelerometerUpdates(to: OperationQueue.main) { (accelerationData, error) in
            // 加速度データの取得
            if let acceleration = accelerationData?.acceleration {
                // 角速度データの取得
                if let gyro = self.motionManager.gyroData?.rotationRate {
                    // タイムスタンプの取得
                    let timestamp = Date().timeIntervalSince1970
                    // タイムスタンプと加速度データとジャイロデータを同時に送信
                    self.sendDataToiPhone(timestamp: timestamp, acceleration: acceleration, gyro: gyro)
                    // 表示テキストの更新
                    self.updateAccelerationText(acceleration)
                    self.updateGyroText(gyro)
                }
            }
        }
        isRecording = true
    }

    // 加速度データの表示テキスト更新
    func updateAccelerationText(_ acceleration: CMAcceleration) {
        self.accelerationText = String(format: "X: %.2f, Y: %.2f, Z: %.2f", acceleration.x, acceleration.y, acceleration.z)
    }

    // ジャイロデータの表示テキスト更新
    func updateGyroText(_ gyro: CMRotationRate) {
        self.gyroText = String(format: "X: %.2f, Y: %.2f, Z: %.2f", gyro.x, gyro.y, gyro.z)
    }

    //  記録終了
    func stopRecording() {
        motionManager.stopAccelerometerUpdates()
        motionManager.stopGyroUpdates()
        isRecording = false
    }

    // データをiPhoneに送信
    func sendDataToiPhone(timestamp: TimeInterval, acceleration: CMAcceleration, gyro: CMRotationRate) {
        guard WCSession.default.isReachable else {
            print("iPhone not reachable")
            return
        }

        let message = ["timestamp": timestamp,
            "acceleration": [acceleration.x, acceleration.y, acceleration.z],
            "gyro": [gyro.x, gyro.y, gyro.z]] as [String : Any]

        WCSession.default.sendMessage(message, replyHandler: nil, errorHandler: { error in
            print("Error sending data to iPhone: \(error)")
        })
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
