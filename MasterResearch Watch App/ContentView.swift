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
    @State private var isRecording = false  // 記録中かどうかの状態を管理
    @State private var accelerationText = "X: 0.0, Y: 0.0, Z: 0.0"  // 加速度データの表示テキスト
    @State private var gyroText = "X: 0.0, Y: 0.0, Z: 0.0"  // ジャイロデータの表示テキスト

    // 画面表示構成
    var body: some View {
        VStack {
            Text("Acceleration")
            Text(accelerationText)
                .padding()
            Text("Gyro")
            Text(gyroText)
                .padding()

            // 記録開始・停止ボタン
            Button(action: {
                self.toggleRecording()
            }) {
                Text(isRecording ? "Stop Recording" : "Start Recording")
            }
        }
        .onAppear() {   // Viewが表示されるタイミングで一度だけ実行される
            // モーションデータのセットアップ
            self.setupMotionManager()
        }
    }

    let motionManager = CMMotionManager()    // CoreMotionのインスタンス作成

    // モーションデータのセットアップ
    func setupMotionManager() {
        if motionManager.isDeviceMotionAvailable {
            motionManager.deviceMotionUpdateInterval = 1
        } else {
            print("motion is not available")
        }
    }

    // 記録の開始・停止を切り替える
    func toggleRecording() {
        if isRecording {
            stopRecording()
        } else {
            startRecording()
        }
    }

    // 記録開始
    func startRecording() {
        motionManager.startDeviceMotionUpdates(to: OperationQueue.main) { (motionData, error) in
            // モーションデータの取得
            if let motion = motionData {
                let timestamp = Date().timeIntervalSince1970 * 1000    // タイムスタンプの取得
                // 加速度データの表示
                let acceleration = motion.userAcceleration
                self.accelerationText = String(format: "X: %.2f, Y: %.2f, Z: %.2f", acceleration.x, acceleration.y, acceleration.z)

                // ジャイロデータの表示
                let gyro = motion.rotationRate
                self.gyroText = String(format: "X: %.2f, Y: %.2f, Z: %.2f", gyro.x, gyro.y, gyro.z)

                // iPhoneにデータを送信
                self.sendDataToiPhone(timestamp: timestamp, acceleration: acceleration, gyro: gyro)
            }
        }
        isRecording = true
    }

    // 記録停止
    func stopRecording() {
        motionManager.stopDeviceMotionUpdates()
        isRecording = false
    }

    // データをiPhoneに送信
    func sendDataToiPhone(timestamp: TimeInterval, acceleration: CMAcceleration, gyro: CMRotationRate) {
        let message = ["timestamp": timestamp,
                       "acceleration": [acceleration.x, acceleration.y, acceleration.z],
                       "gyro": [gyro.x, gyro.y, gyro.z]] as [String : Any]

        if WCSession.default.isReachable {
            WCSession.default.sendMessage(message, replyHandler: nil) { error in
                print("Error sending data to iPhone: \(error)")
            }
        } else {
            print("WCSession is not reachable")
        }
    }

}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
