//
//  ContentView.swift
//  MasterResearch Watch App
//
//  Created by Kawano Hinase on 2024/01/09.
//

// ライブラリのインクルード
import SwiftUI  //AppleのUIフレームワーク

struct ContentView: View {
    @ObservedObject private var motionData = MotionDataManager()
    @State private var isRecording = false

    var body: some View {
        VStack {
            if motionData.isDeviceMotionAvailable {
                Text("Acceleration")
                Text("X: \(String(format: "%.2f", motionData.acceleration.x)), Y: \(String(format: "%.2f", motionData.acceleration.y)), Z: \(String(format: "%.2f", motionData.acceleration.z))")
                    .padding()
                Text("Gyro")
                Text("X: \(String(format: "%.2f", motionData.gyro.x)), Y: \(String(format: "%.2f", motionData.gyro.y)), Z: \(String(format: "%.2f", motionData.gyro.z))")
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
            self.isRecording = false
        }
    }

    func toggleRecording() {
        if isRecording {
            motionData.stopUpdates()
            // ここでiPhoneに受信終了のメッセージを送信します。
            let message = ["recording": "stopped"]
            WatchSessionManager.shared.sendMessage(message)
        } else {
            motionData.startUpdates()
            // ここでiPhoneに送信開始のメッセージを送信します。
            let message = ["recording": "started"]
            WatchSessionManager.shared.sendMessage(message)
        }
        isRecording.toggle()
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
