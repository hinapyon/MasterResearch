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
                Button(action: toggleRecording) {
                    VStack {
                        Image(systemName: isRecording ? "stop.circle" : "play.circle")
                            .resizable()
                            .aspectRatio(contentMode: .fit)
                            .frame(width: 80, height: 80) // アイコンのサイズを調整
                            .foregroundColor(isRecording ? .red : .green)
                        Text(isRecording ? "Stop Recording" : "Start Recording")
                            .foregroundColor(isRecording ? .red : .green)
                            .padding(.top, 8) // テキストとアイコンの間隔を設定
                    }
                    .padding() // ボタン全体のパディングを調整
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
