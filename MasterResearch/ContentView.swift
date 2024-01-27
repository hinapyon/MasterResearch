//
//  ContentView.swift
//  MasterResearch Watch App
//
//  Created by Kawano Hinase on 2024/01/09.
//

// ライブラリのインクルード
import SwiftUI
import Foundation
import WatchConnectivity

struct ContentView: View {
    @StateObject private var wcSessionManager = WCSessionManager()
    
    var body: some View {
        NavigationView {
            List {
                ForEach(wcSessionManager.messages, id: \.self) { msg in
                    Text(msg)
                }
            }
            .navigationBarTitle("Received Data", displayMode: .inline)
            .toolbar {
                Button("Export CSV") {
                    wcSessionManager.exportData()
                }
            }
        }
        .onAppear {
            wcSessionManager.startSession()
        }
    }
}

class WCSessionManager: NSObject, ObservableObject, WCSessionDelegate {
    @Published var messages: [String] = []
    
    override init() {
        super.init()
        startSession()
    }
    
    func startSession() {
        if WCSession.isSupported() {
            let session = WCSession.default
            session.delegate = self
            session.activate()
        }
    }
    
    func session(_ session: WCSession, didReceiveMessage message: [String : Any]) {
        DispatchQueue.main.async {
            // 例としてtimestamp, acceleration, gyroを文字列に変換しています
            if let timestamp = message["timestamp"] as? TimeInterval,
               let acceleration = message["acceleration"] as? [Double],
               let gyro = message["gyro"] as? [Double] {
                let msg = "Time: \(timestamp), Acc: \(acceleration), Gyro: \(gyro)"
                self.messages.append(msg)
            }
        }
    }
    
    // CSVファイルとしてデータを出力
    func exportData() {
        let fileName = "MotionData.csv"
        let documentsDirectory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        let path = documentsDirectory.appendingPathComponent(fileName)

        
        var csvText = "Timestamp,AccX,AccY,AccZ,GyroX,GyroY,GyroZ\n"
        
        for msg in messages {
            csvText += "\(msg)\n"
        }
        
        do {
            try csvText.write(to: path, atomically: true, encoding: String.Encoding.utf8)
            print("Saved to \(path)")
        } catch {
            print("Failed to save file: \(error)")
        }
        
        // ここでファイル共有や他のアクションを実行できます
    }
    
    // 必要なデリゲートメソッドのスタブ
    func session(_ session: WCSession, activationDidCompleteWith activationState: WCSessionActivationState, error: Error?) {}
    
    func sessionDidBecomeInactive(_ session: WCSession) {}
    
    func sessionDidDeactivate(_ session: WCSession) {
        WCSession.default.activate()
    }
}
