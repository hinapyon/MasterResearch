//
//  ContentView.swift
//  MasterResearch Watch App
//
//  Created by Kawano Hinase on 2024/01/09.
//

// ライブラリのインクルード
import SwiftUI  //AppleのUIフレームワーク
import WatchConnectivity    //iPhoneと通信するためのフレームワーク

// WCSessionDelegateを実装するクラス
class SessionManager: NSObject, WCSessionDelegate {
    func session(_ session: WCSession, activationDidCompleteWith activationState: WCSessionActivationState, error: Error?) {
        
    }
    
    func sessionDidBecomeInactive(_ session: WCSession) {
        
    }
    
    func sessionDidDeactivate(_ session: WCSession) {
        
    }
    
    var updateHandler: ((String, String, String) -> Void)?

    override init() {
        super.init()
        if WCSession.isSupported() {
            let session = WCSession.default
            session.delegate = self
            session.activate()
        }
    }

    func session(_ session: WCSession, didReceiveMessage message: [String : Any]) {
        if let timestamp = message["timestamp"] as? TimeInterval,
           let acceleration = message["acceleration"] as? [Double],
           let gyro = message["gyro"] as? [Double] {
            let timestampText = String(format: "%.2f", timestamp)
            let accelerationText = String(format: "X: %.2f, Y: %.2f, Z: %.2f", acceleration[0], acceleration[1], acceleration[2])
            let gyroText = String(format: "X: %.2f, Y: %.2f, Z: %.2f", gyro[0], gyro[1], gyro[2])

            DispatchQueue.main.async {
                self.updateHandler?(timestampText, accelerationText, gyroText)
            }
        }
    }
}

// メインのUI
struct ContentView: View {
    @State private var timestampText = "0.0"
    @State private var accelerationText = "X: 0.0, Y: 0.0, Z: 0.0"
    @State private var gyroText = "X: 0.0, Y: 0.0, Z: 0.0"

    let sessionManager = SessionManager()

    var body: some View {
        VStack {
            Text("Timestamp: \(timestampText)")
                .padding()
            Text("Acceleration")
            Text(accelerationText)
                .padding()
            Text("Gyro")
            Text(gyroText)
                .padding()
        }
        .onAppear() {
            sessionManager.updateHandler = { timestamp, acceleration, gyro in
                self.timestampText = timestamp
                self.accelerationText = acceleration
                self.gyroText = gyro
            }
        }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
