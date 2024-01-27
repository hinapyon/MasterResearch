//
//  ContentView.swift
//  MasterResearch Watch App
//
//  Created by Kawano Hinase on 2024/01/09.
//

// ライブラリのインクルード
import SwiftUI
import WatchConnectivity

// SessionManagerクラス
class SessionManager: NSObject, WCSessionDelegate {
    func sessionDidBecomeInactive(_ session: WCSession) {
        
    }
    
    func sessionDidDeactivate(_ session: WCSession) {
        
    }
    
    static let shared = SessionManager()
    var receivedDataHandler: (([String: Any]) -> Void)?

    override init() {
        super.init()
        if WCSession.isSupported() {
            let session = WCSession.default
            session.delegate = self
            session.activate()
        }
    }

    func session(_ session: WCSession, didReceiveMessage message: [String : Any]) {
        // Watchからのメッセージを受信した際の処理
        DispatchQueue.main.async { [weak self] in
            self?.receivedDataHandler?(message)
        }
    }

    // その他のWCSessionDelegateメソッドは、必要に応じて実装
    func session(_ session: WCSession, activationDidCompleteWith activationState: WCSessionActivationState, error: Error?) {
        // Activationが完了した時の処理
    }
}

// モーションデータを表示するためのViewModel
class MotionDataViewModel: ObservableObject {
    @Published var accelerationText: String = "X: 0.0, Y: 0.0, Z: 0.0"
    @Published var gyroText: String = "X: 0.0, Y: 0.0, Z: 0.0"

    init() {
        SessionManager.shared.receivedDataHandler = { [weak self] data in
            if let accelerationX = data["accelerationX"] as? Double,
               let accelerationY = data["accelerationY"] as? Double,
               let accelerationZ = data["accelerationZ"] as? Double,
               let gyroX = data["gyroX"] as? Double,
               let gyroY = data["gyroY"] as? Double,
               let gyroZ = data["gyroZ"] as? Double {
                self?.accelerationText = String(format: "X: %.2f, Y: %.2f, Z: %.2f", accelerationX, accelerationY, accelerationZ)
                self?.gyroText = String(format: "X: %.2f, Y: %.2f, Z: %.2f", gyroX, gyroY, gyroZ)
            }
        }
    }
}

// ContentView
struct ContentView: View {
    @ObservedObject var viewModel = MotionDataViewModel()

    var body: some View {
        VStack {
            Text("Acceleration")
                .font(.headline)
            Text(viewModel.accelerationText)
                .padding()
            Text("Gyro")
                .font(.headline)
            Text(viewModel.gyroText)
                .padding()
        }
        .onAppear {
            // 必要な初期化処理があればここに記述
        }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
