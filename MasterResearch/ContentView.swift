//
//  ContentView.swift
//  MasterResearch Watch App
//
//  Created by Kawano Hinase on 2024/01/09.
//

// ライブラリのインクルード
import SwiftUI
import WatchConnectivity
import Foundation

class SessionManager: NSObject, ObservableObject, WCSessionDelegate {
    func sessionDidBecomeInactive(_ session: WCSession) {
        
    }
    
    func sessionDidDeactivate(_ session: WCSession) {
        
    }
    
    @Published var receivedDataText = "Waiting for data..."
    static let shared = SessionManager()

    override init() {
        super.init()
        if WCSession.isSupported() {
            let session = WCSession.default
            session.delegate = self
            session.activate()
        }
    }
    
    func session(_ session: WCSession, didReceiveMessage message: [String : Any]) {
        DispatchQueue.main.async { [weak self] in
            self?.handleReceivedMessage(message)
        }
    }
    
    private func handleReceivedMessage(_ message: [String: Any]) {
        if let timestamp = message["timestamp"] as? TimeInterval,
           let accelerationX = message["accelerationX"] as? Double,
           let accelerationY = message["accelerationY"] as? Double,
           let accelerationZ = message["accelerationZ"] as? Double,
           let gyroX = message["gyroX"] as? Double,
           let gyroY = message["gyroY"] as? Double,
           let gyroZ = message["gyroZ"] as? Double {
            
            let formattedDate = DateFormatter.localizedString(from: Date(timeIntervalSince1970: timestamp), dateStyle: .short, timeStyle: .long)
            
            self.receivedDataText = """
            Timestamp: \(formattedDate)
            Acceleration - X: \(accelerationX), Y: \(accelerationY), Z: \(accelerationZ)
            Gyro - X: \(gyroX), Y: \(gyroY), Z: \(gyroZ)
            """
        }
    }
    
    func session(_ session: WCSession, activationDidCompleteWith activationState: WCSessionActivationState, error: Error?) {
        // Handle session activation...
    }
    
    // Handle other delegate methods as needed...
}

struct ContentView: View {
    @ObservedObject var sessionManager = SessionManager.shared

    var body: some View {
        ScrollView {
            Text(sessionManager.receivedDataText)
                .padding()
                .multilineTextAlignment(.leading)
        }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
