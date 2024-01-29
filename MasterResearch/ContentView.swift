//
//  ContentView.swift
//  MasterResearch Watch App
//
//  Created by Kawano Hinase on 2024/01/09.
//

// ライブラリのインクルード
import UIKit
import SwiftUI  //AppleのUIフレームワーク
import WatchConnectivity    //Apple Watchと通信するためのフレームワーク

class SessionManager: NSObject, ObservableObject, WCSessionDelegate {
    @Published var receivedDataText = "Waiting for data..."
    @Published var showExportConfirmation = false // CSV出力の確認ダイアログ表示フラグ
    static let shared = SessionManager()
    // 受信したデータを保存する配列
    var receivedDataArray: [[String: Any]] = []

    override init() {
        super.init()
        if WCSession.isSupported() {
            let session = WCSession.default
            session.delegate = self
            session.activate()
        }
    }
    
    func sessionDidBecomeInactive(_ session: WCSession) {

    }

    func sessionDidDeactivate(_ session: WCSession) {

    }

    func session(_ session: WCSession, didReceiveMessage message: [String : Any]) {
        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            
            // 受信終了のメッセージをチェック
            if let recording = message["recording"] as? String, recording == "stopped" {
                // CSVファイル出力の確認ダイアログを表示
                self.showExportConfirmation = true
            } else {
                // 通常のデータ受信処理
                self.receivedDataArray.append(message)
                self.handleReceivedMessage(message)
            }
        }
    }
    
    private func handleReceivedMessage(_ message: [String: Any]) {
        if let timestamp = message["timestamp"] as? TimeInterval,let accelerationX = message["accelerationX"] as? Double,
           let accelerationY = message["accelerationY"] as? Double,
           let accelerationZ = message["accelerationZ"] as? Double,
           let gyroX = message["gyroX"] as? Double,
           let gyroY = message["gyroY"] as? Double,
           let gyroZ = message["gyroZ"] as? Double {

            let formattedDate = DateFormatter.localizedString(from: Date(timeIntervalSince1970: timestamp), dateStyle: .medium, timeStyle: .medium)

            // 数値を小数点以下2桁でフォーマット
            let formattedAccelerationX = String(format: "%.2f", accelerationX)
            let formattedAccelerationY = String(format: "%.2f", accelerationY)
            let formattedAccelerationZ = String(format: "%.2f", accelerationZ)
            let formattedGyroX = String(format: "%.2f", gyroX)
            let formattedGyroY = String(format: "%.2f", gyroY)
            let formattedGyroZ = String(format: "%.2f", gyroZ)

            self.receivedDataText = """
            Timestamp: \(formattedDate)

            **Acceleration**
            X: \(formattedAccelerationX), Y: \(formattedAccelerationY), Z: \(formattedAccelerationZ)

            **Gyro**
            X: \(formattedGyroX), Y: \(formattedGyroY), Z: \(formattedGyroZ)
            """
        }
    }

    func session(_ session: WCSession, activationDidCompleteWith activationState: WCSessionActivationState, error: Error?) {
        // Handle session activation...
    }

    // Handle other delegate methods as needed...
}

extension SessionManager {
    // CSVファイルを出力するメソッド
    func exportDataToCSV(completion: @escaping (URL) -> Void) {
        let fileName = "MotionData.csv"
        guard let documentDirectoryPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else {
            print("Document directory not found.")
            return
        }
        let fileURL = documentDirectoryPath.appendingPathComponent(fileName)

        var csvText = "Timestamp,AccelerationX,AccelerationY,AccelerationZ,GyroX,GyroY,GyroZ\n"
        
        for dataDict in receivedDataArray {
            let timestamp = dataDict["timestamp"] as? TimeInterval ?? 0
            let accelerationX = dataDict["accelerationX"] as? Double ?? 0
            let accelerationY = dataDict["accelerationY"] as? Double ?? 0
            let accelerationZ = dataDict["accelerationZ"] as? Double ?? 0
            let gyroX = dataDict["gyroX"] as? Double ?? 0
            let gyroY = dataDict["gyroY"] as? Double ?? 0
            let gyroZ = dataDict["gyroZ"] as? Double ?? 0
            
            let newRow = "\(timestamp),\(accelerationX),\(accelerationY),\(accelerationZ),\(gyroX),\(gyroY),\(gyroZ)\n"
            csvText += newRow
        }
        
        do {
            try csvText.write(to: fileURL, atomically: true, encoding: .utf8)
            print("CSV file was successfully saved at: \(fileURL)")
        } catch {
            print("Failed to create CSV file: \(error)")
        }
        
        do {
                   try csvText.write(to: fileURL, atomically: true, encoding: .utf8)
                   print("CSV file was successfully saved at: \(fileURL)")
                   completion(fileURL) // コールバックでファイルURLを返す
               } catch {
                   print("Failed to create CSV file: \(error)")
               }
    }
}

struct ActivityView: UIViewControllerRepresentable {
    var activityItems: [Any]
    var applicationActivities: [UIActivity]? = nil
    
    func makeUIViewController(context: UIViewControllerRepresentableContext<ActivityView>) -> UIActivityViewController {
        let controller = UIActivityViewController(activityItems: activityItems, applicationActivities: applicationActivities)
        return controller
    }
    
    func updateUIViewController(_ uiViewController: UIActivityViewController, context: UIViewControllerRepresentableContext<ActivityView>) {
    }
}

struct ContentView: View {
    @ObservedObject var sessionManager = SessionManager.shared
    @State private var showShareSheet = false
    @State private var fileURLToShare: URL? = nil

    var body: some View {
        ScrollView {
            Text(sessionManager.receivedDataText)
                .padding()
                .multilineTextAlignment(.leading)
        }
        .alert(isPresented: $sessionManager.showExportConfirmation) {
            Alert(
                title: Text("データの受信が終わりました"),
                message: Text("これまでのデータをCSVファイルとして出力しますか？"),
                primaryButton: .default(Text("はい"), action: {
                    // CSVファイルを出力し、共有シートを表示
                    sessionManager.exportDataToCSV { fileURL in
                        self.fileURLToShare = fileURL
                        self.showShareSheet = true
                    }
                }),
                secondaryButton: .cancel(Text("いいえ"))
            )
        }
        .sheet(isPresented: $showShareSheet, content: {
            if let fileURLToShare = fileURLToShare {
                ActivityView(activityItems: [fileURLToShare], applicationActivities: nil)
            }
        })
    }
}


struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
