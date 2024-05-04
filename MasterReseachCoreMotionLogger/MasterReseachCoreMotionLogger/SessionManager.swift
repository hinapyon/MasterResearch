import Foundation
import WatchConnectivity
import SwiftUI

import Foundation
import WatchConnectivity
import SwiftUI

class SessionManager: NSObject, ObservableObject, WCSessionDelegate {
    static let shared = SessionManager()
    @Published var receivedFiles: [URL] = []
    @Published var isReceiving = false
    @Published var isConverting = false

    override init() {
        super.init()
        if WCSession.isSupported() {
            let session = WCSession.default
            session.delegate = self
            session.activate()
        }
    }

    func convertJSONToCSV(fileURL: URL) {
        do {
            let jsonData = try Data(contentsOf: fileURL)
            let jsonArray = try JSONSerialization.jsonObject(with: jsonData, options: .allowFragments) as? [[String: Any]]

            var csvString = "Timestamp,AccelerationX,AccelerationY,AccelerationZ,GyroX,GyroY,GyroZ\n"
            for json in jsonArray ?? [] {
                let timestamp = json["timestamp"] as? Double ?? 0
                let accX = json["accelerationX"] as? Double ?? 0
                let accY = json["accelerationY"] as? Double ?? 0
                let accZ = json["accelerationZ"] as? Double ?? 0
                let gyroX = json["gyroX"] as? Double ?? 0
                let gyroY = json["gyroY"] as? Double ?? 0
                let gyroZ = json["gyroZ"] as? Double ?? 0

                let csvLine = "\(timestamp),\(accX),\(accY),\(accZ),\(gyroX),\(gyroY),\(gyroZ)\n"
                csvString += csvLine
            }

            if let documentDirectory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first {
                // JSONファイル名からCSVファイル名を生成
                let jsonFileName = fileURL.deletingPathExtension().lastPathComponent
                let csvFilename = "\(jsonFileName).csv"
                let csvFileURL = documentDirectory.appendingPathComponent(csvFilename)
                
                try csvString.write(to: csvFileURL, atomically: true, encoding: .utf8)
                DispatchQueue.main.async {
                    self.receivedFiles.append(csvFileURL)  // CSVファイルのURLを保存
                    self.isConverting = false
                    self.cleanup(jsonFileURL: fileURL) // JSONファイルの削除
                }
            }
        } catch {
            print("Error during JSON to CSV conversion: \(error)")
            DispatchQueue.main.async {
                self.isConverting = false
            }
        }
    }

    func cleanup(jsonFileURL: URL) {
        do {
            try FileManager.default.removeItem(at: jsonFileURL)
            DispatchQueue.main.async {
                self.receivedFiles.removeAll { $0 == jsonFileURL }
            }
        } catch {
            print("Failed to delete JSON file: \(error)")
        }
    }

    func session(_ session: WCSession, didReceive file: WCSessionFile) {
        DispatchQueue.main.async {
            self.isReceiving = true
        }

        let fileManager = FileManager.default
        let destinationURL = fileManager.urls(for: .documentDirectory, in: .userDomainMask).first!.appendingPathComponent(file.fileURL.lastPathComponent)
        do {
            if fileManager.fileExists(atPath: destinationURL.path) {
                try fileManager.removeItem(at: destinationURL)
            }
            try fileManager.copyItem(at: file.fileURL, to: destinationURL)
            DispatchQueue.main.async {
                self.isReceiving = false
                self.isConverting = true
                self.convertJSONToCSV(fileURL: destinationURL)  // JSONからCSVへの変換を開始
            }
        } catch {
            print("Error saving file: \(error)")
            DispatchQueue.main.async {
                self.isReceiving = false
            }
        }
    }

    func sessionDidBecomeInactive(_ session: WCSession) {}
    func sessionDidDeactivate(_ session: WCSession) {}
    func session(_ session: WCSession, activationDidCompleteWith activationState: WCSessionActivationState, error: Error?) {}
}
