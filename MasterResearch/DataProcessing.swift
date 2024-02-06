//
//  DataProcessing.swift
//  MasterResearch
//
//  Created by Kawano Hinase on 2024/02/06.
//

import Foundation

extension SessionManager {
    func handleReceivedMessage(_ message: [String: Any]) {
        if let timestamp = message["timestamp"] as? TimeInterval,
           let accelerationX = message["accelerationX"] as? Double,
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
    
    // CSVファイルを出力するメソッド
    func exportDataToCSV(completion: @escaping (URL) -> Void) {
            // 日時をファイル名に含めるためのDateFormatterの設定
            let dateFormatter = DateFormatter()
            dateFormatter.dateFormat = "yyyyMMdd_HHmmss"
            let currentDateTimeString = dateFormatter.string(from: Date())

            // ファイル名に現在の日時を追加
            let fileName = "MotionData_\(currentDateTimeString).csv"
            guard let documentDirectoryPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else {
                print("Document directory not found.")
                return
            }
            let fileURL = documentDirectoryPath.appendingPathComponent(fileName)

            // CSVのヘッダー
            var csvText = "Timestamp,AccelerationX,AccelerationY,AccelerationZ,GyroX,GyroY,GyroZ\n"

            // タイムスタンプをミリ秒まで含むフォーマットで表示
            dateFormatter.dateFormat = "yyyy-MM-dd HH:mm:ss.SSS"

            for dataDict in receivedDataArray {
                let timestamp = dataDict["timestamp"] as? TimeInterval ?? 0
                let date = Date(timeIntervalSince1970: timestamp)
                let formattedDate = dateFormatter.string(from: date) // 日付をフォーマット

                let accelerationX = dataDict["accelerationX"] as? Double ?? 0
                let accelerationY = dataDict["accelerationY"] as? Double ?? 0
                let accelerationZ = dataDict["accelerationZ"] as? Double ?? 0
                let gyroX = dataDict["gyroX"] as? Double ?? 0
                let gyroY = dataDict["gyroY"] as? Double ?? 0
                let gyroZ = dataDict["gyroZ"] as? Double ?? 0

                // CSVの各行のデータ
                let newRow = "\(formattedDate),\(accelerationX),\(accelerationY),\(accelerationZ),\(gyroX),\(gyroY),\(gyroZ)\n"
                csvText += newRow
            }

            // CSVファイルの書き込み
            do {
                try csvText.write(to: fileURL, atomically: true, encoding: .utf8)
                print("CSV file was successfully saved at: \(fileURL)")
                completion(fileURL) // コールバックでファイルURLを返す
            } catch {
                print("Failed to create CSV file: \(error)")
            }
        }
}
