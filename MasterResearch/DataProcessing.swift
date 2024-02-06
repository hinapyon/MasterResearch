//
//  DataProcessing.swift
//  MasterResearch
//
//  Created by Kawano Hinase on 2024/02/06.
//

import Foundation

extension SessionManager {
    func handleReceivedMessage(_ motionData: MotionData) {
        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "yyyy-MM-dd HH:mm:ss.SSS"

        let formattedDate = dateFormatter.string(from: Date(timeIntervalSince1970: motionData.timestamp))

        // データテキストの生成
        receivedDataText = """
            Timestamp: \(formattedDate)

            **Acceleration**
            X: \(String(format: "%.2f", motionData.accelerationX)), Y: \(String(format: "%.2f", motionData.accelerationY)), Z: \(String(format: "%.2f", motionData.accelerationZ))

            **Gyro**
            X: \(String(format: "%.2f", motionData.gyroX)), Y: \(String(format: "%.2f", motionData.gyroY)), Z: \(String(format: "%.2f", motionData.gyroZ))
            """
    }
        // CSVファイルを出力するメソッド
    func exportDataToCSV(completion: @escaping (URL) -> Void) {
        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "yyyyMMdd_HHmmss"
        let currentDateTimeString = dateFormatter.string(from: Date())
        let fileName = "MotionData_\(currentDateTimeString).csv"

        guard let documentDirectoryPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else {
            print("Document directory not found.")
            return
        }

        let fileURL = documentDirectoryPath.appendingPathComponent(fileName)

        // CSVファイルのヘッダー
        var csvText = "Timestamp,AccelerationX,AccelerationY,AccelerationZ,GyroX,GyroY,GyroZ\n"

        dateFormatter.dateFormat = "yyyy-MM-dd HH:mm:ss.SSS"

        for motionData in receivedMotionDataArray {
            let formattedDate = dateFormatter.string(from: Date(timeIntervalSince1970: motionData.timestamp))

            // CSVファイルの各行を構成
            let newRow = "\(formattedDate),\(motionData.accelerationX),\(motionData.accelerationY),\(motionData.accelerationZ),\(motionData.gyroX),\(motionData.gyroY),\(motionData.gyroZ)\n"
            csvText += newRow
        }

        // CSVファイルに文字列を書き込む
        do {
            try csvText.write(to: fileURL, atomically: true, encoding: .utf8)
            print("CSV file was successfully saved at: \(fileURL)")
            completion(fileURL) // ファイルの保存が成功したら、ファイルのURLをコールバックで返す
        } catch {
            print("Failed to create CSV file: \(error)")
        }
    }
    }
