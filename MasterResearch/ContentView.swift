//
//  ContentView.swift
//  MasterResearch for iPhone
//
//  Created by Kawano Hinase on 2024/01/14.
//

import SwiftUI
import WatchConnectivity

struct ContentView: View {
    @State private var receivedData: [[String: Any]] = []
    @State private var isReceiving = false

    var body: some View {
        VStack {
            Image(systemName: "globe")
                .imageScale(.large)
                .foregroundStyle(.tint)
            Text("Hello, world!")

            Button(action: {
                // ボタンが押されるたびに受信状態を切り替え
                isReceiving.toggle()
                if isReceiving {
                    receiveDataFromWatch()
                }
            }) {
                Text(isReceiving ? "Stop Receiving" : "Start Receiving")
                    .padding()
                    .background(isReceiving ? Color.red : Color.green) // 受信中はボタンを赤く、停止中は緑にする例
                    .foregroundColor(.white)
            }
        }
        .padding()
    }

    func receiveDataFromWatch() {
        guard WCSession.default.isReachable else {
            print("Watch not reachable")
            return
        }

        WCSession.default.sendMessage(["request": isReceiving ? "startRecording" : "stopRecording"], replyHandler: { response in
            if let data = response["data"] as? [[String: Any]] {
                self.receivedData = data
                if !isReceiving {
                    self.saveDataToCSV()
                }
            }
        }, errorHandler: { error in
            print("Error receiving data from Watch: \(error)")
        })
    }

    func saveDataToCSV() {
        guard !receivedData.isEmpty else {
            print("No data to save")
            return
        }

        var csvText = "timestamp,acceleration_x,acceleration_y,acceleration_z,gyro_x,gyro_y,gyro_z\n"

        for entry in receivedData {
            if let timestamp = entry["timestamp"] as? TimeInterval,
               let acceleration = entry["acceleration"] as? [Double],
               let gyro = entry["gyro"] as? [Double] {

                let line = "\(timestamp),\(acceleration[0]),\(acceleration[1]),\(acceleration[2]),\(gyro[0]),\(gyro[1]),\(gyro[2])\n"
                csvText.append(line)
            }
        }

        do {
            let fileURL = try FileManager.default.url(for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: true).appendingPathComponent("data.csv")
            try csvText.write(to: fileURL, atomically: true, encoding: .utf8)
            print("Data saved to \(fileURL)")
        } catch {
            print("Error saving data to CSV: \(error)")
        }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
