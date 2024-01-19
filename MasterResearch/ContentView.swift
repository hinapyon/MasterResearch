//
//  ContentView.swift
//  MasterResearch Watch App
//
//  Created by Kawano Hinase on 2024/01/09.
//

// ライブラリのインクルード
import SwiftUI  //AppleのUIフレームワーク
import WatchConnectivity    //iPhoneと通信するためのフレームワーク

//メインのUI
struct ContentView: View {
    @ObservedObject private var connector = WatchConnector()
    @State private var timestampText = "0.0"  // タイムスタンプの表示テキスト
    @State private var accelerationText = "X: 0.0, Y: 0.0, Z: 0.0"  // 加速度データの表示テキスト
    @State private var gyroText = "X: 0.0, Y: 0.0, Z: 0.0"  // ジャイロデータの表示テキスト

    // 画面表示構成
    var body: some View {
        VStack {
            Text("Acceleration")
            Text(accelerationText)
                .padding()
            Text("Gyro")
            Text(gyroText)
                .padding()

        }
        .onAppear() {   // Viewが表示されるタイミングで一度だけ実行される
        }
    }
}

class WatchConnector: NSObject, ObservableObject, WCSessionDelegate {
    func sessionDidBecomeInactive(_ session: WCSession) {
    }

    func sessionDidDeactivate(_ session: WCSession) {

    }

    // WatchConnectivityセッション
    private var session: WCSession?

    // 受信データのプロパティ
    @Published var timestampText = "0.0"
    @Published var accelerationText = "X: 0.0, Y: 0.0, Z: 0.0"
    @Published var gyroText = "X: 0.0, Y: 0.0, Z: 0.0"

    override init() {
        super.init()

        // WCSessionのセットアップ
        if WCSession.isSupported() {
            session = WCSession.default
            session?.delegate = self
            session?.activate()
        }
    }

    // WCSessionがアクティベートされたときに呼ばれる
    func session(_ session: WCSession, activationDidCompleteWith activationState: WCSessionActivationState, error: Error?) {
        // アクティベートが完了したら何か処理を追加する場合に使用
    }

    // データが受信されたときに呼ばれる
    func session(_ session: WCSession, didReceiveMessage message: [String : Any]) {
        DispatchQueue.main.async {
            // 受信したデータを更新
            if let timestamp = message["timestamp"] as? TimeInterval {
                self.timestampText = String(timestamp)
            }

            if let acceleration = message["acceleration"] as? [Double],
            acceleration.count == 3 {
                self.accelerationText = String(format: "X: %.2f, Y: %.2f, Z: %.2f", acceleration[0], acceleration[1], acceleration[2])
            }

            if let gyro = message["gyro"] as? [Double],
            gyro.count == 3 {
                self.gyroText = String(format: "X: %.2f, Y: %.2f, Z: %.2f", gyro[0], gyro[1], gyro[2])
            }
        }
    }
}


struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
