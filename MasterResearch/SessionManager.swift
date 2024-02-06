//
//  SessionManager.swift
//  MasterResearch
//
//  Created by Kawano Hinase on 2024/01/31.
//

import Foundation
import WatchConnectivity
import SwiftUI

class SessionManager: NSObject, ObservableObject, WCSessionDelegate {
    @Published var receivedDataText = "Waiting for data..."
    @Published var showExportConfirmation = false // CSV出力の確認ダイアログ表示フラグ
    static let shared = SessionManager()
    // 受信したデータを保存する配列
    var receivedMotionDataArray: [MotionData] = []


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

            // "Start Recording"メッセージをチェック
            if let recording = message["recording"] as? String, recording == "started" {
                // 配列をリセット
                self.receivedMotionDataArray.removeAll()
            }

            // 受信終了のメッセージをチェック
            else if let recording = message["recording"] as? String, recording == "stopped" {
                // CSVファイル出力の確認ダイアログを表示
                self.showExportConfirmation = true
            } else {
                // JSONメッセージをMotionDataにデコード
                guard let jsonData = try? JSONSerialization.data(withJSONObject: message),
                    let motionData = try? JSONDecoder().decode(MotionData.self, from: jsonData) else {
                    print("Error decoding MotionData")
                    return
                }
                    // 通常のデータ受信処理
                    self.receivedMotionDataArray.append(motionData)
                // 受信データを処理するためにhandleReceivedMessageを呼び出す
                    self.handleReceivedMessage(motionData)
                }
            }
        }

        func session(_ session: WCSession, activationDidCompleteWith activationState: WCSessionActivationState, error: Error?) {
            // Handle session activation...
        }

    }
