//
//  WatchSessionManager.swift
//  MasterResearch Watch App
//
//  Created by Kawano Hinase on 2024/01/31.
//

import Foundation
import WatchConnectivity

class WatchSessionManager: NSObject, WCSessionDelegate {
    static let shared = WatchSessionManager()

    private var dataBuffer: [[String: Any]] = []
    private let bufferInterval = TimeInterval(1)  // バッファ1秒
    
    override init() {
        super.init()
        if WCSession.isSupported() {
            let session = WCSession.default
            session.delegate = self
            session.activate()
            startBufferTimer()
        }
    }
    
    func sendMessage(_ message: [String: Any]) {
         // メッセージをバッファに追加
         dataBuffer.append(message)
     }
    
    private func startBufferTimer() {
          Timer.scheduledTimer(withTimeInterval: bufferInterval, repeats: true) { [weak self] _ in
              self?.flushDataBuffer()
          }
      }

    private func flushDataBuffer() {
        guard !dataBuffer.isEmpty, WCSession.default.isReachable else {
            return
        }

        // バッファリングされたデータを一括で送信
        for message in dataBuffer {
            WCSession.default.sendMessage(message, replyHandler: nil) { error in
                print("Error sending message: \(error.localizedDescription)")
            }
        }
        dataBuffer.removeAll()
    }

    func session(_ session: WCSession, activationDidCompleteWith activationState: WCSessionActivationState, error: Error?) {
        // セッションのアクティベーションが完了した時の処理
    }

    // その他のWCSessionDelegateメソッドは必要に応じて実装
}
