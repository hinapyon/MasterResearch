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

    override init() {
        super.init()
        if WCSession.isSupported() {
            let session = WCSession.default
            session.delegate = self
            session.activate()
        }
    }

    func sendMessage(_ message: [String: Any]) {
        guard WCSession.default.isReachable else {
            print("iPhone is not reachable.")
            return
        }
        WCSession.default.sendMessage(message, replyHandler: nil) { error in
            print("Error sending message: \(error.localizedDescription)")
        }
    }

    func session(_ session: WCSession, activationDidCompleteWith activationState: WCSessionActivationState, error: Error?) {
        // セッションのアクティベーションが完了した時の処理
    }

    // その他のWCSessionDelegateメソッドは必要に応じて実装
}
