import Foundation
import WatchConnectivity

class WatchSessionManager: NSObject, WCSessionDelegate {
    static let shared = WatchSessionManager()
    @Published var isSendingData = false
    
    override init() {
        super.init()
        if WCSession.isSupported() {
            let session = WCSession.default
            session.delegate = self
            session.activate()
        }
    }

    func sendFileToiPhone(_ fileURL: URL) {
        let session = WCSession.default
        if session.isReachable {
            session.transferFile(fileURL, metadata: nil)
            isSendingData = true // 送信開始
        }
    }

    // WCSessionDelegate methods
    func session(_ session: WCSession, activationDidCompleteWith activationState: WCSessionActivationState, error: Error?) {
        // Handle activation completion
    }

    func session(_ session: WCSession, didFinish fileTransfer: WCSessionFileTransfer, error: Error?) {
        if fileTransfer.isTransferring == false {
            isSendingData = false // 送信完了
        }
    }
}
