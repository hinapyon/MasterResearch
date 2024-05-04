import SwiftUI

struct ContentView: View {
    @ObservedObject var sessionManager = SessionManager.shared

    var body: some View {
        VStack {
            if sessionManager.receivedFiles.isEmpty {
                Text("Apple Watchからデータを送信してください")
                    .padding()
            } else {
                FileShareView()
            }

            if sessionManager.isReceiving {
                Text("データを受信しています")
                    .foregroundColor(.blue)
                    .padding()
            }

            if sessionManager.isConverting {
                Text("CSVファイルに変換しています")
                    .foregroundColor(.green)
                    .padding()
            }
        }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
