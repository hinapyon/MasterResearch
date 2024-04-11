//
//  ContentView.swift
//  MasterResearch Watch App
//
//  Created by Kawano Hinase on 2024/01/09.
//

// ライブラリのインクルード
import SwiftUI

struct ContentView: View {
    @ObservedObject var sessionManager = SessionManager.shared
    @State private var showShareSheet = false
    @State private var fileURLToShare: URL? = nil

    var body: some View {
        VStack {
            Spacer()
            
            if sessionManager.isReceivingData {
                VStack {
                    ProgressView("受信中...")
                        .progressViewStyle(CircularProgressViewStyle())
                        .scaleEffect(1.5)
                        .padding()
                    Text("データを受信しています。完了するまでお待ちください。")
                }
            } else {
                Text("受信を開始するには、Apple Watchのアプリで再生ボタンを押してください。")
            }
            
            Spacer()
            
//            // MotionDataViewを使用して受信したデータを表示
//            MotionDataView(receivedDataText: sessionManager.receivedDataText)
//                .padding()
//
//            // MotionDataGraphViewを使用して受信したデータを表示
//            MotionDataGraphView(motionDataArray: sessionManager.receivedMotionDataArray)
//                .padding()

            // その他のUIコンポーネントがある場合はここに追加
        }
        .alert(isPresented: $sessionManager.showExportConfirmation) {
            Alert(
                title: Text("データの受信が終わりました"),
                message: Text("これまでのデータをCSVファイルとして出力しますか？"),
                primaryButton: .default(Text("はい"), action: {
                    // CSVファイルを出力し、共有シートを表示
                    sessionManager.exportDataToCSV { fileURL in
                        self.fileURLToShare = fileURL
                        self.showShareSheet = true
                    }
                }),
                secondaryButton: .cancel(Text("いいえ"))
            )
        }
        .sheet(isPresented: $showShareSheet, content: {
            if let fileURLToShare = fileURLToShare {
                ActivityView(activityItems: [fileURLToShare], applicationActivities: nil)
            }
        })
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
