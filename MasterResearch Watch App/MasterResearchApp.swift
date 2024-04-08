//
//  MasterResearchApp.swift
//  MasterResearch Watch App
//
//  Created by Kawano Hinase on 2024/01/15.
//

import SwiftUI

@main
struct MasterResearch_Watch_AppApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}

class ExtensionDelegate: NSObject, WKExtensionDelegate {
    func applicationDidFinishLaunching() {
        // アプリが起動したときに画面が消灯しないように設定
        WKExtension.shared().isAutorotating = true
    }
}
