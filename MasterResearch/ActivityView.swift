//
//  ActivityView.swift
//  MasterResearch
//
//  Created by Kawano Hinase on 2024/01/31.
//

import SwiftUI
import UIKit

struct ActivityView: UIViewControllerRepresentable {
    var activityItems: [Any]
    var applicationActivities: [UIActivity]? = nil

    func makeUIViewController(context: UIViewControllerRepresentableContext<ActivityView>) -> UIActivityViewController {
        let controller = UIActivityViewController(activityItems: activityItems, applicationActivities: applicationActivities)
        return controller
    }

    func updateUIViewController(_ uiViewController: UIActivityViewController, context: UIViewControllerRepresentableContext<ActivityView>) {
        // このメソッド内での更新は必要ありません。
    }
}
