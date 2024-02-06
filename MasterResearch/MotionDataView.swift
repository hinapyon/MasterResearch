//
//  MotionDataView.swift
//  MasterResearch
//
//  Created by Kawano Hinase on 2024/02/06.
//

import Foundation
import SwiftUI
import Charts

struct MotionDataView: View {
    var receivedDataText: String

    var body: some View {
        Text(receivedDataText)
            .padding()
            .multilineTextAlignment(.leading)
    }
}
