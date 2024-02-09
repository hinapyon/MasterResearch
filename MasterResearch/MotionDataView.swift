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

struct MotionDataGraphView: View {
    var motionDataArray: [MotionData]

    private var baseTimestamp: TimeInterval? {
        motionDataArray.first?.timestamp
    }

    var body: some View {
        Chart {
            if let baseTimestamp = baseTimestamp {
                ForEach(Array(motionDataArray.enumerated()), id: \.element.timestamp) { index, motionData in
                    // X軸のデータ
                    LineMark(
                        x: .value("Time", motionData.timestamp - baseTimestamp),
                        y: .value("Acceleration X", motionData.accelerationX)
                    )
                    .foregroundStyle(.blue) // X軸データの色を指定

                    // Y軸のデータ
                    LineMark(
                        x: .value("Time", motionData.timestamp - baseTimestamp),
                        y: .value("Acceleration Y", motionData.accelerationY)
                    )
                    .foregroundStyle(.green) // Y軸データの色を指定

                    // Z軸のデータ
                    LineMark(
                        x: .value("Time", motionData.timestamp - baseTimestamp),
                        y: .value("Acceleration Z", motionData.accelerationZ)
                    )
                    .foregroundStyle(.red) // Z軸データの色を指定
                }
            }
        }
        .chartXAxis {
            AxisMarks(values: .automatic) // X軸の設定（必要に応じてカスタマイズ）
        }
        .chartYAxis {
            AxisMarks(values: .automatic) // Y軸の設定（必要に応じてカスタマイズ）
        }
    }
}




