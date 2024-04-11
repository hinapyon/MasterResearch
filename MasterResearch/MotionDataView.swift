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

    private var downsampledData: [MotionData] {
        downsample(motionDataArray, to: 100) // 例: 100ポイントにダウンサンプリング
    }

    private var startTime: TimeInterval {
        downsampledData.first?.timestamp ?? 0
    }

    var body: some View {
        ScrollView(.vertical) {
            VStack {
                graphTitle("Acceleration Data")
                scrollViewForGraph {
                    lineChart(for: downsampledData, valueKeyPaths: (\MotionData.accelerationX, \MotionData.accelerationY, \MotionData.accelerationZ), colors: [.red, .green, .blue])
                }
                legend(["X", "Y", "Z"], colors: [.red, .green, .blue])

                graphTitle("Gyroscope Data")
                scrollViewForGraph {
                    lineChart(for: downsampledData, valueKeyPaths: (\MotionData.gyroX, \MotionData.gyroY, \MotionData.gyroZ), colors: [.red, .green, .blue])
                }
                legend(["X", "Y", "Z"], colors: [.red, .green, .blue])
            }
        }
    }

    private func graphTitle(_ title: String) -> some View {
        Text(title)
            .font(.headline)
            .padding()
    }

    private func scrollViewForGraph<Content: View>(@ViewBuilder content: () -> Content) -> some View {
        ScrollView(.horizontal) {
            content()
                .frame(width: max(UIScreen.main.bounds.width, CGFloat(downsampledData.count) * 50), height: 250)
        }
    }

    private func lineChart(for data: [MotionData], valueKeyPaths: (KeyPath<MotionData, Double>, KeyPath<MotionData, Double>, KeyPath<MotionData, Double>), colors: [Color]) -> some View {
        Chart {
            ForEach(data, id: \.timestamp) { dataPoint in
                LineMark(
                    x: .value("Time", dataPoint.timestamp - startTime),
                    y: .value("Value 1", dataPoint[keyPath: valueKeyPaths.0]),
                    series: .value("Series 1", "Value 1")
                )
                .foregroundStyle(colors[0])

                LineMark(
                    x: .value("Time", dataPoint.timestamp - startTime),
                    y: .value("Value 2", dataPoint[keyPath: valueKeyPaths.1]),
                    series: .value("Series 2", "Value 2")
                )
                .foregroundStyle(colors[1])

                LineMark(
                    x: .value("Time", dataPoint.timestamp - startTime),
                    y: .value("Value 3", dataPoint[keyPath: valueKeyPaths.2]),
                    series: .value("Series 3", "Value 3")
                )
                .foregroundStyle(colors[2])
            }
        }
        .chartYScale(domain: -4.0 ... 4.0)
        .chartYAxis {
            AxisMarks(preset: .extended, position: .leading)
        }
    }

    private func legend(_ labels: [String], colors: [Color]) -> some View {
        HStack {
            ForEach(Array(labels.enumerated()), id: \.offset) { index, label in
                HStack {
                    colors[index].frame(width: 16, height: 16)
                    Text(label)
                }
            }
        }
        .padding()
    }

    private func downsample(_ data: [MotionData], to count: Int) -> [MotionData] {
        // データのダウンサンプリングを行うロジック
        let step = max(data.count / count, 1)
        return stride(from: 0, to: data.count, by: step).map { data[$0] }
    }
}
