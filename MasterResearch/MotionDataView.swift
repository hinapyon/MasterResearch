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

    private var startTime: TimeInterval {
        motionDataArray.first?.timestamp ?? 0
    }

    var body: some View {
        ScrollView(.vertical) { // 縦方向のScrollViewを追加
            VStack {
                // 加速度データのグラフタイトルとグラフ
                Text("Acceleration Data")
                    .font(.headline)
                    .padding()

                ScrollView(.horizontal) { // 加速度データの横スクロール
                    Chart {
                        ForEach(motionDataArray, id: \.timestamp) { data in
                            LineMark(
                                x: .value("Time", data.timestamp - startTime),
                                y: .value("Acceleration X", data.accelerationX),
                                series: .value("Series", "Acceleration X")
                            )
                            .foregroundStyle(.red)
                            LineMark(
                                x: .value("Time", data.timestamp - startTime),
                                y: .value("Acceleration Y", data.accelerationY),
                                series: .value("Series", "Acceleration Y")
                            )
                            .foregroundStyle(.green)
                            LineMark(
                                x: .value("Time", data.timestamp - startTime),
                                y: .value("Acceleration Z", data.accelerationZ),
                                series: .value("Series", "Acceleration Z")
                            )
                            .foregroundStyle(.blue)
                        }
                    }
                    .chartYScale(domain: -4.0 ... 4.0) // Y軸の範囲を±4で固定
                    .chartYAxis {
                        AxisMarks(preset: .extended, position: .leading)
                    }
                    .frame(width: max(UIScreen.main.bounds.width, CGFloat(motionDataArray.count) * 50), height: 250)
                }

                // 加速度データの凡例
                HStack {
                    Color.red.frame(width: 16, height: 16)
                    Text("X")
                    Color.green.frame(width: 16, height: 16)
                    Text("Y")
                    Color.blue.frame(width: 16, height: 16)
                    Text("Z")
                }.padding()

                // ジャイロスコープデータのグラフタイトルとグラフ
                Text("Gyroscope Data")
                    .font(.headline)
                    .padding()

                ScrollView(.horizontal) { // ジャイロスコープデータの横スクロール
                    Chart {
                        ForEach(motionDataArray, id: \.timestamp) { data in
                            LineMark(
                                x: .value("Time", data.timestamp - startTime),
                                y: .value("Gyro X", data.gyroX),
                                series: .value("Series", "Gyro X")
                            )
                            .foregroundStyle(.red)
                            LineMark(
                                x: .value("Time", data.timestamp - startTime),
                                y: .value("Gyro Y", data.gyroY),
                                series: .value("Series", "Gyro Y")
                            )
                            .foregroundStyle(.green)
                            LineMark(
                                x: .value("Time", data.timestamp - startTime),
                                y: .value("Gyro Z", data.gyroZ),
                                series: .value("Series", "Gyro Z")
                            )
                            .foregroundStyle(.blue)
                        }
                    }
                    .chartYScale(domain: -20.0 ... 20.0) // Y軸の範囲を±20で固定
                    .chartYAxis { // Y軸のカスタマイズ
                        AxisMarks(preset: .extended, position: .leading)
                    }
                    .frame(width: max(UIScreen.main.bounds.width, CGFloat(motionDataArray.count) * 50), height: 250)
                }

                // ジャイロスコープデータの凡例
                HStack {
                    Color.red.frame(width: 16, height: 16)
                    Text("X")
                    Color.green.frame(width: 16, height: 16)
                    Text("Y")
                    Color.blue.frame(width: 16, height: 16)
                    Text("Z")
                }.padding()
            }
        }
    }
}
