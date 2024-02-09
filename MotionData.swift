//
//  MotionData.swift
//  MasterResearch
//
//  Created by Kawano Hinase on 2024/02/06.
//

import Foundation

// モーションデータを表す構造体
struct MotionData: Identifiable, Codable {
    let timestamp: TimeInterval
    let accelerationX: Double
    let accelerationY: Double
    let accelerationZ: Double
    let gyroX: Double
    let gyroY: Double
    let gyroZ: Double
}
