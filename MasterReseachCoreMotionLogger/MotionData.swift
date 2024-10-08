import Foundation

// モーションデータを表す構造体
struct MotionData: Codable {
    let timestamp: TimeInterval
    let accelerationX: Double
    let accelerationY: Double
    let accelerationZ: Double
    let gyroX: Double
    let gyroY: Double
    let gyroZ: Double
    var mark: Bool
}
