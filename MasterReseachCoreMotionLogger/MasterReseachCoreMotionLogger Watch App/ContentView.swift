import SwiftUI

struct ContentView: View {
    @ObservedObject private var motionData = MotionDataManager()
    @State private var isRecording = false
    @State private var isShowingDataList = false

    var body: some View {
        NavigationStack {
            VStack {
                if motionData.isDeviceMotionAvailable {
                    Button(action: toggleRecording) {
                        VStack {
                            Image(systemName: isRecording ? "stop.circle" : "play.circle")
                                .resizable()
                                .aspectRatio(contentMode: .fit)
                                .frame(width: 60, height: 60)
                                .foregroundColor(isRecording ? .red : .green)
                            Text(isRecording ? "Stop Recording" : "Start Recording")
                                .foregroundColor(isRecording ? .red : .green)
                                .padding(.top)
                        }
                        .padding()
                    }
                    .padding(.bottom)

                    Button("Check Data") {
                        isShowingDataList = true
                    }
                    .buttonStyle(.bordered)
                    .disabled(isRecording)
                    .foregroundColor(isRecording ? .gray : .blue)
                } else {
                    Text("Device Motion Not Available")
                        .foregroundColor(.red)
                }
            }
            .navigationDestination(isPresented: $isShowingDataList) {
                DataListView()
            }
        }
    }

    func toggleRecording() {
        if isRecording {
            motionData.stopUpdates()
        } else {
            motionData.startUpdates()
        }
        isRecording.toggle()
    }
}


