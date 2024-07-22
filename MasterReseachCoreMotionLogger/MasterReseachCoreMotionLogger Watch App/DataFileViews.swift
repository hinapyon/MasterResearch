import SwiftUI

struct DataListView: View {
    @State private var motionDataFiles: [String] = []
    @State private var showingActionSheet = false
    @State private var selectedFile: String?
    @State private var showDeleteAllConfirm = false

    var body: some View {
        VStack {
            HStack {
                Button("Delete All") {
                    showDeleteAllConfirm = true
                }
                .alert("Are you sure you want to delete all files?", isPresented: $showDeleteAllConfirm) {
                    Button("Delete", role: .destructive) {
                        deleteAllFiles()
                    }
                    Button("Cancel", role: .cancel) {}
                }
                .foregroundColor(.red)
                .padding()

                Button("Send All") {
                    sendAllFilesToiPhone()
                }
                .foregroundColor(.blue)
                .padding()
            }
            .font(.caption) // 小さなフォントサイズで表示

            List(motionDataFiles, id: \.self) { file in
                Button(file) {
                    selectedFile = file
                    showingActionSheet = true
                }
            }
            .actionSheet(isPresented: $showingActionSheet) {
                ActionSheet(title: Text("Options for \(selectedFile ?? "Unknown")").font(.caption2),
                message: Text("Choose an option").font(.caption2),
                buttons: [
                    .default(Text("Send to iPhone")) {
                        if let file = selectedFile {
                            sendFileToiPhone(file)
                        }
                    },
                    .destructive(Text("Delete")) {
                        if let file = selectedFile {
                            deleteFile(file)
                        }
                    },
                    .cancel()
                ])
            }
        }
        .onAppear {
            loadMotionDataFiles()
        }
    }

    private func loadMotionDataFiles() {
        let fileManager = FileManager.default
        let documentsPath = fileManager.urls(for: .documentDirectory, in: .userDomainMask).first!
        do {
            let fileURLs = try fileManager.contentsOfDirectory(at: documentsPath, includingPropertiesForKeys: nil)
            motionDataFiles = fileURLs.map { $0.lastPathComponent }
        } catch {
            print("Error loading files: \(error)")
        }
    }

    private func deleteFile(_ fileName: String) {
        let fileManager = FileManager.default
        guard let documentsPath = fileManager.urls(for: .documentDirectory, in: .userDomainMask).first else { return }
        let fileURL = documentsPath.appendingPathComponent(fileName)
        do {
            try fileManager.removeItem(at: fileURL)
            loadMotionDataFiles()  // ファイル一覧を再読み込み
        } catch {
            print("Failed to delete file: \(error)")
        }
    }

    private func deleteAllFiles() {
        for fileName in motionDataFiles {
            deleteFile(fileName)
        }
        loadMotionDataFiles()
    }

    private func sendFileToiPhone(_ fileName: String) {
        guard let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else {
            print("Documents directory not found.")
            return
        }
        let fileURL = documentsPath.appendingPathComponent(fileName)
        WatchSessionManager.shared.sendFileToiPhone(fileURL)
    }

    private func sendAllFilesToiPhone() {
        for fileName in motionDataFiles {
            sendFileToiPhone(fileName)
        }
    }
}
