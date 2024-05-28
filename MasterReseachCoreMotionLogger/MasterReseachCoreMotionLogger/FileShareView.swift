import SwiftUI

struct FileShareView: View {
    @ObservedObject var sessionManager = SessionManager.shared
    @State private var showDeleteConfirmation = false
    @State private var showingShareSheet = false
    @State private var fileToShare: URL?
    @State private var tempDirectory: URL?
    @State private var isProcessing = false

    var body: some View {
        VStack {
            List(sessionManager.receivedFiles, id: \.self) { file in
                ShareLink(item: file) {
                    Label("Share \(file.lastPathComponent)", systemImage: "square.and.arrow.up")
                }
            }

            HStack {
                Button("まとめて送信") {
                    createTempDirectoryWithFiles()
                }
                .padding()
                .disabled(isProcessing)

                Button("まとめて削除") {
                    showDeleteConfirmation = true
                }
                .padding()
                .foregroundColor(.red)
            }
            .alert(isPresented: $showDeleteConfirmation) {
                Alert(
                    title: Text("確認"),
                    message: Text("すべてのファイルを削除してもよろしいですか？"),
                    primaryButton: .destructive(Text("削除")) {
                        deleteAllFiles()
                    },
                    secondaryButton: .cancel()
                )
            }

            if isProcessing {
                ProgressView("Processing...")
                    .padding()
            }
        }
        .sheet(isPresented: $showingShareSheet, onDismiss: {
            cleanUpTempDirectory()
        }) {
            if let tempDirectory = tempDirectory {
                ShareSheet(activityItems: [tempDirectory])
            }
        }
    }

    private func deleteAllFiles() {
        let fileManager = FileManager.default
        for file in sessionManager.receivedFiles {
            try? fileManager.removeItem(at: file)
        }
        sessionManager.receivedFiles.removeAll()
    }

    private func createTempDirectoryWithFiles() {
        isProcessing = true
        DispatchQueue.global(qos: .userInitiated).async {
            let fileManager = FileManager.default
            let dateFormatter = DateFormatter()
            dateFormatter.dateFormat = "yyyyMMdd_HHmmss"
            let timestamp = dateFormatter.string(from: Date())
            let tempDirectoryName = "SharedFiles_\(timestamp)"
            let tempDirectoryURL = fileManager.temporaryDirectory.appendingPathComponent(tempDirectoryName)

            do {
                try fileManager.createDirectory(at: tempDirectoryURL, withIntermediateDirectories: true, attributes: nil)
                for file in self.sessionManager.receivedFiles {
                    let destinationURL = tempDirectoryURL.appendingPathComponent(file.lastPathComponent)
                    try fileManager.copyItem(at: file, to: destinationURL)
                }
                DispatchQueue.main.async {
                    self.tempDirectory = tempDirectoryURL
                    self.showingShareSheet = true
                    self.isProcessing = false
                }
            } catch {
                print("Failed to create temp directory or copy files: \(error)")
                DispatchQueue.main.async {
                    self.isProcessing = false
                }
            }
        }
    }

    private func cleanUpTempDirectory() {
        if let tempDirectory = tempDirectory {
            try? FileManager.default.removeItem(at: tempDirectory)
        }
    }
}

struct ShareSheet: UIViewControllerRepresentable {
    var activityItems: [Any]

    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: activityItems, applicationActivities: nil)
    }

    func updateUIViewController(_ uiViewController: UIActivityViewController, context: Context) {}
}
