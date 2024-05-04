import SwiftUI
import ZIPFoundation

struct FileShareView: View {
    @ObservedObject var sessionManager = SessionManager.shared
    @State private var showDeleteConfirmation = false
    @State private var showingShareSheet = false
    @State private var fileToShare: URL?

    var body: some View {
        VStack {
            List(sessionManager.receivedFiles, id: \.self) { file in
                ShareLink(item: file) {
                    Label("Share \(file.lastPathComponent)", systemImage: "square.and.arrow.up")
                }
            }

            Button("まとめて削除") {
                showDeleteConfirmation = true
            }
            .padding()
            .foregroundColor(.red)
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
        }
    }

    private func deleteAllFiles() {
        let fileManager = FileManager.default
        for file in sessionManager.receivedFiles {
            try? fileManager.removeItem(at: file)
        }
        sessionManager.receivedFiles.removeAll()
    }
}

struct ShareSheet: UIViewControllerRepresentable {
    var activityItems: [Any]

    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: activityItems, applicationActivities: nil)
    }

    func updateUIViewController(_ uiViewController: UIActivityViewController, context: Context) {}
}
