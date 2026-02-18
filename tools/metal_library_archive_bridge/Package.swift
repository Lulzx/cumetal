// swift-tools-version: 5.10
import PackageDescription

let package = Package(
    name: "metal_library_archive_bridge",
    platforms: [
        .macOS(.v13)
    ],
    products: [
        .executable(name: "cumetal-mla-validate", targets: ["cumetal-mla-validate"])
    ],
    dependencies: [
        .package(url: "https://github.com/YuAo/MetalLibraryArchive.git", branch: "master")
    ],
    targets: [
        .executableTarget(
            name: "cumetal-mla-validate",
            dependencies: [
                .product(name: "MetalLibraryArchive", package: "MetalLibraryArchive")
            ]
        )
    ]
)
