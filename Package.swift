// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "LocalWispr",
    platforms: [
        .macOS(.v26)
    ],
    products: [
        .library(
            name: "LocalWispr",
            targets: ["LocalWispr"]
        )
    ],
    dependencies: [
        .package(url: "https://github.com/altic-dev/FluidAudio.git", branch: "B/cohere-coreml-asr")
    ],
    targets: [
        .target(
            name: "LocalWispr",
            dependencies: [
                "FluidAudio"
            ],
            resources: [
                .process("Resources")
            ]
        ),
        .testTarget(
            name: "LocalWisprTests",
            dependencies: ["LocalWispr"]
        )
    ]
)
