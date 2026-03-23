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
    targets: [
        .target(
            name: "LocalWispr",
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
