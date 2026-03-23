import AppKit
import CoreText
import SwiftUI

@MainActor
public enum AppTheme {
    public enum SansWeight {
        case regular
        case medium
        case semibold
        case bold
    }

    public enum MonoWeight {
        case regular
        case medium
        case semibold
        case bold
    }

    private static var didBootstrap = false

    public static func bootstrap() {
        guard !didBootstrap else { return }
        didBootstrap = true
        registerBundledFonts()
    }

    public static func sans(_ size: CGFloat, weight: SansWeight = .regular) -> Font {
        Font.custom(sansFontName(for: weight), size: size)
    }

    public static func mono(_ size: CGFloat, weight: MonoWeight = .regular) -> Font {
        Font.custom(monoFontName(for: weight), size: size)
    }

    public static var accent: Color {
        Color(red: 0.97, green: 0.98, blue: 1.00)
    }

    public static var accentSoft: Color {
        Color.white.opacity(0.72)
    }

    public static var backgroundTop: Color {
        Color(red: 0.015, green: 0.016, blue: 0.022)
    }

    public static var backgroundBottom: Color {
        Color(red: 0.035, green: 0.038, blue: 0.050)
    }

    public static var panelFill: Color {
        Color.white.opacity(0.035)
    }

    public static var panelStroke: Color {
        Color.white.opacity(0.12)
    }

    public static var softStroke: Color {
        Color.white.opacity(0.07)
    }

    public static var secondaryText: Color {
        Color.white.opacity(0.66)
    }

    public static var tertiaryText: Color {
        Color.white.opacity(0.46)
    }

    public static var danger: Color {
        Color(red: 1.0, green: 0.34, blue: 0.34)
    }

    public static var warning: Color {
        Color(red: 1.0, green: 0.68, blue: 0.28)
    }

    public static var success: Color {
        Color(red: 0.37, green: 0.92, blue: 0.58)
    }

    private static func registerBundledFonts() {
        guard let fontURLs = Bundle.module.urls(forResourcesWithExtension: "ttf", subdirectory: "Fonts") else {
            return
        }

        for fontURL in fontURLs {
            CTFontManagerRegisterFontsForURL(fontURL as CFURL, .process, nil)
        }
    }

    private static func sansFontName(for weight: SansWeight) -> String {
        switch weight {
        case .regular:
            return "Geist-Regular"
        case .medium:
            return "Geist-Medium"
        case .semibold:
            return "Geist-SemiBold"
        case .bold:
            return "Geist-Bold"
        }
    }

    private static func monoFontName(for weight: MonoWeight) -> String {
        switch weight {
        case .regular:
            return "GeistMono-Regular"
        case .medium:
            return "GeistMono-Medium"
        case .semibold:
            return "GeistMono-SemiBold"
        case .bold:
            return "GeistMono-Bold"
        }
    }
}
