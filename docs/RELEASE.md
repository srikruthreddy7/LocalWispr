# Releasing a downloadable DMG

The app is distributed as a **compressed disk image** (`.dmg`) built from a **Release** Xcode build. Binaries are **not** committed to git; you attach the DMG to a **GitHub Release** (or another host).

---

## Prerequisites

- **macOS** with **Xcode 26.x** (matches `Package.swift` / project deployment target).
- Apple Developer account if you will **notarize** and staple tickets (recommended for wide distribution outside the Mac App Store).
- Command-line tools: `xcodebuild`, `hdiutil`, `ditto` (all ship with Xcode).

---

## Build the DMG locally

From the repository root:

```bash
chmod +x scripts/build-dmg.sh
./scripts/build-dmg.sh
```

Outputs:

| Path | Description |
|------|-------------|
| `dist/LocalWispr.dmg` | Compressed read-only disk image for drag-to-`Applications` install |
| `dist/LocalWispr.dmg.sha256` | SHA-256 checksum (verify after download) |
| `dist/dmg-root/` | Staging folder (safe to delete after build) |

Optional environment variables:

| Variable | Default | Purpose |
|----------|---------|---------|
| `DERIVED_DATA_PATH` | `/tmp/LocalWisprReleaseDerived` | Xcode derived data for Release build |
| `VOLUME_NAME` | `LocalWispr` | DMG volume label |
| `DMG_BASENAME` | `LocalWispr` | Base filename (`${DMG_BASENAME}.dmg`) |

---

## Publish on GitHub Releases

1. **Tag** a version (example): `git tag v1.0.0 && git push origin v1.0.0`
2. On GitHub: **Releases → Draft a new release**, choose the tag, add release notes.
3. **Attach** `dist/LocalWispr.dmg` as a release asset (drag-and-drop or upload).
4. Paste the contents of `dist/LocalWispr.dmg.sha256` into the release notes so users can verify:

   ```bash
   shasum -a 256 -c LocalWispr.dmg.sha256
   ```

5. In the main **README**, point the **Downloads** section at that release (update the version link when you ship).

---

## CI note (GitHub Actions)

`.github/workflows/release-dmg.yml` can build and upload a **workflow artifact** for testing. **GitHub-hosted runners** may lag behind the **Xcode / macOS** version this project requires—if the workflow fails, build the DMG on a **local Mac** with the correct Xcode and upload the file manually to the release.

---

## Notarization (optional, recommended)

Apple’s documentation covers notarizing command-line tools and apps. Typical flow: **archive** in Xcode, **notarize** with `notarytool`, then export/distribute. Notarization is separate from this repo’s `build-dmg.sh`; add your own signing/notarization steps before wide distribution.

---

## Security reminders

- Prefer **HTTPS** download links only (GitHub Releases uses HTTPS).
- Rotate signing identities if a private key is ever exposed.
- Do not commit **`.p12`**, provisioning profiles, or App Store Connect secrets to the repository.
