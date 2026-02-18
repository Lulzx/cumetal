import Foundation
import MetalLibraryArchive

func printUsage() {
    fputs("Usage: cumetal-mla-validate <path-to-metallib>\n", stderr)
}

func scanU16LE(_ data: Data, _ index: Int) -> UInt16 {
    data.withUnsafeBytes { ptr in
        ptr.bindMemory(to: UInt16.self)[index].littleEndian
    }
}

func scanU64LE(_ data: Data, _ index: Int) -> UInt64 {
    data.withUnsafeBytes { ptr in
        ptr.bindMemory(to: UInt64.self)[index].littleEndian
    }
}

func parseVersionTag(_ tags: [Tag]) -> String? {
    guard let tag = tags.first(where: { $0.name == "VERS" }), tag.content.count >= 8 else {
        return nil
    }
    let airMajor = scanU16LE(tag.content, 0)
    let airMinor = scanU16LE(tag.content, 1)
    return "\(airMajor).\(airMinor)"
}

func parseOffsetsTag(_ tags: [Tag]) -> (UInt64, UInt64, UInt64)? {
    guard let tag = tags.first(where: { $0.name == "OFFT" }), tag.content.count >= 24 else {
        return nil
    }
    let publicOffset = scanU64LE(tag.content, 0)
    let privateOffset = scanU64LE(tag.content, 1)
    let bitcodeOffset = scanU64LE(tag.content, 2)
    return (publicOffset, privateOffset, bitcodeOffset)
}

do {
    guard CommandLine.arguments.count == 2 else {
        printUsage()
        exit(2)
    }

    let path = CommandLine.arguments[1]
    let url = URL(fileURLWithPath: path)
    let data = try Data(contentsOf: url)

    let archive = try Archive(data: data)
    print("OK: parsed metallib")
    print("archive_version=\(archive.version.major).\(archive.version.minor)")
    print("functions=\(archive.functions.count)")

    for function in archive.functions {
        print("name=\(function.name)")
        print("type=\(String(describing: function.type))")
        print("language_version=\(function.languageVersion.major).\(function.languageVersion.minor)")
        if let airVersion = parseVersionTag(function.tags) {
            print("air_version=\(airVersion)")
        }
        if let offsets = parseOffsetsTag(function.tags) {
            print("public_metadata_offset=\(offsets.0)")
            print("private_metadata_offset=\(offsets.1)")
            print("bitcode_offset=\(offsets.2)")
        }
        print("bitcode_size=\(function.bitcode.count)")
        print("public_metadata_tags=\(function.publicMetadataTags.count)")
        print("private_metadata_tags=\(function.privateMetadataTags.count)")
    }
} catch {
    fputs("ERROR: \(error)\n", stderr)
    exit(1)
}
