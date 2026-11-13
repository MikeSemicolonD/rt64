#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdio>
#include <vector>

#ifdef _WIN32
#   include <io.h>
#else
#   include <unistd.h>
#endif

std::vector<char> read_file(const char* path) {
    std::ifstream input_file{path, std::ios::binary};
    std::vector<char> ret{};

    if (!input_file.good()) {
        return ret;
    }

    // Get the length of the file
    input_file.seekg(0, std::ios::end);
    ret.resize(input_file.tellg());
    
    // Read the file contents into the vector
    input_file.seekg(0, std::ios::beg);
    input_file.read(ret.data(), ret.size());

    return ret;
}

void create_parent_if_needed(const char* path) {
    std::filesystem::path parent_path = std::filesystem::path{path}.parent_path();
    if (!parent_path.empty()) {
        std::filesystem::create_directories(parent_path);
    }
}

// Temp file, flushed to disk, then renamed: a crash or power loss mid-build cannot leave a zero-filled or partial output.
bool write_file_durably(const char* path, const std::string& text) {
    std::string temp_path = std::string{path} + ".tmp";
    FILE* f = fopen(temp_path.c_str(), "wb");

    if (f == nullptr) {
        fprintf(stderr, "Failed to write %s\n", temp_path.c_str());
        return false;
    }

    bool ok = fwrite(text.data(), 1, text.size(), f) == text.size() && fflush(f) == 0;

#ifdef _WIN32
    ok = ok && _commit(_fileno(f)) == 0;
#else
    ok = ok && fsync(fileno(f)) == 0;
#endif

    ok = fclose(f) == 0 && ok;

    std::error_code ec;

    if (ok) {
        std::filesystem::rename(temp_path, path, ec);
    }

    if (!ok || ec) {
        fprintf(stderr, "Failed to write %s\n", path);
        std::filesystem::remove(temp_path, ec);
        return false;
    }

    return true;
}

int main(int argc, const char** argv) {
    if (argc != 5) {
        printf("Usage: %s [input file] [array name] [output C file] [output C header]\n", argv[0]);
        return EXIT_SUCCESS;
    }

    const char* input_path = argv[1];
    const char* array_name = argv[2];
    const char* output_c_path = argv[3];
    const char* output_h_path = argv[4];

    // Read the input file's contents
    std::vector<char> contents = read_file(input_path);

    if (contents.empty()) {
        fprintf(stderr, "Failed to open file %s! (Or it's empty)\n", input_path);
        return EXIT_FAILURE;
    }

    // Create the output directories if they don't exist
    create_parent_if_needed(output_c_path);
    create_parent_if_needed(output_h_path);

    // The header goes first: the .c is the rule's output, so a run cut short before it is rerun by the next build.
    std::ostringstream output_h;
    output_h <<
        "#ifdef __cplusplus\n"
        "  extern \"C\" {\n"
        "#endif\n"
        "#include <stddef.h>\n"
        "extern const char " << array_name << "[" << contents.size() << "];\n"
        "extern const size_t " << array_name << "_size;\n"
        "#ifdef __cplusplus\n"
        "  }\n"
        "#endif\n";

    if (!write_file_durably(output_h_path, output_h.str())) {
        return EXIT_FAILURE;
    }

    std::ostringstream output_c;
    output_c << "#include <stddef.h>\n";
    output_c << "extern const char " << array_name << "[" << contents.size() << "];\n";
    output_c << "const char " << array_name << "[" << contents.size() << "] = {";

    for (char x : contents) {
        output_c << (int)x << ", ";
    }

    output_c << "};\n";
    output_c << "extern const size_t " << array_name << "_size;\n";
    output_c << "const size_t " << array_name << "_size = sizeof(" << array_name << ") / sizeof(" << array_name << "[0]);\n";

    if (!write_file_durably(output_c_path, output_c.str())) {
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
