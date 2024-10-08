#include "fileSystem.h"
#include <filesystem>

namespace FileSystem {
    string GetDirectory(const string &filepath) {
        string directory;
        const size_t last_slash_idx = filepath.rfind('/');
        if (std::string::npos != last_slash_idx) {
            directory = filepath.substr(0, last_slash_idx);
        }
        return directory;
    }

    string GetDDSFilepath(const string &ddsDirectory, const string filename) {
        return std::filesystem::path(ddsDirectory + "/" + filename).replace_extension("dds").string();
    }
};