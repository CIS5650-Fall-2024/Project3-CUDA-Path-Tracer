#pragma once

#include <string>

using namespace std;

namespace FileSystem {
    string GetDirectory(const string &filepath);

    string GetDDSFilepath(const string& ddsDirectory, const string filename);
};