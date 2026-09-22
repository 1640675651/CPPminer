#pragma once

#include "opencl_context.hpp"

#include <cstdio>
#include <string>
#include <vector>

namespace onednn_intel {

inline std::string lower_ascii(std::string s) {
    for (char &c : s) {
        if (c >= 'A' && c <= 'Z') {
            c = static_cast<char>(c - 'A' + 'a');
        }
    }
    return s;
}

inline bool is_intel_gpu(const OclDeviceInfo &d) {
    if (d.type != CL_DEVICE_TYPE_GPU) {
        return false;
    }
    const std::string pn = lower_ascii(d.platform_name);
    const std::string dn = lower_ascii(d.device_name);
    const std::string vn = lower_ascii(d.vendor_name);
    return pn.find("intel") != std::string::npos || vn.find("intel") != std::string::npos ||
           dn.find("intel") != std::string::npos || dn.find("uhd") != std::string::npos ||
           dn.find("iris") != std::string::npos || dn.find("hd graphics") != std::string::npos;
}

inline std::vector<OclDeviceInfo> enumerate_intel_gpus(int platform_filter = -1) {
    const std::vector<OclDeviceInfo> all = OpenClContext::enumerate_devices(platform_filter);
    std::vector<OclDeviceInfo> out;
    out.reserve(all.size());
    for (const OclDeviceInfo &d : all) {
        if (is_intel_gpu(d)) {
            out.push_back(d);
        }
    }
    for (size_t i = 0; i < out.size(); ++i) {
        out[i].flat_index = static_cast<int>(i);
    }
    return out;
}

inline int list_intel_gpus(int platform_filter = -1) {
    const std::vector<OclDeviceInfo> devices = enumerate_intel_gpus(platform_filter);
    if (devices.empty()) {
        std::printf("[onednn] no Intel GPU found (Case 5 needs XeLP/Gen12LP or XeHPG)\n");
        return 0;
    }

    std::printf("[onednn] Intel GPU devices (use --devices N; default is first Intel GPU):\n");
    for (const OclDeviceInfo &d : devices) {
        std::printf("  [%d] %s\n", d.flat_index, d.device_name.c_str());
        std::printf("      platform[%d]=%s  integrated GPU%s\n", d.platform_index,
                    d.platform_name.c_str(), d.integer_dot_product ? "  int-dot" : "");
    }
    return static_cast<int>(devices.size());
}

} // namespace onednn_intel
