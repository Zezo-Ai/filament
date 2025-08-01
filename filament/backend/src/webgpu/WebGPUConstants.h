/*
 * Copyright (C) 2025 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef TNT_FILAMENT_BACKEND_WEBGPUCONSTANTS_H
#define TNT_FILAMENT_BACKEND_WEBGPUCONSTANTS_H

#include <utils/Logger.h>

#include <cstdint>

constexpr size_t FILAMENT_WEBGPU_BUFFER_SIZE_MODULUS = 4;

// FWGPU is short for Filament WebGPU

// turn on runtime validation, namely for debugging, that would normally not run (for release)
#define FWGPU_DEBUG_VALIDATION     0x00000001
// print/log system component details to the console, e.g. about the
// instance, surface, adapter, device, etc.
#define FWGPU_PRINT_SYSTEM         0x00000002

// Set this to enable logging "only" to one output stream. This is useful in the case where we want
// to debug with print statements and want ordered logging (e.g LOG(INFO) and LOG(ERROR) will not
// appear in order of calls).
#define FWGPU_DEBUG_FORCE_LOG_TO_I 0x00000004

// Useful default combinations
#define FWGPU_DEBUG_EVERYTHING     0xFFFFFFFF

#if defined(FILAMENT_BACKEND_DEBUG_FLAG)
    #define FWGPU_DEBUG_FORWARDED_FLAG (FILAMENT_BACKEND_DEBUG_FLAG & FWGPU_DEBUG_EVERYTHING)
#else
    #define FWGPU_DEBUG_FORWARDED_FLAG 0
#endif

#ifndef NDEBUG
    #define FWGPU_DEBUG_FLAGS FWGPU_DEBUG_FORWARDED_FLAG
#else
    #define FWGPU_DEBUG_FLAGS 0
#endif

#define FWGPU_ENABLED(flags) (((FWGPU_DEBUG_FLAGS) & (flags)) == (flags))

#if FWGPU_ENABLED(FWGPU_DEBUG_FORCE_LOG_TO_I)
    #define FWGPU_LOGI LOG(INFO)
    #define FWGPU_LOGD FWGPU_LOGI
    #define FWGPU_LOGE FWGPU_LOGI
    #define FWGPU_LOGW FWGPU_LOGI
#else
    #define FWGPU_LOGE LOG(ERROR)
    #define FWGPU_LOGW LOG(WARNING)
    #define FWGPU_LOGD DLOG(INFO)
    #define FWGPU_LOGI LOG(INFO)
#endif

constexpr uint64_t FILAMENT_WEBGPU_REQUEST_ADAPTER_TIMEOUT_NANOSECONDS =
        /* milliseconds */ 1000u * /* converted to ns */ 1000000u;

constexpr uint64_t FILAMENT_WEBGPU_REQUEST_DEVICE_TIMEOUT_NANOSECONDS =
        /* milliseconds */ 1000u * /* converted to ns */ 1000000u;

constexpr uint64_t FILAMENT_WEBGPU_SHADER_COMPILATION_TIMEOUT_NANOSECONDS =
        /* milliseconds */ 1000u * /* converted to ns */ 1000000u;

// if a render pipeline is not used in this number of consecutive frames,
// then expire/release it from the cache.
// A smaller number means more frequent pipeline creation events, taking more time.
// A larger number means more pipelines stored, taking more memory.
constexpr uint64_t FILAMENT_WEBGPU_RENDER_PIPELINE_EXPIRATION_IN_FRAME_COUNT = 45;

// if a pipeline layout is not used in this number of consecutive frames,
// then expire/release it from the cache.
// A smaller number means more frequent pipeline layout creation events, taking more time.
// A larger number means more layouts stored, taking more memory.
constexpr uint64_t FILAMENT_WEBGPU_PIPELINE_LAYOUT_EXPIRATION_IN_FRAME_COUNT = 90;

#endif// TNT_FILAMENT_BACKEND_WEBGPUCONSTANTS_H
