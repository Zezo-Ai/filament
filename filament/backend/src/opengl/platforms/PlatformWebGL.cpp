/*
 * Copyright (C) 2018 The Android Open Source Project
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

#include <backend/platforms/PlatformWebGL.h>

namespace filament::backend {

using namespace backend;

Driver* PlatformWebGL::createDriver(void* sharedGLContext,
        const Platform::DriverConfig& driverConfig) noexcept {
    return OpenGLPlatform::createDefaultDriver(this, sharedGLContext, driverConfig);
}

int PlatformWebGL::getOSVersion() const noexcept {
    return 0;
}

void PlatformWebGL::terminate() noexcept {
}

Platform::SwapChain* PlatformWebGL::createSwapChain(
        void* nativeWindow, uint64_t flags) noexcept {
    return (SwapChain*)nativeWindow;
}

Platform::SwapChain* PlatformWebGL::createSwapChain(
        uint32_t width, uint32_t height, uint64_t flags) noexcept {
    // TODO: implement headless SwapChain
    return nullptr;
}

void PlatformWebGL::destroySwapChain(Platform::SwapChain* swapChain) noexcept {
}

bool PlatformWebGL::makeCurrent(ContextType type, SwapChain* drawSwapChain,
        SwapChain* readSwapChain) {
    return true;
}

void PlatformWebGL::commit(Platform::SwapChain* swapChain) noexcept {
}

} // namespace filament::backend
