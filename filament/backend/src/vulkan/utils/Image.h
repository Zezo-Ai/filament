/*
 * Copyright (C) 2023 The Android Open Source Project
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

#ifndef TNT_FILAMENT_BACKEND_VULKAN_UTILS_IMAGE_H
#define TNT_FILAMENT_BACKEND_VULKAN_UTILS_IMAGE_H

#include <backend/DriverEnums.h>

#include <utils/Log.h>

#include <bluevk/BlueVK.h>

namespace filament::backend {

struct VulkanTexture;

enum class VulkanLayout : uint8_t {
    // The initial layout after the creation of the VkImage. We use this to denote the state before
    // any transition.
    UNDEFINED,
    // Fragment/vertex shader accessible layout for reading and writing.
    STAGING,
    // Fragment shader accessible layout for reading only.
    FRAG_READ,
    // Vertex shader accessible layout for reading only.
    VERT_READ,
    // For the source of a copy operation.
    TRANSFER_SRC,
    // For the destination of a copy operation.
    TRANSFER_DST,
    // For using a depth texture as an attachment.
    DEPTH_ATTACHMENT,
    // For using a depth texture both as an attachment and as a sampler.
    DEPTH_SAMPLER,
    // For swapchain images that will be presented.
    PRESENT,
    // For color attachments, but also used when the image is a sampler.
    // TODO: explore separate layout policies for attachment+sampling and just attachment.
    COLOR_ATTACHMENT,
    // For color attachment MSAA resolves.
    COLOR_ATTACHMENT_RESOLVE,
};

struct VulkanLayoutTransition {
    VkImage image;
    VulkanLayout oldLayout;
    VulkanLayout newLayout;
    VkImageSubresourceRange subresources;
};

namespace fvkutils {

constexpr inline VkImageLayout getVkLayout(VulkanLayout layout) {
    switch (layout) {
        case VulkanLayout::UNDEFINED:
            return VK_IMAGE_LAYOUT_UNDEFINED;
        case VulkanLayout::STAGING:
            return VK_IMAGE_LAYOUT_GENERAL;
        case VulkanLayout::FRAG_READ:
        case VulkanLayout::VERT_READ:
            return VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        case VulkanLayout::TRANSFER_SRC:
            return VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        case VulkanLayout::TRANSFER_DST:
            return VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        case VulkanLayout::DEPTH_ATTACHMENT:
            return VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        case VulkanLayout::DEPTH_SAMPLER:
            return VK_IMAGE_LAYOUT_GENERAL;
        case VulkanLayout::PRESENT:
            return VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        // Filament sometimes samples from one miplevel while writing to another level in the
        // same texture (e.g. bloom does this). So, keep it simple and use GENERAL for all
        // color-attachable textures.
        case VulkanLayout::COLOR_ATTACHMENT:
            return VK_IMAGE_LAYOUT_GENERAL;
        case VulkanLayout::COLOR_ATTACHMENT_RESOLVE:
            return VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    }
}

// Returns true if a transition has been added to the command buffer, false otherwis (where there is
// no transition necessary).
bool transitionLayout(VkCommandBuffer cmdbuffer, VulkanLayoutTransition transition);

bool isVkDepthFormat(VkFormat format);

bool isVkStencilFormat(VkFormat format);

bool isVKYcbcrConversionFormat(VkFormat format);

VkImageAspectFlags getImageAspect(VkFormat format);

uint8_t reduceSampleCount(uint8_t sampleCount, VkSampleCountFlags mask);

} // namespace fvkutils

} // namespace filament::backend

bool operator<(const VkImageSubresourceRange& a, const VkImageSubresourceRange& b);

namespace utils::io {
utils::io::ostream& operator<<(utils::io::ostream& out, const filament::backend::VulkanLayout& layout);
}

#endif // TNT_FILAMENT_BACKEND_VULKAN_UTILS_IMAGE_H
