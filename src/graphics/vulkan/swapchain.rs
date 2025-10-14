use anyhow::Result;
use ash::{vk, khr::swapchain};
use winit::window::Window;
use super::context::VulkanContext;

#[allow(dead_code)]
pub struct VulkanSwapchain {
    pub loader: swapchain::Device,
    pub swapchain: vk::SwapchainKHR,
    pub extent: vk::Extent2D,
    pub format: vk::Format,
    pub images: Vec<vk::Image>,
    pub views: Vec<vk::ImageView>,
}

#[allow(dead_code)]
impl VulkanSwapchain {
    pub unsafe fn new(context: &VulkanContext, window: &Window, vsync: bool) -> Result<Self> {
        let loader = swapchain::Device::new(&context.instance, &context.device);
        let surface_caps = unsafe { context
            .surface_loader
            .get_physical_device_surface_capabilities(context.physical_device, context.surface)? };
        let formats = unsafe { context
            .surface_loader
            .get_physical_device_surface_formats(context.physical_device, context.surface)? };

        let chosen = Self::choose_surface_format(&formats);
        let format = chosen.format;
        let extent = Self::choose_extent(&surface_caps, window);
        let image_count = Self::choose_image_count(&surface_caps);
        let present_mode = unsafe { Self::choose_present_mode(context, vsync) };

        let create_info = vk::SwapchainCreateInfoKHR {
            surface: context.surface,
            min_image_count: image_count,
            image_format: format,
            image_color_space: chosen.color_space,
            image_extent: extent,
            image_array_layers: 1,
            image_usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
            image_sharing_mode: vk::SharingMode::EXCLUSIVE,
            pre_transform: surface_caps.current_transform,
            composite_alpha: vk::CompositeAlphaFlagsKHR::OPAQUE,
            present_mode,
            clipped: vk::TRUE,
            ..Default::default()
        };

        let swapchain = unsafe { loader.create_swapchain(&create_info, None)? };
        let images = unsafe { loader.get_swapchain_images(swapchain)? };
        let views = unsafe { Self::create_image_views(&context.device, &images, format)? };

        Ok(Self {
            loader,
            swapchain,
            extent,
            format,
            images,
            views,
        })
    }

    fn choose_surface_format(formats: &[vk::SurfaceFormatKHR]) -> vk::SurfaceFormatKHR {
        formats
            .iter()
            .find(|f| {
                f.format == vk::Format::B8G8R8A8_SRGB
                    && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            })
            .copied()
            .unwrap_or_else(|| formats[0])
    }

    unsafe fn choose_present_mode(context: &VulkanContext, vsync: bool) -> vk::PresentModeKHR {
        if vsync {
            return vk::PresentModeKHR::FIFO; // always available
        }
        let modes = unsafe { context
            .surface_loader
            .get_physical_device_surface_present_modes(context.physical_device, context.surface) }
            .unwrap_or_default();
        if modes.contains(&vk::PresentModeKHR::MAILBOX) {
            vk::PresentModeKHR::MAILBOX
        } else if modes.contains(&vk::PresentModeKHR::IMMEDIATE) {
            vk::PresentModeKHR::IMMEDIATE
        } else {
            vk::PresentModeKHR::FIFO
        }
    }

    fn choose_extent(caps: &vk::SurfaceCapabilitiesKHR, window: &Window) -> vk::Extent2D {
        if caps.current_extent.width != u32::MAX {
            // Clamp the current extent to valid bounds in case the window manager provides invalid values
            vk::Extent2D {
                width: caps.current_extent.width
                    .clamp(caps.min_image_extent.width, caps.max_image_extent.width)
                    .max(1),
                height: caps.current_extent.height
                    .clamp(caps.min_image_extent.height, caps.max_image_extent.height)
                    .max(1),
            }
        } else {
            let size = window.inner_size();
            vk::Extent2D {
                width: size.width
                    .clamp(caps.min_image_extent.width, caps.max_image_extent.width)
                    .max(1),
                height: size.height
                    .clamp(caps.min_image_extent.height, caps.max_image_extent.height)
                    .max(1),
            }
        }
    }

    fn choose_image_count(caps: &vk::SurfaceCapabilitiesKHR) -> u32 {
        let desired = caps.min_image_count + 1;
        if caps.max_image_count > 0 && desired > caps.max_image_count {
            caps.max_image_count
        } else {
            desired
        }
    }

    unsafe fn create_image_views(
        device: &ash::Device,
        images: &[vk::Image],
        format: vk::Format,
    ) -> Result<Vec<vk::ImageView>> {
        images
            .iter()
            .map(|&image| {
                let create_info = vk::ImageViewCreateInfo {
                    image,
                    view_type: vk::ImageViewType::TYPE_2D,
                    format,
                    components: vk::ComponentMapping::default(),
                    subresource_range: vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                    ..Default::default()
                };
                unsafe { device.create_image_view(&create_info, None) }
                    .map_err(|e| anyhow::anyhow!("Failed to create image view: {:?}", e))
            })
            .collect()
    }

    pub unsafe fn cleanup(&mut self, device: &ash::Device) {
        for &view in &self.views {
            unsafe { device.destroy_image_view(view, None) };
        }
        unsafe { self.loader.destroy_swapchain(self.swapchain, None) };
    }
}