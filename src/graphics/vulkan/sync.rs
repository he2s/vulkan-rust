use anyhow::Result;
use ash::vk;
use super::context::VulkanContext;

#[allow(dead_code)]
pub struct VulkanSync {
    pub image_available: vk::Semaphore,
    pub render_finished: vk::Semaphore,
    pub in_flight: vk::Fence,
}

#[allow(dead_code)]
impl VulkanSync {
    pub unsafe fn new(context: &VulkanContext) -> Result<Self> {
        let semaphore_info = vk::SemaphoreCreateInfo::default();
        let fence_info = vk::FenceCreateInfo {
            flags: vk::FenceCreateFlags::SIGNALED,
            ..Default::default()
        };

        Ok(Self {
            image_available: unsafe { context.device.create_semaphore(&semaphore_info, None)? },
            render_finished: unsafe { context.device.create_semaphore(&semaphore_info, None)? },
            in_flight: unsafe { context.device.create_fence(&fence_info, None)? },
        })
    }
}