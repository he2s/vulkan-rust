use anyhow::Result;
use ash::vk;
use super::context::VulkanContext;

#[allow(dead_code)]
pub struct VulkanCommands {
    pool: vk::CommandPool,
    buffers: Vec<vk::CommandBuffer>,
    current_frame: usize,
}

#[allow(dead_code)]
impl VulkanCommands {
    pub unsafe fn new(context: &VulkanContext) -> Result<Self> {
        let pool_info = vk::CommandPoolCreateInfo {
            flags: vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
            queue_family_index: context.queue_family_index,
            ..Default::default()
        };

        let pool = unsafe { context.device.create_command_pool(&pool_info, None)? };

        let alloc_info = vk::CommandBufferAllocateInfo {
            command_pool: pool,
            level: vk::CommandBufferLevel::PRIMARY,
            command_buffer_count: 1,
            ..Default::default()
        };

        let buffers = unsafe { context.device.allocate_command_buffers(&alloc_info)? };

        Ok(Self {
            pool,
            buffers,
            current_frame: 0,
        })
    }

    pub fn get_current_buffer(&self) -> vk::CommandBuffer {
        self.buffers[self.current_frame]
    }

    pub fn next_frame(&mut self) {
        self.current_frame = (self.current_frame + 1) % self.buffers.len();
    }
}