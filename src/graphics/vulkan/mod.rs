pub mod context;
pub mod swapchain;
pub mod pipeline;
pub mod commands;
pub mod sync;

// Vulkan module exports - currently most are unused due to modular refactoring
// pub use context::VulkanContext; // Unused in current structure
// pub use swapchain::VulkanSwapchain; // Unused in current structure
// pub use pipeline::VulkanPipeline; // Not yet extracted
// pub use commands::VulkanCommands; // Unused in current structure
// pub use sync::VulkanSync; // Unused in current structure

// Re-export common Vulkan buffers
// use anyhow::Result; // Unused
// use ash::vk; // Unused
// use super::{Vertex, InstanceData, GeometryMode}; // Unused
// Removed duplicate import

/*
pub struct VulkanBuffers {
    pub vertex_buffer: vk::Buffer,
    pub vertex_memory: vk::DeviceMemory,
    pub instance_buffer: vk::Buffer,
    pub instance_memory: vk::DeviceMemory,
    pub storage_buffer: vk::Buffer,
    pub storage_memory: vk::DeviceMemory,
}

impl VulkanBuffers {
    pub unsafe fn new(context: &VulkanContext) -> Result<Self> {
        let vertex_data = [
            Vertex { pos: [-1.0, -1.0], uv: [0.0, 0.0] },
            Vertex { pos: [1.0, -1.0], uv: [1.0, 0.0] },
            Vertex { pos: [1.0, 1.0], uv: [1.0, 1.0] },
            Vertex { pos: [-1.0, 1.0], uv: [0.0, 1.0] },
        ];

        let (vertex_buffer, vertex_memory) = unsafe {
            Self::create_vertex_buffer(&context.device, context.physical_device, &vertex_data)?
        };

        let instance_count = 100000;
        let instance_data: Vec<InstanceData> = (0..instance_count)
            .map(|i| {
                let angle = i as f32 * 0.1;
                InstanceData {
                    offset: [angle.sin() * 0.5, angle.cos() * 0.5],
                    scale: [0.01, 0.01],
                    rotation_cos: angle.cos(),
                    rotation_sin: angle.sin(),
                    color_index: i % 16,
                    _padding: 0,
                }
            })
            .collect();

        let (instance_buffer, instance_memory) = unsafe {
            Self::create_instance_buffer(&context.device, context.physical_device, &instance_data)?
        };

        let storage_buffer_size = 4 * 1024 * 1024; // 4MB for generated points
        let (storage_buffer, storage_memory) = unsafe {
            Self::create_storage_buffer(&context.device, context.physical_device, storage_buffer_size)?
        };

        Ok(Self {
            vertex_buffer,
            vertex_memory,
            instance_buffer,
            instance_memory,
            storage_buffer,
            storage_memory,
        })
    }

    unsafe fn create_vertex_buffer(
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        data: &[Vertex],
    ) -> Result<(vk::Buffer, vk::DeviceMemory)> {
        let buffer_size = (std::mem::size_of::<Vertex>() * data.len()) as vk::DeviceSize;

        let buffer_info = vk::BufferCreateInfo {
            size: buffer_size,
            usage: vk::BufferUsageFlags::VERTEX_BUFFER,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };

        let buffer = unsafe { device.create_buffer(&buffer_info, None)? };
        let memory_requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
        let memory_type = Self::find_memory_type(
            device,
            physical_device,
            memory_requirements.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        let alloc_info = vk::MemoryAllocateInfo {
            allocation_size: memory_requirements.size,
            memory_type_index: memory_type,
            ..Default::default()
        };

        let memory = unsafe { device.allocate_memory(&alloc_info, None)? };
        unsafe { device.bind_buffer_memory(buffer, memory, 0)? };

        let ptr = unsafe { device.map_memory(memory, 0, buffer_size, vk::MemoryMapFlags::empty())? };
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr as *mut Vertex, data.len());
        }
        unsafe { device.unmap_memory(memory) };

        Ok((buffer, memory))
    }

    unsafe fn create_instance_buffer(
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        data: &[InstanceData],
    ) -> Result<(vk::Buffer, vk::DeviceMemory)> {
        let buffer_size = (std::mem::size_of::<InstanceData>() * data.len()) as vk::DeviceSize;

        let buffer_info = vk::BufferCreateInfo {
            size: buffer_size,
            usage: vk::BufferUsageFlags::VERTEX_BUFFER,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };

        let buffer = unsafe { device.create_buffer(&buffer_info, None)? };
        let memory_requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
        let memory_type = Self::find_memory_type(
            device,
            physical_device,
            memory_requirements.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        let alloc_info = vk::MemoryAllocateInfo {
            allocation_size: memory_requirements.size,
            memory_type_index: memory_type,
            ..Default::default()
        };

        let memory = unsafe { device.allocate_memory(&alloc_info, None)? };
        unsafe { device.bind_buffer_memory(buffer, memory, 0)? };

        let ptr = unsafe { device.map_memory(memory, 0, buffer_size, vk::MemoryMapFlags::empty())? };
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr as *mut InstanceData, data.len());
        }
        unsafe { device.unmap_memory(memory) };

        Ok((buffer, memory))
    }

    unsafe fn create_storage_buffer(
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        size: vk::DeviceSize,
    ) -> Result<(vk::Buffer, vk::DeviceMemory)> {
        let buffer_info = vk::BufferCreateInfo {
            size,
            usage: vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::VERTEX_BUFFER,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };

        let buffer = unsafe { device.create_buffer(&buffer_info, None)? };
        let memory_requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
        let memory_type = Self::find_memory_type(
            device,
            physical_device,
            memory_requirements.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        let alloc_info = vk::MemoryAllocateInfo {
            allocation_size: memory_requirements.size,
            memory_type_index: memory_type,
            ..Default::default()
        };

        let memory = unsafe { device.allocate_memory(&alloc_info, None)? };
        unsafe { device.bind_buffer_memory(buffer, memory, 0)? };

        Ok((buffer, memory))
    }

    fn find_memory_type(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        type_filter: u32,
        properties: vk::MemoryPropertyFlags,
    ) -> Result<u32> {
        let memory_properties = unsafe { instance.get_physical_device_memory_properties(physical_device) };

        for (i, memory_type) in memory_properties.memory_types.iter().enumerate() {
            if (type_filter & (1 << i)) != 0 && memory_type.property_flags.contains(properties) {
                return Ok(i as u32);
            }
        }

        Err(anyhow::anyhow!("Failed to find suitable memory type"))
    }
}

impl Drop for VulkanBuffers {
    fn drop(&mut self) {
        // Note: This should be cleaned up properly in a real implementation
        // For now, we'll rely on the device cleanup
    }
}
*/