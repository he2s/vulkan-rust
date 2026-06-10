pub mod context;
pub mod swapchain;

pub use context::VulkanContext;
pub use swapchain::VulkanSwapchain;

use anyhow::Result;
use ash::vk;
use super::{Vertex, InstanceData};

// VulkanBuffers implementation
pub struct VulkanBuffers {
    // Complex geometry buffers
    pub vertex_buffer: vk::Buffer,
    pub vertex_memory: vk::DeviceMemory,
    pub index_buffer: vk::Buffer,
    pub index_memory: vk::DeviceMemory,
    pub instance_buffer: vk::Buffer,
    pub instance_memory: vk::DeviceMemory,
    // Storage buffer for compute-generated points
    pub point_storage_buffer: vk::Buffer,
    pub point_storage_memory: vk::DeviceMemory,

    // Trivial geometry buffers (optimized fullscreen quad)
    pub trivial_vertex_buffer: vk::Buffer,
    pub trivial_vertex_memory: vk::DeviceMemory,
}

impl VulkanBuffers {
    /// # Safety
    /// `context` must hold a live device; the returned buffers must be
    /// destroyed before the device.
    pub unsafe fn new(context: &VulkanContext) -> Result<Self> {
        let vertex_data = [
            Vertex { pos: [-1.0, -1.0], uv: [0.0, 0.0] },
            Vertex { pos: [1.0, -1.0], uv: [1.0, 0.0] },
            Vertex { pos: [1.0, 1.0], uv: [1.0, 1.0] },
            Vertex { pos: [-1.0, 1.0], uv: [0.0, 1.0] },
        ];

        let (vertex_buffer, vertex_memory) = unsafe {
            Self::create_vertex_buffer(&context.device, &context.instance, context.physical_device, &vertex_data)?
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
            Self::create_instance_buffer(&context.device, &context.instance, context.physical_device, &instance_data)?
        };

        // Create index buffer for triangle indices (rectangle = 2 triangles)
        let index_data: [u16; 6] = [0, 1, 2, 0, 2, 3];
        let (index_buffer, index_memory) = unsafe {
            Self::create_index_buffer(&context.device, &context.instance, context.physical_device, &index_data)?
        };

        // Create point storage buffer for compute-generated points
        let storage_buffer_size = 4 * 1024 * 1024; // 4MB for generated points
        let (point_storage_buffer, point_storage_memory) = unsafe {
            Self::create_storage_buffer(&context.device, &context.instance, context.physical_device, storage_buffer_size)?
        };

        // Create trivial vertex buffer for fullscreen triangle
        let trivial_vertex_data = [
            Vertex { pos: [-1.0, -1.0], uv: [0.0, 0.0] },
            Vertex { pos: [ 3.0, -1.0], uv: [2.0, 0.0] },
            Vertex { pos: [-1.0,  3.0], uv: [0.0, 2.0] },
        ];
        let (trivial_vertex_buffer, trivial_vertex_memory) = unsafe {
            Self::create_vertex_buffer(&context.device, &context.instance, context.physical_device, &trivial_vertex_data)?
        };

        Ok(Self {
            vertex_buffer,
            vertex_memory,
            index_buffer,
            index_memory,
            instance_buffer,
            instance_memory,
            point_storage_buffer,
            point_storage_memory,
            trivial_vertex_buffer,
            trivial_vertex_memory,
        })
    }

    unsafe fn create_vertex_buffer(
        device: &ash::Device,
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        data: &[Vertex],
    ) -> Result<(vk::Buffer, vk::DeviceMemory)> {
        let buffer_size = std::mem::size_of_val(data) as vk::DeviceSize;

        let buffer_info = vk::BufferCreateInfo {
            size: buffer_size,
            usage: vk::BufferUsageFlags::VERTEX_BUFFER,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };

        let buffer = unsafe { device.create_buffer(&buffer_info, None)? };
        let memory_requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
        let memory_type = Self::find_memory_type(
            instance,
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
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        data: &[InstanceData],
    ) -> Result<(vk::Buffer, vk::DeviceMemory)> {
        let buffer_size = std::mem::size_of_val(data) as vk::DeviceSize;

        let buffer_info = vk::BufferCreateInfo {
            size: buffer_size,
            usage: vk::BufferUsageFlags::VERTEX_BUFFER,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };

        let buffer = unsafe { device.create_buffer(&buffer_info, None)? };
        let memory_requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
        let memory_type = Self::find_memory_type(
            instance,
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
        instance: &ash::Instance,
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
            instance,
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

    unsafe fn create_index_buffer(
        device: &ash::Device,
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        data: &[u16],
    ) -> Result<(vk::Buffer, vk::DeviceMemory)> {
        let buffer_size = std::mem::size_of_val(data) as vk::DeviceSize;

        let buffer_info = vk::BufferCreateInfo {
            size: buffer_size,
            usage: vk::BufferUsageFlags::INDEX_BUFFER,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };

        let buffer = unsafe { device.create_buffer(&buffer_info, None)? };
        let memory_requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
        let memory_type = Self::find_memory_type(
            instance,
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
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr as *mut u16, data.len());
        }
        unsafe { device.unmap_memory(memory) };

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

impl VulkanBuffers {
    /// # Safety
    /// This function is unsafe because it calls unsafe Vulkan functions to destroy buffers and free memory.
    /// The caller must ensure that all operations using these buffers have completed before calling this function.
    pub unsafe fn cleanup(&mut self, device: &ash::Device) {
        // Destroy all buffers and free memory
        unsafe {
            device.destroy_buffer(self.vertex_buffer, None);
            device.free_memory(self.vertex_memory, None);

            device.destroy_buffer(self.index_buffer, None);
            device.free_memory(self.index_memory, None);

            device.destroy_buffer(self.instance_buffer, None);
            device.free_memory(self.instance_memory, None);

            device.destroy_buffer(self.point_storage_buffer, None);
            device.free_memory(self.point_storage_memory, None);

            device.destroy_buffer(self.trivial_vertex_buffer, None);
            device.free_memory(self.trivial_vertex_memory, None);
        }
    }
}