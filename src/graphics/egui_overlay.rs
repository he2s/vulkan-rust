use anyhow::{anyhow, Result};
use ash::vk;
use egui::{Context, TextureId, ViewportId, epaint::Primitive};
use egui_winit::State as EguiWinitState;
use std::collections::HashMap;
use winit::event::WindowEvent;
use winit::window::Window;

/// egui vertex format matching shader layout
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct EguiVertex {
    pub pos: [f32; 2],
    pub uv: [f32; 2],
    pub color: [u8; 4],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct PushConstants {
    screen_size: [f32; 2],
}

struct TextureResources {
    image: vk::Image,
    image_view: vk::ImageView,
    memory: vk::DeviceMemory,
    sampler: vk::Sampler,
    descriptor_set: vk::DescriptorSet,
}

/// Lightweight egui overlay with full Vulkan rendering
pub struct EguiOverlay {
    pub egui_ctx: Context,
    egui_winit: EguiWinitState,
    pub show_overlay: bool,

    // Vulkan resources
    vertex_buffer: Option<vk::Buffer>,
    vertex_buffer_memory: Option<vk::DeviceMemory>,
    vertex_buffer_size: vk::DeviceSize,

    index_buffer: Option<vk::Buffer>,
    index_buffer_memory: Option<vk::DeviceMemory>,
    index_buffer_size: vk::DeviceSize,

    pipeline: Option<vk::Pipeline>,
    pipeline_layout: Option<vk::PipelineLayout>,
    descriptor_set_layout: Option<vk::DescriptorSetLayout>,
    descriptor_pool: Option<vk::DescriptorPool>,

    textures: HashMap<TextureId, TextureResources>,
    font_texture_uploaded: bool,

    // Store for deferred initialization
    instance: Option<ash::Instance>,
    physical_device: Option<vk::PhysicalDevice>,
    command_pool: Option<vk::CommandPool>,
    queue: Option<vk::Queue>,
}

impl EguiOverlay {
    pub fn new(window: &Window) -> Result<Self> {
        let egui_ctx = Context::default();

        let egui_winit = EguiWinitState::new(
            egui_ctx.clone(),
            ViewportId::ROOT,
            window,
            None,
            None,
            None,
        );

        Ok(Self {
            egui_ctx,
            egui_winit,
            show_overlay: false,
            vertex_buffer: None,
            vertex_buffer_memory: None,
            vertex_buffer_size: 0,
            index_buffer: None,
            index_buffer_memory: None,
            index_buffer_size: 0,
            pipeline: None,
            pipeline_layout: None,
            descriptor_set_layout: None,
            descriptor_pool: None,
            textures: HashMap::new(),
            font_texture_uploaded: false,
            instance: None,
            physical_device: None,
            command_pool: None,
            queue: None,
        })
    }

    /// Initialize Vulkan resources for rendering
    pub unsafe fn init_vulkan(
        &mut self,
        instance: &ash::Instance,
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        render_pass: vk::RenderPass,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
    ) -> Result<()> {
        unsafe {
            self.create_descriptor_set_layout(device)?;
            self.create_pipeline(device, render_pass)?;
            self.create_descriptor_pool(device)?;
        }

        // Store for deferred font upload (after first egui frame)
        // Clone instance handle - this is a lightweight operation that just clones the handle struct
        self.instance = Some(instance.clone());
        self.physical_device = Some(physical_device);
        self.command_pool = Some(command_pool);
        self.queue = Some(queue);

        Ok(())
    }

    unsafe fn create_descriptor_set_layout(&mut self, device: &ash::Device) -> Result<()> {
        let bindings = [vk::DescriptorSetLayoutBinding {
            binding: 0,
            descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::FRAGMENT,
            ..Default::default()
        }];

        let layout_info = vk::DescriptorSetLayoutCreateInfo::default()
            .bindings(&bindings);

        let layout = unsafe { device.create_descriptor_set_layout(&layout_info, None)? };
        self.descriptor_set_layout = Some(layout);
        Ok(())
    }

    unsafe fn create_pipeline(&mut self, device: &ash::Device, render_pass: vk::RenderPass) -> Result<()> {
        // Compile shaders
        let vert_code = Self::compile_shader("shaders/egui.vert", shaderc::ShaderKind::Vertex)?;
        let frag_code = Self::compile_shader("shaders/egui.frag", shaderc::ShaderKind::Fragment)?;

        let vert_module = unsafe { Self::create_shader_module(device, &vert_code)? };
        let frag_module = unsafe { Self::create_shader_module(device, &frag_code)? };

        let entry_point = std::ffi::CString::new("main")?;

        let shader_stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vert_module)
                .name(&entry_point),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(frag_module)
                .name(&entry_point),
        ];

        // Vertex input
        let binding_descriptions = [vk::VertexInputBindingDescription {
            binding: 0,
            stride: std::mem::size_of::<EguiVertex>() as u32,
            input_rate: vk::VertexInputRate::VERTEX,
        }];

        let attribute_descriptions = [
            vk::VertexInputAttributeDescription {
                location: 0,
                binding: 0,
                format: vk::Format::R32G32_SFLOAT,
                offset: 0,
            },
            vk::VertexInputAttributeDescription {
                location: 1,
                binding: 0,
                format: vk::Format::R32G32_SFLOAT,
                offset: 8,
            },
            vk::VertexInputAttributeDescription {
                location: 2,
                binding: 0,
                format: vk::Format::R8G8B8A8_UNORM,
                offset: 16,
            },
        ];

        let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(&binding_descriptions)
            .vertex_attribute_descriptions(&attribute_descriptions);

        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);

        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);

        let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.0)
            .cull_mode(vk::CullModeFlags::NONE)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE);

        let multisampling = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);

        let color_blend_attachment = vk::PipelineColorBlendAttachmentState {
            blend_enable: vk::TRUE,
            src_color_blend_factor: vk::BlendFactor::ONE,
            dst_color_blend_factor: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
            color_blend_op: vk::BlendOp::ADD,
            src_alpha_blend_factor: vk::BlendFactor::ONE,
            dst_alpha_blend_factor: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
            alpha_blend_op: vk::BlendOp::ADD,
            color_write_mask: vk::ColorComponentFlags::RGBA,
        };

        let color_blending = vk::PipelineColorBlendStateCreateInfo::default()
            .attachments(std::slice::from_ref(&color_blend_attachment));

        let push_constant_range = vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::VERTEX,
            offset: 0,
            size: std::mem::size_of::<PushConstants>() as u32,
        };

        let descriptor_set_layout = self.descriptor_set_layout
            .ok_or_else(|| anyhow!("Descriptor set layout not created"))?;

        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(std::slice::from_ref(&descriptor_set_layout))
            .push_constant_ranges(std::slice::from_ref(&push_constant_range));

        let pipeline_layout = unsafe { device.create_pipeline_layout(&pipeline_layout_info, None)? };
        self.pipeline_layout = Some(pipeline_layout);

        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state = vk::PipelineDynamicStateCreateInfo::default()
            .dynamic_states(&dynamic_states);

        let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&shader_stages)
            .vertex_input_state(&vertex_input_info)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterizer)
            .multisample_state(&multisampling)
            .color_blend_state(&color_blending)
            .dynamic_state(&dynamic_state)
            .layout(pipeline_layout)
            .render_pass(render_pass)
            .subpass(0);

        let pipelines = unsafe {
            device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
                .map_err(|e| anyhow!("Failed to create pipeline: {:?}", e.1))?
        };

        self.pipeline = Some(pipelines[0]);

        unsafe {
            device.destroy_shader_module(vert_module, None);
            device.destroy_shader_module(frag_module, None);
        }

        Ok(())
    }

    fn compile_shader(path: &str, kind: shaderc::ShaderKind) -> Result<Vec<u32>> {
        let source = std::fs::read_to_string(path)?;
        let compiler = shaderc::Compiler::new()
            .ok_or_else(|| anyhow!("Failed to create shader compiler"))?;

        let binary_result = compiler.compile_into_spirv(&source, kind, path, "main", None)?;
        Ok(binary_result.as_binary().to_vec())
    }

    unsafe fn create_shader_module(device: &ash::Device, code: &[u32]) -> Result<vk::ShaderModule> {
        let create_info = vk::ShaderModuleCreateInfo::default().code(code);
        Ok(device.create_shader_module(&create_info, None)?)
    }

    unsafe fn create_descriptor_pool(&mut self, device: &ash::Device) -> Result<()> {
        let pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: 100,
        }];

        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(100)
            .pool_sizes(&pool_sizes);

        let pool = device.create_descriptor_pool(&pool_info, None)?;
        self.descriptor_pool = Some(pool);
        Ok(())
    }

    unsafe fn upload_font_texture(
        &mut self,
        instance: &ash::Instance,
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
    ) -> Result<()> {
        let font_image = self.egui_ctx.fonts(|fonts| fonts.image());

        let width = font_image.width();
        let height = font_image.height();
        let pixels: Vec<u8> = font_image
            .srgba_pixels(None)
            .flat_map(|color| [color.r(), color.g(), color.b(), color.a()])
            .collect();

        let texture_id = TextureId::default();
        self.create_texture(
            instance,
            device,
            physical_device,
            command_pool,
            queue,
            texture_id,
            &pixels,
            width,
            height,
        )?;

        Ok(())
    }

    unsafe fn create_texture(
        &mut self,
        instance: &ash::Instance,
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
        texture_id: TextureId,
        pixels: &[u8],
        width: usize,
        height: usize,
    ) -> Result<()> {
        let image_size = (width * height * 4) as vk::DeviceSize;

        // Create staging buffer
        let (staging_buffer, staging_memory) = self.create_buffer(
            instance,
            device,
            physical_device,
            image_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        // Copy pixels to staging buffer
        let ptr = device.map_memory(staging_memory, 0, image_size, vk::MemoryMapFlags::empty())?;
        std::ptr::copy_nonoverlapping(pixels.as_ptr(), ptr as *mut u8, pixels.len());
        device.unmap_memory(staging_memory);

        // Create image
        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .extent(vk::Extent3D {
                width: width as u32,
                height: height as u32,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .format(vk::Format::R8G8B8A8_UNORM)
            .tiling(vk::ImageTiling::OPTIMAL)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .samples(vk::SampleCountFlags::TYPE_1);

        let image = device.create_image(&image_info, None)?;
        let mem_requirements = device.get_image_memory_requirements(image);
        let memory_type = Self::find_memory_type(
            instance,
            physical_device,
            mem_requirements.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_requirements.size)
            .memory_type_index(memory_type);

        let memory = device.allocate_memory(&alloc_info, None)?;
        device.bind_image_memory(image, memory, 0)?;

        // Transition and copy
        self.transition_image_layout(
            device,
            command_pool,
            queue,
            image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        )?;

        self.copy_buffer_to_image(device, command_pool, queue, staging_buffer, image, width as u32, height as u32)?;

        self.transition_image_layout(
            device,
            command_pool,
            queue,
            image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        )?;

        // Clean up staging
        device.destroy_buffer(staging_buffer, None);
        device.free_memory(staging_memory, None);

        // Create image view
        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(vk::Format::R8G8B8A8_UNORM)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });

        let image_view = device.create_image_view(&view_info, None)?;

        // Create sampler
        let sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .anisotropy_enable(false)
            .border_color(vk::BorderColor::INT_OPAQUE_BLACK)
            .unnormalized_coordinates(false)
            .compare_enable(false)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR);

        let sampler = device.create_sampler(&sampler_info, None)?;

        // Create descriptor set
        let descriptor_set_layout = self.descriptor_set_layout
            .ok_or_else(|| anyhow!("Descriptor set layout not created"))?;
        let descriptor_pool = self.descriptor_pool
            .ok_or_else(|| anyhow!("Descriptor pool not created"))?;

        let layouts = [descriptor_set_layout];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&layouts);

        let descriptor_sets = device.allocate_descriptor_sets(&alloc_info)?;
        let descriptor_set = descriptor_sets[0];

        // Update descriptor set
        let image_info_descriptor = vk::DescriptorImageInfo {
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            image_view,
            sampler,
        };

        let descriptor_write = vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(0)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(std::slice::from_ref(&image_info_descriptor));

        device.update_descriptor_sets(&[descriptor_write], &[]);

        // Store texture resources
        self.textures.insert(
            texture_id,
            TextureResources {
                image,
                image_view,
                memory,
                sampler,
                descriptor_set,
            },
        );

        Ok(())
    }

    /// Render egui overlay
    pub unsafe fn render(
        &mut self,
        instance: &ash::Instance,
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        cmd_buffer: vk::CommandBuffer,
        extent: vk::Extent2D,
        clipped_primitives: &[egui::ClippedPrimitive],
    ) -> Result<()> {
        if clipped_primitives.is_empty() {
            return Ok(());
        }

        // Upload font texture on first render (after egui has run once)
        if !self.font_texture_uploaded {
            let inst = self.instance.clone()
                .ok_or_else(|| anyhow!("Instance not initialized"))?;
            let phys_dev = self.physical_device
                .ok_or_else(|| anyhow!("Physical device not initialized"))?;
            let pool = self.command_pool
                .ok_or_else(|| anyhow!("Command pool not initialized"))?;
            let q = self.queue
                .ok_or_else(|| anyhow!("Queue not initialized"))?;

            self.upload_font_texture(&inst, device, phys_dev, pool, q)?;
            self.font_texture_uploaded = true;
        }

        // Bind pipeline
        let pipeline = self.pipeline
            .ok_or_else(|| anyhow!("Pipeline not created"))?;
        device.cmd_bind_pipeline(cmd_buffer, vk::PipelineBindPoint::GRAPHICS, pipeline);

        // Set viewport
        let viewport = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: extent.width as f32,
            height: extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        device.cmd_set_viewport(cmd_buffer, 0, &[viewport]);

        // Push constants
        let push_constants = PushConstants {
            screen_size: [extent.width as f32, extent.height as f32],
        };

        let pipeline_layout = self.pipeline_layout
            .ok_or_else(|| anyhow!("Pipeline layout not created"))?;

        device.cmd_push_constants(
            cmd_buffer,
            pipeline_layout,
            vk::ShaderStageFlags::VERTEX,
            0,
            std::slice::from_raw_parts(
                &push_constants as *const PushConstants as *const u8,
                std::mem::size_of::<PushConstants>(),
            ),
        );

        // Render each primitive
        for egui::ClippedPrimitive { clip_rect, primitive } in clipped_primitives {
            let mesh = match primitive {
                Primitive::Mesh(mesh) => mesh,
                Primitive::Callback(_) => continue,
            };

            if mesh.vertices.is_empty() || mesh.indices.is_empty() {
                continue;
            }

            // Set scissor
            let min_x = clip_rect.min.x.max(0.0) as u32;
            let min_y = clip_rect.min.y.max(0.0) as u32;
            let max_x = (clip_rect.max.x.min(extent.width as f32)) as u32;
            let max_y = (clip_rect.max.y.min(extent.height as f32)) as u32;

            let scissor = vk::Rect2D {
                offset: vk::Offset2D {
                    x: min_x as i32,
                    y: min_y as i32,
                },
                extent: vk::Extent2D {
                    width: (max_x - min_x).max(1),
                    height: (max_y - min_y).max(1),
                },
            };

            device.cmd_set_scissor(cmd_buffer, 0, &[scissor]);

            // Update buffers
            self.update_vertex_buffer(instance, device, physical_device, &mesh.vertices)?;
            self.update_index_buffer(instance, device, physical_device, &mesh.indices)?;

            // Bind buffers
            let vertex_buffer = self.vertex_buffer
                .ok_or_else(|| anyhow!("Vertex buffer not created"))?;
            let index_buffer = self.index_buffer
                .ok_or_else(|| anyhow!("Index buffer not created"))?;

            device.cmd_bind_vertex_buffers(cmd_buffer, 0, &[vertex_buffer], &[0]);
            device.cmd_bind_index_buffer(cmd_buffer, index_buffer, 0, vk::IndexType::UINT32);

            // Bind descriptor set for texture
            if let Some(texture) = self.textures.get(&mesh.texture_id) {
                device.cmd_bind_descriptor_sets(
                    cmd_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    pipeline_layout,
                    0,
                    &[texture.descriptor_set],
                    &[],
                );
            }

            // Draw
            device.cmd_draw_indexed(cmd_buffer, mesh.indices.len() as u32, 1, 0, 0, 0);
        }

        Ok(())
    }

    unsafe fn update_vertex_buffer(
        &mut self,
        instance: &ash::Instance,
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        vertices: &[egui::epaint::Vertex],
    ) -> Result<()> {
        let converted_vertices: Vec<EguiVertex> = vertices
            .iter()
            .map(|v| EguiVertex {
                pos: [v.pos.x, v.pos.y],
                uv: [v.uv.x, v.uv.y],
                color: [v.color.r(), v.color.g(), v.color.b(), v.color.a()],
            })
            .collect();

        let buffer_size = (std::mem::size_of::<EguiVertex>() * converted_vertices.len()) as vk::DeviceSize;

        if buffer_size > self.vertex_buffer_size {
            if let Some(buffer) = self.vertex_buffer.take() {
                device.destroy_buffer(buffer, None);
            }
            if let Some(memory) = self.vertex_buffer_memory.take() {
                device.free_memory(memory, None);
            }

            let (buffer, memory) = self.create_buffer(
                instance,
                device,
                physical_device,
                buffer_size,
                vk::BufferUsageFlags::VERTEX_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?;

            self.vertex_buffer = Some(buffer);
            self.vertex_buffer_memory = Some(memory);
            self.vertex_buffer_size = buffer_size;
        }

        let vertex_buffer_memory = self.vertex_buffer_memory
            .ok_or_else(|| anyhow!("Vertex buffer memory not allocated"))?;

        let ptr = device.map_memory(vertex_buffer_memory, 0, buffer_size, vk::MemoryMapFlags::empty())?;
        std::ptr::copy_nonoverlapping(converted_vertices.as_ptr(), ptr as *mut EguiVertex, converted_vertices.len());
        device.unmap_memory(vertex_buffer_memory);

        Ok(())
    }

    unsafe fn update_index_buffer(
        &mut self,
        instance: &ash::Instance,
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        indices: &[u32],
    ) -> Result<()> {
        let buffer_size = (std::mem::size_of::<u32>() * indices.len()) as vk::DeviceSize;

        if buffer_size > self.index_buffer_size {
            if let Some(buffer) = self.index_buffer.take() {
                device.destroy_buffer(buffer, None);
            }
            if let Some(memory) = self.index_buffer_memory.take() {
                device.free_memory(memory, None);
            }

            let (buffer, memory) = self.create_buffer(
                instance,
                device,
                physical_device,
                buffer_size,
                vk::BufferUsageFlags::INDEX_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?;

            self.index_buffer = Some(buffer);
            self.index_buffer_memory = Some(memory);
            self.index_buffer_size = buffer_size;
        }

        let index_buffer_memory = self.index_buffer_memory
            .ok_or_else(|| anyhow!("Index buffer memory not allocated"))?;

        let ptr = device.map_memory(index_buffer_memory, 0, buffer_size, vk::MemoryMapFlags::empty())?;
        std::ptr::copy_nonoverlapping(indices.as_ptr(), ptr as *mut u32, indices.len());
        device.unmap_memory(index_buffer_memory);

        Ok(())
    }

    // Helper functions
    unsafe fn create_buffer(
        &self,
        instance: &ash::Instance,
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        properties: vk::MemoryPropertyFlags,
    ) -> Result<(vk::Buffer, vk::DeviceMemory)> {
        let buffer_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = device.create_buffer(&buffer_info, None)?;
        let mem_requirements = device.get_buffer_memory_requirements(buffer);
        let memory_type = Self::find_memory_type(instance, physical_device, mem_requirements.memory_type_bits, properties)?;

        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_requirements.size)
            .memory_type_index(memory_type);

        let memory = device.allocate_memory(&alloc_info, None)?;
        device.bind_buffer_memory(buffer, memory, 0)?;

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

        Err(anyhow!("Failed to find suitable memory type"))
    }

    unsafe fn transition_image_layout(
        &self,
        device: &ash::Device,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
        image: vk::Image,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
    ) -> Result<()> {
        let cmd_buffer = self.begin_single_time_commands(device, command_pool)?;

        let (src_access_mask, dst_access_mask, src_stage, dst_stage) = match (old_layout, new_layout) {
            (vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL) => (
                vk::AccessFlags::empty(),
                vk::AccessFlags::TRANSFER_WRITE,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
            ),
            (vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL) => (
                vk::AccessFlags::TRANSFER_WRITE,
                vk::AccessFlags::SHADER_READ,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
            ),
            _ => return Err(anyhow!("Unsupported layout transition")),
        };

        let barrier = vk::ImageMemoryBarrier::default()
            .old_layout(old_layout)
            .new_layout(new_layout)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            })
            .src_access_mask(src_access_mask)
            .dst_access_mask(dst_access_mask);

        device.cmd_pipeline_barrier(cmd_buffer, src_stage, dst_stage, vk::DependencyFlags::empty(), &[], &[], &[barrier]);

        self.end_single_time_commands(device, command_pool, queue, cmd_buffer)?;
        Ok(())
    }

    unsafe fn copy_buffer_to_image(
        &self,
        device: &ash::Device,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
        buffer: vk::Buffer,
        image: vk::Image,
        width: u32,
        height: u32,
    ) -> Result<()> {
        let cmd_buffer = self.begin_single_time_commands(device, command_pool)?;

        let region = vk::BufferImageCopy::default()
            .buffer_offset(0)
            .buffer_row_length(0)
            .buffer_image_height(0)
            .image_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            })
            .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
            .image_extent(vk::Extent3D { width, height, depth: 1 });

        device.cmd_copy_buffer_to_image(cmd_buffer, buffer, image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[region]);

        self.end_single_time_commands(device, command_pool, queue, cmd_buffer)?;
        Ok(())
    }

    unsafe fn begin_single_time_commands(&self, device: &ash::Device, command_pool: vk::CommandPool) -> Result<vk::CommandBuffer> {
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let cmd_buffer = device.allocate_command_buffers(&alloc_info)?[0];

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        device.begin_command_buffer(cmd_buffer, &begin_info)?;
        Ok(cmd_buffer)
    }

    unsafe fn end_single_time_commands(
        &self,
        device: &ash::Device,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
        cmd_buffer: vk::CommandBuffer,
    ) -> Result<()> {
        device.end_command_buffer(cmd_buffer)?;

        let cmd_buffers = [cmd_buffer];
        let submit_info = vk::SubmitInfo::default()
            .command_buffers(&cmd_buffers);

        let submit_infos = [submit_info];
        device.queue_submit(queue, &submit_infos, vk::Fence::null())?;
        device.queue_wait_idle(queue)?;
        device.free_command_buffers(command_pool, &cmd_buffers);

        Ok(())
    }

    pub fn handle_event(&mut self, window: &Window, event: &WindowEvent) -> bool {
        let response = self.egui_winit.on_window_event(window, event);
        response.consumed
    }

    pub fn run_ui(&mut self, window: &Window, current_preset: &str) -> (egui::FullOutput, Vec<egui::ClippedPrimitive>) {
        let raw_input = self.egui_winit.take_egui_input(window);

        let full_output = self.egui_ctx.run(raw_input, |ctx| {
            if self.show_overlay {
                egui::Area::new("preset_overlay".into())
                    .fixed_pos(egui::pos2(10.0, 10.0))
                    .show(ctx, |ui| {
                        ui.visuals_mut().window_fill = egui::Color32::from_black_alpha(200);
                        egui::Frame::window(ui.style())
                            .fill(egui::Color32::from_black_alpha(200))
                            .show(ui, |ui| {
                                ui.heading("🎨 Current Preset");
                                ui.separator();
                                ui.label(egui::RichText::new(current_preset).size(20.0).strong());
                                ui.add_space(4.0);
                                ui.label(egui::RichText::new("Press H to toggle").size(12.0).weak());
                            });
                    });
            }
        });

        let clipped_primitives = self.egui_ctx.tessellate(full_output.shapes.clone(), full_output.pixels_per_point);
        self.egui_winit.handle_platform_output(window, full_output.platform_output.clone());

        (full_output, clipped_primitives)
    }

    pub fn toggle_overlay(&mut self) {
        self.show_overlay = !self.show_overlay;
    }

    pub unsafe fn destroy(&mut self, device: &ash::Device) {
        if let Some(buffer) = self.vertex_buffer.take() {
            device.destroy_buffer(buffer, None);
        }
        if let Some(memory) = self.vertex_buffer_memory.take() {
            device.free_memory(memory, None);
        }
        if let Some(buffer) = self.index_buffer.take() {
            device.destroy_buffer(buffer, None);
        }
        if let Some(memory) = self.index_buffer_memory.take() {
            device.free_memory(memory, None);
        }

        for (_, texture) in self.textures.drain() {
            device.destroy_sampler(texture.sampler, None);
            device.destroy_image_view(texture.image_view, None);
            device.destroy_image(texture.image, None);
            device.free_memory(texture.memory, None);
        }

        if let Some(pipeline) = self.pipeline.take() {
            device.destroy_pipeline(pipeline, None);
        }
        if let Some(layout) = self.pipeline_layout.take() {
            device.destroy_pipeline_layout(layout, None);
        }
        if let Some(layout) = self.descriptor_set_layout.take() {
            device.destroy_descriptor_set_layout(layout, None);
        }
        if let Some(pool) = self.descriptor_pool.take() {
            device.destroy_descriptor_pool(pool, None);
        }
    }
}
