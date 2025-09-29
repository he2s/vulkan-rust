use anyhow::{Result, anyhow};
use ash::{Entry, vk};
use ash::khr::surface;
use std::ffi::{CString, c_char};
use winit::window::Window;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

pub struct VulkanContext {
    pub _entry: Entry,
    pub instance: ash::Instance,
    pub surface_loader: surface::Instance,
    pub surface: vk::SurfaceKHR,
    pub physical_device: vk::PhysicalDevice,
    pub device: ash::Device,
    pub queue_family_index: u32,
    pub queue: vk::Queue,
}

impl VulkanContext {
    pub unsafe fn new(window: &Window, validation_layers: bool) -> Result<Self> {
        let entry = Entry::linked();
        let display_handle = window.display_handle()?.as_raw();
        let window_handle = window.window_handle()?.as_raw();
        let required_extensions =
            ash_window::enumerate_required_extensions(display_handle)?.to_vec();

        let instance = unsafe { Self::create_instance(&entry, &required_extensions, validation_layers)? };
        let surface =
            unsafe { ash_window::create_surface(&entry, &instance, display_handle, window_handle, None)? };
        let surface_loader = surface::Instance::new(&entry, &instance);
        let (physical_device, queue_family_index) =
            unsafe { Self::select_physical_device(&instance, &surface_loader, surface)? };
        let (device, queue) =
            unsafe { Self::create_logical_device(&instance, physical_device, queue_family_index)? };

        Ok(Self {
            _entry: entry,
            instance,
            surface_loader,
            surface,
            physical_device,
            device,
            queue_family_index,
            queue,
        })
    }

    unsafe fn create_instance(
        entry: &Entry,
        required_extensions: &[*const c_char],
        validation_layers: bool,
    ) -> Result<ash::Instance> {
        let app_name = CString::new("vulkan-pixel-shader")?;

        let layer_names: Vec<CString> = if validation_layers {
            vec![CString::new("VK_LAYER_KHRONOS_validation")?]
        } else {
            vec![]
        };

        let layer_name_pointers: Vec<*const c_char> =
            layer_names.iter().map(|name| name.as_ptr()).collect();

        let app_info = vk::ApplicationInfo {
            p_application_name: app_name.as_ptr(),
            application_version: vk::make_api_version(0, 1, 0, 0),
            p_engine_name: app_name.as_ptr(),
            engine_version: vk::make_api_version(0, 1, 0, 0),
            api_version: vk::make_api_version(0, 1, 2, 0),
            ..Default::default()
        };

        let create_info = vk::InstanceCreateInfo {
            p_application_info: &app_info,
            enabled_layer_count: layer_name_pointers.len() as u32,
            pp_enabled_layer_names: layer_name_pointers.as_ptr(),
            enabled_extension_count: required_extensions.len() as u32,
            pp_enabled_extension_names: required_extensions.as_ptr(),
            ..Default::default()
        };

        Ok(unsafe { entry.create_instance(&create_info, None)? })
    }

    unsafe fn select_physical_device(
        instance: &ash::Instance,
        surface_loader: &surface::Instance,
        surface: vk::SurfaceKHR,
    ) -> Result<(vk::PhysicalDevice, u32)> {
        let physical_devices = unsafe { instance.enumerate_physical_devices()? };

        for device in physical_devices {
            let queue_families = unsafe { instance.get_physical_device_queue_family_properties(device) };

            for (index, queue_family) in queue_families.iter().enumerate() {
                let index = index as u32;

                if queue_family.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                    && unsafe { surface_loader.get_physical_device_surface_support(device, index, surface)? }
                {
                    return Ok((device, index));
                }
            }
        }

        Err(anyhow!("No suitable GPU found"))
    }

    unsafe fn create_logical_device(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        queue_family_index: u32,
    ) -> Result<(ash::Device, vk::Queue)> {
        let queue_priorities = [1.0f32];
        let queue_create_info = vk::DeviceQueueCreateInfo {
            queue_family_index,
            queue_count: 1,
            p_queue_priorities: queue_priorities.as_ptr(),
            ..Default::default()
        };

        let device_extensions = [ash::khr::swapchain::NAME.as_ptr()];
        let device_create_info = vk::DeviceCreateInfo {
            queue_create_info_count: 1,
            p_queue_create_infos: &queue_create_info,
            enabled_extension_count: device_extensions.len() as u32,
            pp_enabled_extension_names: device_extensions.as_ptr(),
            ..Default::default()
        };

        let device = unsafe { instance.create_device(physical_device, &device_create_info, None)? };
        let queue = unsafe { device.get_device_queue(queue_family_index, 0) };

        Ok((device, queue))
    }
}

impl Drop for VulkanContext {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_device(None);
            self.surface_loader.destroy_surface(self.surface, None);
            self.instance.destroy_instance(None);
        }
    }
}