// src/renderer/context.rs

use anyhow::{anyhow, Result};
use ash::{vk, Entry, Instance, Device};
use ash::khr::surface;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use std::ffi::{CString, c_char};
use std::sync::Arc;
use winit::window::Window;
use tracing::{info, debug};

pub struct VulkanContext {
    entry: Entry,
    instance: Instance,
    physical_device: vk::PhysicalDevice,
    device: Device,
    queue: vk::Queue,
    queue_family_index: u32,
    surface_loader: surface::Instance,
}

impl VulkanContext {
    pub fn new(window: &Window, enable_validation: bool) -> Result<Self> {
        info!("Creating Vulkan context");

        let entry = Entry::linked();

        // Get required extensions
        let display_handle = window.display_handle()?.as_raw();
        let extensions = ash_window::enumerate_required_extensions(display_handle)?;

        // Create instance
        let instance = create_instance(&entry, extensions, enable_validation)?;

        // Create surface loader
        let surface_loader = surface::Instance::new(&entry, &instance);

        // Select physical device
        let (physical_device, queue_family_index) = select_physical_device(
            &instance,
            &surface_loader,
            window,
        )?;

        // Create logical device
        let (device, queue) = create_logical_device(
            &instance,
            physical_device,
            queue_family_index,
        )?;

        info!("Vulkan context created successfully");

        Ok(Self {
            entry,
            instance,
            physical_device,
            device,
            queue,
            queue_family_index,
            surface_loader,
        })
    }

    pub fn create_surface(&self, window: &Window) -> Result<vk::SurfaceKHR> {
        let display_handle = window.display_handle()?.as_raw();
        let window_handle = window.window_handle()?.as_raw();

        unsafe {
            Ok(ash_window::create_surface(
                &self.entry,
                &self.instance,
                display_handle,
                window_handle,
                None,
            )?)
        }
    }

    pub fn destroy_surface(&self, surface: vk::SurfaceKHR) {
        unsafe {
            self.surface_loader.destroy_surface(surface, None);
        }
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn queue(&self) -> vk::Queue {
        self.queue
    }

    pub fn queue_family_index(&self) -> u32 {
        self.queue_family_index
    }

    pub fn physical_device(&self) -> vk::PhysicalDevice {
        self.physical_device
    }

    pub fn surface_loader(&self) -> &surface::Instance {
        &self.surface_loader
    }
}

impl Drop for VulkanContext {
    fn drop(&mut self) {
        unsafe {
            debug!("Destroying Vulkan context");
            self.device.device_wait_idle().ok();
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

fn create_instance(
    entry: &Entry,
    extensions: &[*const c_char],
    enable_validation: bool,
) -> Result<Instance> {
    let app_name = CString::new("Vulkan Visualizer")?;
    let engine_name = CString::new("No Engine")?;

    let app_info = vk::ApplicationInfo::default()
        .application_name(&app_name)
        .application_version(vk::make_api_version(0, 1, 0, 0))
        .engine_name(&engine_name)
        .engine_version(vk::make_api_version(0, 1, 0, 0))
        .api_version(vk::API_VERSION_1_2);

    let mut create_info = vk::InstanceCreateInfo::default()
        .application_info(&app_info)
        .enabled_extension_names(extensions);

    let validation_layer = CString::new("VK_LAYER_KHRONOS_validation")?;
    let validation_layers = [validation_layer.as_ptr()];

    if enable_validation {
        if !check_validation_layer_support(entry)? {
            return Err(anyhow!("Validation layers requested but not available"));
        }
        create_info = create_info.enabled_layer_names(&validation_layers);
        info!("Validation layers enabled");
    }

    unsafe { Ok(entry.create_instance(&create_info, None)?) }
}

fn check_validation_layer_support(entry: &Entry) -> Result<bool> {
    let layer_properties = unsafe { entry.enumerate_instance_layer_properties()? };

    for layer in layer_properties {
        let name = unsafe {
            std::ffi::CStr::from_ptr(layer.layer_name.as_ptr())
        };
        if name.to_str()? == "VK_LAYER_KHRONOS_validation" {
            return Ok(true);
        }
    }

    Ok(false)
}

fn select_physical_device(
    instance: &Instance,
    surface_loader: &surface::Instance,
    window: &Window,
) -> Result<(vk::PhysicalDevice, u32)> {
    let display_handle = window.display_handle()?.as_raw();
    let window_handle = window.window_handle()?.as_raw();

    let temp_surface = unsafe {
        ash_window::create_surface(
            &Entry::linked(),
            instance,
            display_handle,
            window_handle,
            None,
        )?
    };

    let physical_devices = unsafe { instance.enumerate_physical_devices()? };

    let result = physical_devices
        .into_iter()
        .find_map(|pdev| {
            let queue_families = unsafe {
                instance.get_physical_device_queue_family_properties(pdev)
            };

            queue_families
                .iter()
                .enumerate()
                .find(|(idx, family)| {
                    family.queue_flags.contains(vk::QueueFlags::GRAPHICS) &&
                        unsafe {
                            surface_loader
                                .get_physical_device_surface_support(pdev, *idx as u32, temp_surface)
                                .unwrap_or(false)
                        }
                })
                .map(|(idx, _)| (pdev, idx as u32))
        })
        .ok_or_else(|| anyhow!("No suitable GPU found"));

    unsafe {
        surface_loader.destroy_surface(temp_surface, None);
    }

    let (pdev, qfi) = result?;

    let properties = unsafe { instance.get_physical_device_properties(pdev) };
    let device_name = unsafe {
        std::ffi::CStr::from_ptr(properties.device_name.as_ptr())
    };
    info!("Selected GPU: {}", device_name.to_str()?);

    Ok((pdev, qfi))
}

fn create_logical_device(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
    queue_family_index: u32,
) -> Result<(Device, vk::Queue)> {
    let queue_priorities = [1.0f32];

    let queue_create_info = vk::DeviceQueueCreateInfo::default()
        .queue_family_index(queue_family_index)
        .queue_priorities(&queue_priorities);

    let device_extensions = [
        ash::khr::swapchain::NAME.as_ptr(),
    ];

    let features = vk::PhysicalDeviceFeatures::default();

    let create_info = vk::DeviceCreateInfo::default()
        .queue_create_infos(std::slice::from_ref(&queue_create_info))
        .enabled_extension_names(&device_extensions)
        .enabled_features(&features);

    let device = unsafe {
        instance.create_device(physical_device, &create_info, None)?
    };

    let queue = unsafe {
        device.get_device_queue(queue_family_index, 0)
    };

    Ok((device, queue))
}