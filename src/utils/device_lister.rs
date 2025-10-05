use anyhow::Result;
use ash::vk;
use cpal::traits::{DeviceTrait, HostTrait};
use midir::{Ignore, MidiInput};
use std::ffi::{CString, CStr};

pub struct DeviceLister;

impl DeviceLister {
    pub fn list_all_devices() {
        println!("\n=== Available Devices ===\n");
        Self::list_gpus();
        Self::list_midi_devices();
        Self::list_audio_devices();
    }

    fn list_gpus() {
        println!("📊 Graphics Devices (GPUs):");
        println!("----------------------------");

        match Self::enumerate_gpus() {
            Ok(gpus) => {
                if gpus.is_empty() {
                    println!("  No Vulkan-capable GPUs found");
                } else {
                    for (i, gpu) in gpus.iter().enumerate() {
                        println!("  [{i}] {gpu}");
                    }
                }
            }
            Err(e) => {
                println!("  Error enumerating GPUs: {e}");
            }
        }
        println!();
    }

    fn enumerate_gpus() -> Result<Vec<String>> {
        unsafe {
            let entry = ash::Entry::linked();

            let app_name = CString::new("device-lister")?;
            let app_info = vk::ApplicationInfo {
                p_application_name: app_name.as_ptr(),
                application_version: vk::make_api_version(0, 1, 0, 0),
                p_engine_name: app_name.as_ptr(),
                engine_version: vk::make_api_version(0, 1, 0, 0),
                api_version: vk::make_api_version(0, 1, 0, 0),
                ..Default::default()
            };

            let create_info = vk::InstanceCreateInfo {
                p_application_info: &app_info,
                ..Default::default()
            };

            let instance = entry.create_instance(&create_info, None)?;
            let physical_devices = instance.enumerate_physical_devices()?;

            let mut gpu_list = Vec::new();

            for device in physical_devices {
                let properties = instance.get_physical_device_properties(device);
                let name = CStr::from_ptr(properties.device_name.as_ptr())
                    .to_string_lossy()
                    .into_owned();

                let device_type = match properties.device_type {
                    vk::PhysicalDeviceType::DISCRETE_GPU => "Discrete GPU",
                    vk::PhysicalDeviceType::INTEGRATED_GPU => "Integrated GPU",
                    vk::PhysicalDeviceType::VIRTUAL_GPU => "Virtual GPU",
                    vk::PhysicalDeviceType::CPU => "CPU",
                    _ => "Other",
                };

                let mem_props = instance.get_physical_device_memory_properties(device);
                let mut total_vram = 0u64;
                for i in 0..mem_props.memory_heap_count as usize {
                    let heap = mem_props.memory_heaps[i];
                    if heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL) {
                        total_vram += heap.size;
                    }
                }
                let vram_gb = total_vram as f64 / (1024.0 * 1024.0 * 1024.0);

                let major = vk::api_version_major(properties.api_version);
                let minor = vk::api_version_minor(properties.api_version);
                let patch = vk::api_version_patch(properties.api_version);

                let info = format!("{name} ({device_type}) - VRAM: {vram_gb:.1} GB, Vulkan: {major}.{minor}.{patch}");

                gpu_list.push(info);
            }

            instance.destroy_instance(None);
            Ok(gpu_list)
        }
    }

    fn list_midi_devices() {
        println!("🎹 MIDI Input Devices:");
        println!("----------------------");

        match MidiInput::new("device-lister") {
            Ok(mut midi_in) => {
                midi_in.ignore(Ignore::None);
                let ports = midi_in.ports();

                if ports.is_empty() {
                    println!("  No MIDI input devices found");
                } else {
                    for (i, port) in ports.iter().enumerate() {
                        match midi_in.port_name(port) {
                            Ok(name) => println!("  [{i}] {name}"),
                            Err(e) => println!("  [{i}] <Error reading name: {e}>"),
                        }
                    }
                }
            }
            Err(e) => {
                println!("  Error initializing MIDI: {e}");
            }
        }
        println!();
    }

    fn list_audio_devices() {
        println!("🎵 Audio Input Devices:");
        println!("----------------------");

        let host = cpal::default_host();

        if let Some(device) = host.default_input_device() {
            match device.name() {
                Ok(name) => println!("  [DEFAULT] {name}"),
                Err(e) => println!("  [DEFAULT] <Error reading name: {e}>"),
            }
        }

        match host.input_devices() {
            Ok(devices) => {
                let devices: Vec<_> = devices.collect();
                if devices.is_empty() {
                    println!("  No audio input devices found");
                } else {
                    for (i, device) in devices.iter().enumerate() {
                        match device.name() {
                            Ok(name) => {
                                // Get supported configs for more info
                                let config_info = match device.supported_input_configs() {
                                    Ok(configs) => {
                                        let configs: Vec<_> = configs.collect();
                                        if !configs.is_empty() {
                                            let first = &configs[0];
                                            let last = &configs[configs.len() - 1];
                                            format!(
                                                " ({}Hz-{}Hz, {} channels)",
                                                first.min_sample_rate().0,
                                                last.max_sample_rate().0,
                                                first.channels()
                                            )
                                        } else {
                                            String::new()
                                        }
                                    }
                                    Err(_) => String::new(),
                                };

                                println!("  [{i}] {name}{config_info}");
                            }
                            Err(e) => println!("  [{i}] <Error reading name: {e}>"),
                        }
                    }
                }
            }
            Err(e) => {
                println!("  Error enumerating audio devices: {e}");
            }
        }

        println!("\n🎵 Audio Output Devices:");
        println!("----------------------");

        if let Some(device) = host.default_output_device() {
            match device.name() {
                Ok(name) => println!("  [DEFAULT] {name}"),
                Err(e) => println!("  [DEFAULT] <Error reading name: {e}>"),
            }
        }

        match host.output_devices() {
            Ok(devices) => {
                let devices: Vec<_> = devices.collect();
                if devices.is_empty() {
                    println!("  No audio output devices found");
                } else {
                    for (i, device) in devices.iter().enumerate() {
                        match device.name() {
                            Ok(name) => println!("  [{i}] {name}"),
                            Err(e) => println!("  [{i}] <Error reading name: {e}>"),
                        }
                    }
                }
            }
            Err(e) => {
                println!("  Error enumerating audio output devices: {e}");
            }
        }
        println!();
    }
}