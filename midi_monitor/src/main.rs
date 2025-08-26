use midir::{MidiInput, Ignore};
use std::error::Error;
use std::io::{stdin, stdout, Write};

fn main() {
    match run() {
        Ok(_) => (),
        Err(err) => println!("Error: {}", err),
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let mut midi_in = MidiInput::new("MIDI Monitor")?;
    midi_in.ignore(Ignore::None);

    // Get available MIDI input ports
    let in_ports = midi_in.ports();
    
    if in_ports.is_empty() {
        println!("No MIDI input ports available!");
        return Ok(());
    }

    // List available ports
    println!("Available MIDI input ports:");
    for (i, port) in in_ports.iter().enumerate() {
        println!("{}: {}", i, midi_in.port_name(port)?);
    }

    // Let user select a port
    print!("Please select a port (0-{}): ", in_ports.len() - 1);
    stdout().flush()?;
    
    let mut input = String::new();
    stdin().read_line(&mut input)?;
    let port_index: usize = input.trim().parse()?;
    
    if port_index >= in_ports.len() {
        return Err("Invalid port selection".into());
    }

    let in_port = &in_ports[port_index];
    println!("Opening connection to: {}", midi_in.port_name(in_port)?);
    println!("Listening for MIDI events... Press Enter to exit.\n");

    // Connect to the selected port
    let _conn_in = midi_in.connect(in_port, "midi-monitor", move |timestamp, message, _| {
        print_midi_event(timestamp, message);
    }, ())?;

    // Keep the program running until user presses Enter
    let mut input = String::new();
    stdin().read_line(&mut input)?;
    
    println!("Closing connection");
    Ok(())
}

fn print_midi_event(timestamp: u64, message: &[u8]) {
    print!("[{}] ", timestamp);
    
    if message.is_empty() {
        println!("Empty MIDI message");
        return;
    }

    let status = message[0];
    let channel = (status & 0x0F) + 1; // MIDI channels are 1-16, not 0-15
    
    match status & 0xF0 {
        0x80 => {
            // Note Off
            if message.len() >= 3 {
                println!("Note Off  - Channel: {:2}, Note: {:3}, Velocity: {:3}", 
                        channel, message[1], message[2]);
            } else {
                println!("Note Off  - Incomplete message: {:?}", message);
            }
        }
        0x90 => {
            // Note On (velocity 0 is also Note Off)
            if message.len() >= 3 {
                if message[2] == 0 {
                    println!("Note Off  - Channel: {:2}, Note: {:3}, Velocity:   0", 
                            channel, message[1]);
                } else {
                    println!("Note On   - Channel: {:2}, Note: {:3}, Velocity: {:3}", 
                            channel, message[1], message[2]);
                }
            } else {
                println!("Note On   - Incomplete message: {:?}", message);
            }
        }
        0xA0 => {
            // Polyphonic Key Pressure
            if message.len() >= 3 {
                println!("Key Press - Channel: {:2}, Note: {:3}, Pressure: {:3}", 
                        channel, message[1], message[2]);
            } else {
                println!("Key Press - Incomplete message: {:?}", message);
            }
        }
        0xB0 => {
            // Control Change
            if message.len() >= 3 {
                println!("Ctrl Chng - Channel: {:2}, Controller: {:3}, Value: {:3}", 
                        channel, message[1], message[2]);
            } else {
                println!("Ctrl Chng - Incomplete message: {:?}", message);
            }
        }
        0xC0 => {
            // Program Change
            if message.len() >= 2 {
                println!("Prog Chng - Channel: {:2}, Program: {:3}", 
                        channel, message[1]);
            } else {
                println!("Prog Chng - Incomplete message: {:?}", message);
            }
        }
        0xD0 => {
            // Channel Pressure
            if message.len() >= 2 {
                println!("Ch Press  - Channel: {:2}, Pressure: {:3}", 
                        channel, message[1]);
            } else {
                println!("Ch Press  - Incomplete message: {:?}", message);
            }
        }
        0xE0 => {
            // Pitch Bend
            if message.len() >= 3 {
                let bend_value = (message[2] as u16) << 7 | (message[1] as u16);
                println!("Pitch Bnd - Channel: {:2}, Value: {:5} (raw: {:3}, {:3})", 
                        channel, bend_value, message[1], message[2]);
            } else {
                println!("Pitch Bnd - Incomplete message: {:?}", message);
            }
        }
        0xF0 => {
            // System messages
            match status {
                0xF0 => println!("SysEx Start: {:?}", message),
                0xF1 => println!("MTC Quarter Frame: {:?}", message),
                0xF2 => println!("Song Position: {:?}", message),
                0xF3 => println!("Song Select: {:?}", message),
                0xF6 => println!("Tune Request: {:?}", message),
                0xF7 => println!("SysEx End: {:?}", message),
                0xF8 => println!("Timing Clock: {:?}", message),
                0xFA => println!("Start: {:?}", message),
                0xFB => println!("Continue: {:?}", message),
                0xFC => println!("Stop: {:?}", message),
                0xFE => println!("Active Sensing: {:?}", message),
                0xFF => println!("System Reset: {:?}", message),
                _ => println!("Unknown System: {:?}", message),
            }
        }
        _ => {
            println!("Unknown MIDI message: {:?}", message);
        }
    }
}
