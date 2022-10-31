pub mod primitive;
pub mod mmu;
pub mod emulator;

use emulator::{Register, Emulator, VmExit};
use mmu::{VirtAddr, PERM_WRITE, PERM_EXEC, PERM_READ, Perm, Section};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

fn handle_syscall(emu: &mut Emulator) -> Result<(), VmExit> {
    // Get the syscall number
    let num = emu.reg(Register::A7);

    match num {
        96 => {
            // set_tid_address(), just return the TID
            emu.set_reg(Register::A0, 1337);
            Ok(())
        }
        29 => {
            // ioctl()
            emu.set_reg(Register::A0, !0);
            Ok(())
        }
        66 => {
            // writev()
            let fd = emu.reg(Register::A0);
            let iov = emu.reg(Register::A1);
            let iovcnt = emu.reg(Register::A2);

            // We currently only handle stdout and stderr
            if fd != 1  && fd != 2 {
                // Return error
                emu.set_reg(Register::A0, !0);
                return Ok(());
            }

            let mut bytes_written = 0;

            for idx in 0..iovcnt {
                // Compute the pointer the IO vector entry corresponding to this index
                // and validate that is will not overflow pointer size for the size of
                // the `_iovec`
                let ptr = 16u64.checked_mul(idx)
                    .and_then(|x| x.checked_add(iov))
                    .and_then(|x| x.checked_add(15))
                    .ok_or(VmExit::SyscallIntegerOverflow)? as usize - 15;

                // Read the iovec entry pointer and length
                let buf: usize = emu.memory.read(VirtAddr(ptr + 0))?;
                let len: usize = emu.memory.read(VirtAddr(ptr + 8))?;

                // Look at the buffer!
                let data = emu.memory.peek_perms(VirtAddr(buf), len, Perm(PERM_READ))?;

                /*
                if let Ok(st) = core::str::from_utf8(data) {
                    print!("{}", st);
                }
                */

                // Update number of bytes writen
                bytes_written += len as u64;
            }

            // Return number of bytes written
            emu.set_reg(Register::A0, bytes_written);
            Ok(())
        }
        94 => {
            Err(VmExit::Exit)
        }
        _ => {
            panic!("Unhandled syscall {}\n", num);
        }
    }
}

/// Statistics during fuzzing
#[derive(Default)]
struct Statistics {
    fuzz_cases: AtomicU64,
}

fn worker(mut emu: Emulator, original: Arc<Emulator>, stats: Arc<Statistics>) {
    loop {
        // Reset emu to original state
        emu.reset(&*original);

        let vmexit = loop {
            let vmexit = emu.run().expect_err("Failed to execute emulator");
            match vmexit {
                VmExit::Syscall => {
                    if let Err(vmexit) = handle_syscall(&mut emu) {
                        break vmexit;
                    }
                    // Advance PC(always)
                    let pc = emu.reg(Register::Pc);
                    emu.set_reg(Register::Pc, pc.wrapping_add(4));
                }
                _ => break vmexit,
            }
        };

        stats.fuzz_cases.fetch_add(1, Ordering::Relaxed);
        //print!("Vm exit with {:#x?}\n", vmexit);
    }
}

fn main() {
    // Make an emulator with 1 Meg of memory
    let mut emu = Emulator::new(32 * 1024 * 1024);

    emu.memory.load("./test_app", &[
        Section {
            file_off: 0x0000000000000000,
            virt_addr: VirtAddr(0x0000000000010000),
            file_size: 0x0000000000000190,
            mem_size: 0x0000000000000190,
            permissions: Perm(PERM_READ),
        },
        Section {
            file_off: 0x0000000000000190,
            virt_addr: VirtAddr(0x0000000000011190),
            file_size: 0x00000000000020fc,
            mem_size: 0x00000000000020fc,
            permissions: Perm(PERM_EXEC),
        },
        Section {
            file_off: 0x0000000000002290,
            virt_addr: VirtAddr(0x0000000000014290),
            file_size: 0x0000000000000108,
            mem_size: 0x0000000000000760,
            permissions: Perm(PERM_READ | PERM_WRITE),
        },
    ]).expect("Failed to load test application into address space");

    // Set the program entry point
    emu.set_reg(Register::Pc, 0x11190);

    // Set up a stack
    let stack = emu.memory.allocate(32 * 1024).expect("Failed to allocate stack");
    // Go to the top of the stack
    emu.set_reg(Register::Sp, stack.0 as u64 + 32 * 1024);

    // Set up null terminated arg vectors
    let argv = emu.memory.allocate(8).expect("Failed to allocate argv");
    emu.memory.write_from(argv, b"test\0").expect("Failed to null-terminate argv");

    macro_rules! push {
        ($expr:expr) => {
            let sp = emu.reg(Register::Sp) - core::mem::size_of_val(&$expr) as u64;
            emu.memory.write(VirtAddr(sp as usize), $expr).expect("Push failed");
            emu.set_reg(Register::Sp, sp);
        }
    }

    // Allocate 32Kb and set up the stack
    // Push arguments from last to first on the stack
    push!(0u64); // Auxp
    push!(0u64); // Envp
    push!(0u64); // Argv end
    push!(argv.0); // Argv first argument
    push!(0u64); // Argc

    use std::time::{Duration, Instant};

    // Start a timer
    let start = Instant::now();

    // Wrap the original emulator in an `Arc`
    let emu = Arc::new(emu);

    // Create a new stats structure
    let stats = Arc::new(Statistics::default());

    for _ in 0..12 {
        let new_emu= emu.fork();
        let stats = stats.clone();
        let parent = emu.clone();
        std::thread::spawn(move || {
            worker(new_emu, parent, stats);
        });
    }

    loop {
        std::thread::sleep(Duration::from_millis(1000));
        let elapsed = start.elapsed().as_secs_f64();
        let fuzz_cases = stats.fuzz_cases.load(Ordering::Relaxed);
        print!("[{:10.4}] Fuzz cases {:10} | fcps {:10.2}\n", elapsed, fuzz_cases,
            fuzz_cases as f64 / elapsed);
    }
}
