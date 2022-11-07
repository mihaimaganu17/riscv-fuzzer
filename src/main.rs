pub mod primitive;
pub mod mmu;
pub mod emulator;

use emulator::{Register, Emulator, VmExit};
use mmu::{VirtAddr, PERM_WRITE, PERM_EXEC, PERM_READ, Perm, Section};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// If true, the guest writes to stdout and std err will be printed to our own stdout and stderr
const VERBOSE_GUEST_PRINTS: bool = false;

fn rdtsc() -> u64 {
    unsafe { std::arch::x86_64::_rdtsc() }
}

fn handle_syscall(emu: &mut Emulator) -> Result<(), VmExit> {
    // Get the syscall number
    let num = emu.reg(Register::A7);

    match num {
        214 => {
            let req_base = emu.reg(Register::A0);
            let cur_base = emu.memory.allocate(0).unwrap();


            let increment = if req_base != 0 {
                (req_base as i64).checked_sub(cur_base.0 as i64)
                    .ok_or(VmExit::SyscallIntegerOverflow)?
            } else {
                0
            };

            // We don't handle negative brk()'s
            assert!(increment >= 0);

            // Atempt to extend data section by increment
            if let Some(_base) = emu.memory.allocate(increment as usize) {
                let new_base = cur_base.0 + increment as usize;
                emu.set_reg(Register::A0, new_base as u64);
            } else {
                emu.set_reg(Register::A0, !0);
            }

            Ok(())
        }
        64 => {
            // write()
            let fd = emu.reg(Register::A0);
            let buf = emu.reg(Register::A1);
            let len = emu.reg(Register::A2);

            if fd == 1 || fd == 2 {
                // Write to stdout and stderr

                // Get access to the underlying bytes to write
                let bytes = emu.memory.peek(VirtAddr(buf as usize), len as usize, Perm(PERM_READ))?;

                if VERBOSE_GUEST_PRINTS {
                    if let Ok(st) = core::str::from_utf8(bytes) {
                        print!("{}", st);
                    }
                }

                // Set that all byte were read in the return value
                emu.set_reg(Register::A0, len);
            } else {
                // Unknown FD
                emu.set_reg(Register::A0, !0);
            }

            Ok(())
        }
        57 => {
            // close()
            // Just return success for now
            emu.set_reg(Register::A0, 0);
            Ok(())
        }
        93 => {
            // Exit()
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
    /// Number of fuzz cases
    fuzz_cases: u64,
    /// Number of instructions executred
    inst_exec: u64,
    /// Total number of CPU cycles spent inthe workers
    total_cycles: u64,
    /// Total number of CPU cycles spent resetting the guest
    reset_cycles: u64,
    /// Total number of CPU cycles spent emulating
    vm_cycles: u64,
}

fn worker(mut emu: Emulator, original: Arc<Emulator>, stats: Arc<Mutex<Statistics>>) {
    const BATCH_SIZE: usize = 100;
    loop {
        let batch_start = rdtsc();

        let mut local_stats = Statistics::default();

        for _ in 0..BATCH_SIZE {
            // Reset emu to original state
            let it = rdtsc();
            emu.reset(&*original);
            local_stats.reset_cycles += (rdtsc() - it);

            let vmexit = loop {
                let it = rdtsc();
                let vmexit = emu.run(&mut local_stats.inst_exec).expect_err("Failed to execute emulator");
                local_stats.vm_cycles += (rdtsc() - it);

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

            //print!("Vmexit {:#x} {:?}\n", emu.reg(Register::Pc), vmexit);
            local_stats.fuzz_cases += 1;
        }

        let mut stats = stats.lock().unwrap();


        stats.fuzz_cases += local_stats.fuzz_cases;
        stats.inst_exec+= local_stats.inst_exec;
        stats.reset_cycles += local_stats.reset_cycles;
        stats.vm_cycles += local_stats.vm_cycles;

        //print!("Vm exit with {:#x?}\n", vmexit);
        let batch_end = rdtsc() - batch_start;
        stats.total_cycles += batch_end;
    }
}

fn main() {
    // Make an emulator with 1 Meg of memory
    let mut emu = Emulator::new(32 * 1024 * 1024);

    emu.memory.load("./objdump", &[
        Section {
            file_off: 0x0000000000000000,
            virt_addr: VirtAddr(0x0000000000010000),
            file_size: 0x00000000000e1a44,
            mem_size: 0x00000000000e1a44,
            permissions: Perm(PERM_READ | PERM_EXEC),
        },
        Section {
            file_off: 0x00000000000e2000,
            virt_addr: VirtAddr(0x00000000000f2000),
            file_size: 0x0000000000001e32,
            mem_size: 0x00000000000046c8,
            permissions: Perm(PERM_READ | PERM_WRITE),
        },
    ]).expect("Failed to load test application into address space");

    println!("Food baby");

    // Set the program entry point
    emu.set_reg(Register::Pc, 0x10554);

    // Set up a stack
    let stack = emu.memory.allocate(32 * 1024).expect("Failed to allocate stack");
    // Go to the top of the stack
    emu.set_reg(Register::Sp, stack.0 as u64 + 32 * 1024);

    // Set up null terminated arg vectors
    let argv = emu.memory.allocate(4096).expect("Failed to allocate argv");
    emu.memory.write_from(argv, b"objdump\0").expect("Failed to null-terminate argv");

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
    push!(1u64); // Argc

    use std::time::{Duration, Instant};

    // Wrap the original emulator in an `Arc`
    let emu = Arc::new(emu);

    // Create a new stats structure
    let stats = Arc::new(Mutex::new(Statistics::default()));

    for _ in 0..4 {
        let new_emu= emu.fork();
        let stats = stats.clone();
        let parent = emu.clone();
        std::thread::spawn(move || {
            worker(new_emu, parent, stats);
        });
    }

    // Start a timer
    let start = Instant::now();

    // Save the time stamp of start of execution
    let start_cycles = rdtsc();

    let mut last_cases = 0;
    let mut last_inst = 0;
    let mut last_time = Instant::now();

    loop {
        std::thread::sleep(Duration::from_millis(1000));
        let stats = stats.lock().unwrap();

        let time_delta = last_time.elapsed().as_secs_f64();
        let elapsed = start.elapsed().as_secs_f64();
        let fuzz_cases = stats.fuzz_cases;
        let instrs = stats.inst_exec;

        // Compute performance numbers
        let resetc = stats.reset_cycles as f64 / stats.total_cycles as f64;
        let vmc = stats.vm_cycles as f64 / stats.total_cycles as f64;
        print!("[{:10.4}] Fuzz cases {:10} | fcps {:10.1} | inst/sec {:10.1}\n\
            reset {:8.4} | vm {:8.4}\n",
            elapsed, fuzz_cases,
            (fuzz_cases - last_cases) as f64 / time_delta,
            (instrs - last_inst) as f64 / time_delta, resetc, vmc);
        last_cases = fuzz_cases;
        last_inst = instrs;
        last_time = Instant::now();
    }
}
