pub mod primitive;
pub mod mmu;
pub mod emulator;
pub mod jitcache;

use emulator::{Register, Emulator, VmExit, File};
use mmu::{VirtAddr, PERM_WRITE, PERM_EXEC, PERM_READ, Perm, Section};
use std::sync::{Arc, Mutex};
use jitcache::JitCache;

/// If true, the guest writes to stdout and std err will be printed to our own stdout and stderr
const VERBOSE_GUEST_PRINTS: bool = false;

fn rdtsc() -> u64 {
    unsafe { std::arch::x86_64::_rdtsc() }
}

fn handle_syscall(emu: &mut Emulator) -> Result<(), VmExit> {
    // Get the syscall number
    let num = emu.reg(Register::A7);

    print!("Syscall {}\n", num);

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
            let fd = emu.reg(Register::A0) as usize;
            let buf = emu.reg(Register::A1);
            let len = emu.reg(Register::A2);

            let file = emu.files.get_file(fd);

            if let Some(Some(file)) = file {
                if file == &File::Stdout || file == &File::Stderr {
                    // Write to stdout and stderr

                    // Get access to the underlying bytes to write
                    let bytes = emu
                        .memory
                        .peek(VirtAddr(buf as usize), len as usize, Perm(PERM_READ))?;

                    if VERBOSE_GUEST_PRINTS {
                        if let Ok(st) = core::str::from_utf8(bytes) {
                            print!("{}", st);
                        }
                    }

                    // Set that all byte were read in the return value
                    emu.set_reg(Register::A0, len);
                } else {
                    panic!("Write to valid but unhandled FD");
                }
            } else {
                // Unknown FD
                emu.set_reg(Register::A0, !0);
            }

            Ok(())
        }
        63 => {
            // read()
            let fd = emu.reg(Register::A0) as usize;
            let buf = emu.reg(Register::A1) as usize;
            let len = emu.reg(Register::A2) as usize;

            // Check if the FD is valid
            let file = emu.files.get_file(fd);
            if file.is_none() || file.as_ref().unwrap().is_none() {
                // FD was not valid, return out with an error
                emu.set_reg(Register::A0, !0);
                return Ok(());
            }


            if let Some(Some(File::FuzzInput { ref mut cursor })) = file {
                // Compute the ending cursor from this read
                let result_cursor = core::cmp::min(
                    cursor.saturating_add(len),
                    emu.fuzz_input.len(),
                );

                // Write in the bytes
                emu.memory.write_from(
                    VirtAddr(buf),
                    &emu.fuzz_input[*cursor..result_cursor],
                )?;

                // Compute bytes read
                let bread = result_cursor - *cursor;

                // Update the cursor
                *cursor = result_cursor;

                // Return number of bytes read
                emu.set_reg(Register::A0, bread as u64);
            } else {
                unreachable!();
            }

            Ok(())
        }
        62 => {
            // lseek()
            let fd= emu.reg(Register::A0) as usize;
            let offset = emu.reg(Register::A1) as i64;
            let whence = emu.reg(Register::A2) as i32;

            const SEEK_SET: i32 = 0;
            const SEEK_CUR: i32 = 1;
            const SEEK_END: i32 = 2;

            // Check if the FD is valid
            let file = emu.files.get_file(fd);
            if file.is_none() || file.as_ref().unwrap().is_none() {
                // FD was not valid, return out with an error
                emu.set_reg(Register::A0, !0);
                return Ok(());
            }

            if let Some(Some(File::FuzzInput { ref mut cursor })) = file {
                let new_cursor = match whence {
                    SEEK_SET => offset,
                    SEEK_CUR => (*cursor as i64).saturating_add(offset),
                    SEEK_END => (emu.fuzz_input.len() as i64).saturating_add(offset),
                    _ => {
                        // Invalid whence
                        emu.set_reg(Register::A0, !0);
                        return Ok(());
                    }
                };

                // Make sure the cursor falls in bounds of [0, file_size]
                let new_cursor = core::cmp::max(0, new_cursor);
                let new_cursor = core::cmp::min(new_cursor, emu.fuzz_input.len() as i64);

                // Update the cursor
                *cursor = new_cursor as usize;

                // Return the new cursor position
                emu.set_reg(Register::A0, new_cursor as u64);
            } else {
                unreachable!();
            }

            Ok(())
        }
        1024 => {
            // open()
            let filename = emu.reg(Register::A0) as usize;
            let flags = emu.reg(Register::A1);
            let _mode = emu.reg(Register::A2);

            assert!(flags == 0, "Currently we only handle 0 RDONLY");

            // Determine the length of the filename
            let mut fnlen = 0;
            while emu.memory.read::<u8>(VirtAddr(filename + fnlen))? != 0 {
                fnlen += 1;
            }

            let bytes = emu.memory.peek(VirtAddr(filename), fnlen, Perm(PERM_READ))?;

            if bytes == b"test_app" {
                // Create a new file descriptor
                let fd = emu.alloc_file();

                // Get access to the file, unwrap here is safe because there's no way the file is
                // not a valid FD if we hot it from our own APIs
                let file = emu.files.get_file(fd).unwrap();
                //print!("Allocated FD {}\n", fd);

                // Mark that this file be backed by our fuzz input
                *file = Some(File::FuzzInput { cursor: 0 });

                // Return a new fd
                emu.set_reg(Register::A0, fd as u64);
            } else {
                print!("Unknon Filename is {:?}\n", core::str::from_utf8(bytes));
                // Unknown filename
                emu.set_reg(Register::A0, !0);
            }

            Ok(())
        }
        80 => {
            // fstat()
            let fd = emu.reg(Register::A0) as usize;
            let statbuf = emu.reg(Register::A1);

            /// Stat structure from kernel stat64
            #[repr(C)]
            #[derive(Default, Debug)]
            struct Stat {
                st_dev: u64,
                st_ino: u64,
                st_mode: u32,
                st_nlink: u32,
                st_uid: u32,
                st_gid: u32,
                st_rdev: u64,
                __pad1: u64,

                st_size: i64,
                st_blksize: i32,
                __pad2: i32,

                st_blocks: i64,

                st_atime: u64,
                st_atimensec: u64,
                st_mtime: u64,
                st_mtimensec: u64,
                st_ctime: u64,
                st_ctimensec: u64,

                __glibc_reserved: [i32; 2],
            }

            // Check if the FD is valid
            let file = emu.files.get_file(fd);
            if file.is_none() || file.as_ref().unwrap().is_none() {
                // FD was not valid, return out with an error
                emu.set_reg(Register::A0, !0);
                return Ok(());
            }

            if let Some(Some(File::FuzzInput { .. })) = file {
                let mut stat = Stat::default();
                stat.st_dev = 0x803;
                stat.st_ino = 0x81889;
                stat.st_mode = 0x81a4;
                stat.st_nlink = 0x1;
                stat.st_uid = 0x3e8;
                stat.st_gid = 0x3e8;
                stat.st_rdev = 0x0;
                stat.st_size = emu.fuzz_input.len() as i64;
                stat.st_blksize = 0x1000;
                stat.st_blocks = (emu.fuzz_input.len() as i64 + 511) / 512;
                stat.st_atime = 0x5f0fe246;
                stat.st_mtime = 0x5f0fe244;
                stat.st_ctime = 0x5f0fe244;

                // Cast the stat structure to raw bytes
                let stat = unsafe {
                    core::slice::from_raw_parts(&stat as *const Stat as *const u8,
                        core::mem::size_of_val(&stat))
                };

                // Write in the stat data
                emu.memory.write_from(VirtAddr(statbuf as usize), stat)?;

            } else {
                // Error
                emu.set_reg(Register::A0, !0);
            }

            Ok(())
        }
        57 => {
            // close()
            let fd = emu.reg(Register::A0) as usize;

            if let Some(file) = emu.files.get_file(fd) {
                if file.is_some() {
                    // File was present and currently open, close it

                    // Close the file
                    *file = None;

                    // Just return success for now
                    emu.set_reg(Register::A0, 0);
                } else {
                    // File was in a closed state
                    emu.set_reg(Register::A0, !0);
                }
            } else {
                emu.set_reg(Register::A0, !0);

            }

            Ok(())
        }
        93 => {
            // Exit()
            Err(VmExit::Exit)
        }
        _ => {
            panic!("Unhandled syscall {} as {:#x}\n", num, emu.reg(Register::Pc));
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
    const BATCH_SIZE: usize = 1;
    loop {
        let batch_start = rdtsc();

        let mut local_stats = Statistics::default();

        for _ in 0..BATCH_SIZE {
            // Reset emu to original state
            let it = rdtsc();
            emu.reset(&*original);
            local_stats.reset_cycles += rdtsc() - it;

            let vmexit = loop {
                let it = rdtsc();
                let vmexit = emu.run(&mut local_stats.inst_exec).expect_err("Failed to execute emulator");
                local_stats.vm_cycles += rdtsc() - it;

                emu.fuzz_input.extend_from_slice(include_bytes!("../xauth"));

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

            if vmexit != VmExit::Exit {
                panic!("Vmexit {:#x} {:?}\n", emu.reg(Register::Pc), vmexit);
            }
            local_stats.fuzz_cases += 1;
        }

        let mut stats = stats.lock().unwrap();


        stats.fuzz_cases += local_stats.fuzz_cases;
        stats.inst_exec+= local_stats.inst_exec;
        stats.reset_cycles += local_stats.reset_cycles;
        stats.vm_cycles += local_stats.vm_cycles;

        let batch_end = rdtsc() - batch_start;
        stats.total_cycles += batch_end;
    }
}

fn main() {
    // Create a JIT Cache
    let jit_cache = Arc::new(JitCache::new(VirtAddr(1024 * 1024)));

    // Create an emulator using the JIT
    let mut emu = Emulator::new((32 * 1024 * 1024)).enable_jit(jit_cache);

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
    let progname = emu.memory.allocate(4096).expect("Failed to allocate argv");
    emu.memory.write_from(progname, b"objdump\0").expect("Failed to null-terminate argv");

    let arg1 = emu.memory.allocate(4096).expect("Failed to allocate arg1");
    emu.memory.write_from(arg1, b"-x\0").expect("Failed to null-terminate arg1");

    let arg2 = emu.memory.allocate(4096).expect("Failed to allocate arg2");
    emu.memory.write_from(arg2, b"test_app\0").expect("Failed to null-terminate arg2");

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
    push!(arg2.0); // Argv first argument
    push!(arg1.0); // Argv first argument
    push!(progname.0); // Argv first argument
    push!(3u64); // Argc

    use std::time::{Duration, Instant};

    // Wrap the original emulator in an `Arc`
    let emu = Arc::new(emu);

    // Create a new stats structure
    let stats = Arc::new(Mutex::new(Statistics::default()));

    for _ in 0..1 {
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
    let _start_cycles = rdtsc();

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
