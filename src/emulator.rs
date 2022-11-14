//! A 64-bit RISC-V RV64i interpreter

use crate::mmu::{Mmu, Perm, VirtAddr, PERM_READ, PERM_WRITE, PERM_RAW, PERM_EXEC, DIRTY_BLOCK_SIZE};
use crate::jitcache::JitCache;
use std::sync::Arc;
use std::fmt;
use std::process::Command;
use std::arch::asm;

/// An R-type instruction
#[derive(Debug)]
struct Rtype {
    funct7: u32,
    rs2: Register,
    rs1: Register,
    funct3: u32,
    rd: Register,
}

impl From<u32> for Rtype {
    fn from(inst: u32) -> Self {
        Rtype {
            funct7: (inst >> 25) & 0b1111111,
            rs2: Register::from((inst >> 20) & 0b11111),
            rs1: Register::from((inst >> 15) & 0b11111),
            funct3: (inst >> 12) & 0b111,
            rd: Register::from((inst >> 7) & 0b11111),
        }
    }
}

/// An S-type instruction
#[derive(Debug)]
struct Stype {
    imm: i32,
    rs2: Register,
    rs1: Register,
    funct3: u32,
}

impl From<u32> for Stype {
    fn from(inst: u32) -> Self {
        let imm115 = (inst >> 25) & 0b1111111;
        let imm40 = (inst >> 7) & 0b11111;

        let imm = (imm115 << 5) | imm40;
        let imm = ((imm as i32) << 20) >> 20;

        Stype {
            imm: imm,
            rs2: Register::from((inst >> 20) & 0b11111),
            rs1: Register::from((inst >> 15) & 0b11111),
            funct3: (inst >> 12) & 0b111,
        }
    }
}

/// A J-type instruction
#[derive(Debug)]
struct Jtype {
    imm: i32,
    rd: Register,
}

impl From<u32> for Jtype {
    fn from(inst: u32) -> Self {
        let imm20 = (inst >> 31) & 1;
        let imm101 = (inst >> 21) & 0b1111111111;
        let imm11 = (inst >> 20) & 1;
        let imm1912 = (inst >> 12) & 0b11111111;

        let imm = (imm20 << 20) | (imm1912 << 12) | (imm11 << 11) | (imm101 << 1);
        let imm = ((imm as i32) << 11) >> 11;

        Jtype {
            imm: imm,
            rd: Register::from((inst >> 7) & 0b11111),
        }
    }
}

/// A B-type instruction
#[derive(Debug)]
struct Btype {
    imm: i32,
    rs2: Register,
    rs1: Register,
    funct3: u32,
}

impl From<u32> for Btype {
    fn from(inst: u32) -> Self {
        let imm12 = (inst >> 31) & 1;
        let imm105 = (inst >> 25) & 0b111111;
        let imm41 = (inst >> 8) & 0b1111;
        let imm11 = (inst >> 7) & 1;

        let imm = (imm12 << 12) | (imm11 << 11) | (imm105 << 5) | (imm41 << 1);
        let imm = ((imm as i32) << 19) >> 19;

        Btype {
            imm: imm,
            rs2: Register::from((inst >> 20) & 0b11111),
            rs1: Register::from((inst >> 15) & 0b11111),
            funct3: (inst >> 12) & 0b111,
        }
    }
}

/// An I-type instruction
#[derive(Debug)]
struct Itype {
    imm: i32,
    rs1: Register,
    funct3: u32,
    rd: Register,
}

impl From<u32> for Itype {
    fn from(inst: u32) -> Self {
        let imm = (inst as i32) >> 20;
        Itype {
            imm: imm,
            rs1: Register::from((inst >> 15) & 0b11111),
            funct3: (inst >> 12) & 0b111,
            rd: Register::from((inst >> 7) & 0b11111),
        }
    }
}

#[derive(Debug)]
struct Utype {
    imm: i32,
    rd: Register,
}

impl From<u32> for Utype {
    fn from(inst: u32) -> Self {
        Utype {
            imm: (inst & !0xfff) as i32,
            rd: Register::from((inst >> 7) & 0b11111),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum File {
    Stdin,
    Stdout,
    Stderr,

    // A file which is 
    FuzzInput { cursor: usize },
}

/// A list of all open files
#[derive(Clone, Debug, PartialEq)]
pub struct Files(Vec<Option<File>>);

impl Files {
    /// Get access to a file descriptor for `fd`
    pub fn get_file(&mut self, fd: usize) -> Option<&mut Option<File>> {
        self.0.get_mut(fd)
    }
}

/// All the state of the emulated system
pub struct Emulator {
    /// Memory for the emulator
    pub memory: Mmu,

    /// All RV64i registers
    registers: [u64; 33],

    /// Fuzz input for the program
    pub fuzz_input: Vec<u8>,

    /// File Hande table (indexed by file descriptor)
    pub files: Files,

    /// JIT cache, if we are using a JIT
    jit_cache: Option<Arc<JitCache>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// Reasons why the VM exited
pub enum VmExit {
    /// The VM exited due to a syscall instruction
    Syscall,

    /// The VM exited cleanly as requested by the VM
    Exit,

    /// A branch occured to a location outside of the JIT cache region
    JitOob,

    /// An integer overflow occured during a syscall due to bad supplied
    /// arguments by the program
    SyscallIntegerOverflow,

    /// A read or write memory request overflowed the address size
    AddressIntegerOverflow,

    /// The address requested was not in bounds of the guest memory space
    AddressMiss(VirtAddr, usize),

    /// An read of `VirtAddr` failed due to missing permissions
    ReadFault(VirtAddr),

    /// A read memory which is uninitialized, but otherwise readable failed at `VirtAddr`
    UninitFault(VirtAddr),

    /// An write of `VirtAddr` failed due to missing permissions
    WriteFault(VirtAddr),
}

impl fmt::Display for Emulator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            r#"zero {:016x} ra {:016x} sp  {:016x} gp  {:016x}
tp   {:016x} t0 {:016x} t1  {:016x} t2  {:016x}
s0   {:016x} s1 {:016x} a0  {:016x} a1  {:016x}
a2   {:016x} a3 {:016x} a4  {:016x} a5  {:016x}
a6   {:016x} a7 {:016x} s2  {:016x} s3  {:016x}
s4   {:016x} s5 {:016x} s6  {:016x} s7  {:016x}
s8   {:016x} s9 {:016x} s10 {:016x} s11 {:016x}
t3   {:016x} t4 {:016x} t5  {:016x} t6  {:016x}
pc   {:016x}"#,
            self.reg(Register::Zero),
            self.reg(Register::Ra),
            self.reg(Register::Sp),
            self.reg(Register::Gp),
            self.reg(Register::Tp),
            self.reg(Register::T0),
            self.reg(Register::T1),
            self.reg(Register::T2),
            self.reg(Register::S0),
            self.reg(Register::S1),
            self.reg(Register::A0),
            self.reg(Register::A1),
            self.reg(Register::A2),
            self.reg(Register::A3),
            self.reg(Register::A4),
            self.reg(Register::A5),
            self.reg(Register::A6),
            self.reg(Register::A7),
            self.reg(Register::S2),
            self.reg(Register::S3),
            self.reg(Register::S4),
            self.reg(Register::S5),
            self.reg(Register::S6),
            self.reg(Register::S7),
            self.reg(Register::S8),
            self.reg(Register::S9),
            self.reg(Register::S10),
            self.reg(Register::S11),
            self.reg(Register::T3),
            self.reg(Register::T4),
            self.reg(Register::T5),
            self.reg(Register::T6),
            self.reg(Register::Pc)
        )
    }
}

/// 64-bit RISC-V registers
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(usize)]
pub enum Register {
    Zero = 0,
    Ra,
    Sp,
    Gp,
    Tp,
    T0,
    T1,
    T2,
    S0,
    S1,
    A0,
    A1,
    A2,
    A3,
    A4,
    A5,
    A6,
    A7,
    S2,
    S3,
    S4,
    S5,
    S6,
    S7,
    S8,
    S9,
    S10,
    S11,
    T3,
    T4,
    T5,
    T6,
    Pc,
}

impl From<u32> for Register {
    fn from(val: u32) -> Self {
        assert!(val < 32);
        unsafe { core::ptr::read_unaligned(&(val as usize) as *const usize as *const Register) }
    }
}

impl Emulator {
    /// Creates a new emulator with `size` bytes of memory
    pub fn new(size: usize) -> Self {
        Emulator {
            memory: Mmu::new(size),
            registers: [0; 33],
            fuzz_input: Vec::new(),
            files: Files(vec![
                Some(File::Stdin),
                Some(File::Stdout),
                Some(File::Stderr),
            ]),
            jit_cache: None,
        }
    }

    /// Fork an emulator into a new emulator which will diff from the original
    pub fn fork(&self) -> Self {
        Emulator {
            memory: self.memory.fork(),
            registers: self.registers.clone(),
            fuzz_input: self.fuzz_input.clone(),
            files: self.files.clone(),
            jit_cache: self.jit_cache.clone(),
        }
    }

    /// Enable the JIT and use a specified JitCache
    pub fn enable_jit(mut self, jit_cache: Arc<JitCache>) -> Self {
        self.jit_cache = Some(jit_cache);
        self
    }

    /// Reset the state of `self` to `other`, assuming that `self` is
    /// forked off of `other`. If it is not, the results are invalid.
    pub fn reset(&mut self, other: &Self) {
        // Reset memory state
        self.memory.reset(&other.memory);

        // Reset register state
        self.registers = other.registers;

        // Reset file state
        self.files.0.clear();
        self.files.0.extend_from_slice(&other.files.0);
    }

    /// Get access to a file descriptor for `fd`
    pub fn get_file(&mut self, fd: usize) -> Option<&mut Option<File>> {
        self.files.0.get_mut(fd)
    }

    /// Allocate a new file descriptor
    pub fn alloc_file(&mut self) -> usize {
        for (fd, file) in self.files.0.iter().enumerate() {
            if file.is_none() {
                // File not present, we can reuse the FD
                return fd;
            }
        }

        // If we got here, no FD is present, create a new one
        let fd = self.files.0.len();
        self.files.0.push(None);
        fd
    }

    /// Get a register from the guest
    pub fn reg(&self, register: Register) -> u64 {
        if register != Register::Zero {
            self.registers[register as usize]
        } else {
            0
        }
    }

    /// Set a register in the guest
    pub fn set_reg(&mut self, register: Register, val: u64) {
        if register != Register::Zero {
            self.registers[register as usize] = val;
        }
    }

    /// Run the VM using either the emulator or the JIT
    pub fn run(&mut self, instrs_execed: &mut u64) -> Result<(), VmExit> {
        if self.jit_cache.is_some() {
            self.run_jit(instrs_execed)
        } else {
            self.run_emu(instrs_execed)
        }
    }

    /// Run the VM using the JIT
    pub fn run_jit(&mut self, instrs_execed: &mut u64) -> Result<(), VmExit> {
        let (memory, perms, dirty, dirty_bitmap) = self.memory.jit_addrs();
        // Get the translation table
        let trans_table = self.jit_cache.as_ref().unwrap().translation_table();

        loop {
            // Get the current PC
            let pc = self.reg(Register::Pc);
            let (jit_addr, num_blocks) = {
                let jit_cache = self.jit_cache.as_ref().unwrap();
                (
                    jit_cache.lookup(VirtAddr(pc as usize)),
                    jit_cache.num_blocks(),
                )
            };

            let jit_addr = if let Some(jit_addr) = jit_addr {
                jit_addr
            } else {
                // Go through each instruction in the block, and accumulate an assembly string
                // which we will assemble using `nasm` on the command line
                let asm = self.generate_jit(VirtAddr(pc as usize), num_blocks)?;

                // Write out the assembly
                let asmfn = std::env::temp_dir()
                    .join(format!("fwetmp_{:?}.asm", std::thread::current().id()));
                let binfn = std::env::temp_dir()
                    .join(format!("fwetmp_{:?}.bin", std::thread::current().id()));
                std::fs::write(&asmfn, &asm).expect("Failed to write out asm");

                // Invoke NASM to generate the binary
                let res = Command::new("nasm")
                    .args(&["-f", "bin", "-o",
                          binfn.to_str().unwrap(),
                          asmfn.to_str().unwrap()]).status()
                    .expect("Failed to run `nasm`, is it in you path?");
                assert!(res.success(), "nasm returned an error");

                // Read the binary
                let tmp = std::fs::read(&binfn)
                    .expect("Failed to read nasm output");

                // Update the JIT tables
                self.jit_cache.as_ref().unwrap().add_mapping(VirtAddr(pc as usize), &tmp)
            };


            unsafe {
                // Invoke the JIT
                let exit_code: u64;
                let reentry_pc: u64;
                let exit_info: u64;

                let dirty_inuse = self.memory.dirty_len();
                let new_dirty_inuse: usize;
                let mut instcount = *instrs_execed;

                asm!(r#"
                    call {entry}
                "#,
                // We discard these outputs(rax, rbx, rcx)
                entry = in(reg) jit_addr,
                out("rax") exit_code,
                out("rdx") reentry_pc,
                out("rcx") exit_info,
                out("rdi") _,
                in("r8") memory,
                in("r9") perms,
                in("r10") dirty,
                in("r11") dirty_bitmap,
                inout("r12") dirty_inuse => new_dirty_inuse,
                in("r13") self.registers.as_ptr(),
                in("r14") trans_table,
                inout("r15") instcount,
                );

                // Update the PC reentry point
                self.set_reg(Register::Pc, reentry_pc);

                // Update insts execed
                *instrs_execed = instcount;

                // Update the dirty state
                self.memory.set_dirty_len(new_dirty_inuse);

                match exit_code {
                    1 => {
                        // Branch decode request, just continue as PC has been updated to the new
                        // target
                    }
                    2 => {
                        // Syscall
                        return Err(VmExit::Syscall);
                    }
                    4 => {
                        return Err(VmExit::ReadFault(VirtAddr(exit_info as usize)));
                    }
                    5 => {
                        return Err(VmExit::WriteFault(VirtAddr(exit_info as usize)));
                    }
                    _ => unreachable!(),
                }

            }
        }
    }

    /// Generates the assembly string for `pc` during JIT
    pub fn generate_jit(&self, pc: VirtAddr, num_blocks: usize) -> Result<String, VmExit> {
        let mut asm = "[bits 64]\n".to_string();

        let mut pc = pc.0 as u64;

        let mut block_instrs = 0;

        'next_inst: loop {
            // Get the current program counter
            let inst: u32 = self
                .memory
                .read_perms(VirtAddr(pc as usize), Perm(PERM_EXEC))?;

            // Extract the opcode from the instruction
            let opcode = inst & 0b1111111;

            // Add a lable to this instruction
            asm += &format!("inst_pc_{:#x}:\n", pc);

            // Track number of instructions in the block
            block_instrs += 1;

            // Produce the assembly statement to load RISC-V `reg` into `x86` reg
            macro_rules! load_reg {
                ($x86reg:expr, $reg:expr) => {
                    if $reg == Register::Zero {
                        format!("xor {x86reg}, {x86reg}\n", x86reg = $x86reg)
                    } else {
                        format!("mov {x86reg}, qword [r13 + {reg} * 8]\n",
                                x86reg = $x86reg, reg = $reg as usize)
                    }
                }
            }

            // Produce the assembly statement to store RISC-V `reg` into `x86` reg
            macro_rules! store_reg {
                ($reg:expr, $x86reg:expr) => {
                    if $reg == Register::Zero {
                        // If we are writing to the Zero register, do not emit any code
                        String::new()
                    } else {
                        format!("mov qword [r13 + {reg} * 8], {x86reg}\n",
                                x86reg = $x86reg,
                                reg = $reg as usize)
                    }
                }
            }

            match opcode {
                0b0110111 => {
                    // LUI
                    let inst = Utype::from(inst);
                    asm += &store_reg!(inst.rd, inst.imm);
                }
                0b0010111 => {
                    // AUIPC
                    let inst = Utype::from(inst);

                    let val = (inst.imm as i64 as u64).wrapping_add(pc);
                    asm += &format!(r#"
                        mov rax, {imm:#x}
                        {store_rd_from_rax}
                    "#, store_rd_from_rax = store_reg!(inst.rd, "rax"), imm = val);
                }
                0b1101111 => {
                    // JAL
                    let inst = Jtype::from(inst);

                    // Compute the return address
                    let ret = pc.wrapping_add(4);

                    // Compute the branch target
                    let target = pc.wrapping_add(inst.imm as i64 as u64);

                    if (target / 4) >= num_blocks as u64 {
                        // Branch target is out of bounds
                        return Err(VmExit::JitOob);
                    }

                    asm += &format!(r#"
                        mov rax, {ret}
                        {store_rd_from_rax}

                        mov rax, [r14 + {target}]
                        test rax, rax
                        jz .jit_resolve

                        add r15, {block_instrs}
                        jmp rax

                        .jit_resolve:
                        mov rax, 1
                        mov rdx, {target_pc}

                        add r15, {block_instrs}
                        ret
                    "#, store_rd_from_rax = store_reg!(inst.rd, "rax"),
                        ret = ret,
                        target = (target / 4) * 8,
                        target_pc = target,
                        block_instrs = block_instrs,
                    );

                    break 'next_inst;
                }
                0b1100111 => {
                    // We know it's an Itype
                    let inst = Itype::from(inst);

                    match inst.funct3 {
                        0b000 => {
                            // JALR

                            // Compute the return address
                            let ret = pc.wrapping_add(4);

                            asm += &format!(r#"
                                mov rax, {ret}
                                {store_rd_from_rax}

                                {load_rax_from_rs1}
                                add rax, {imm}

                                shr rax, 2
                                cmp rax, {num_blocks}
                                jae .jit_resolve

                                mov rax, [r14 + rax*8]
                                test rax, rax
                                jz .jit_resolve

                                add r15, {block_instrs}
                                jmp rax

                                .jit_resolve:
                                {load_rdx_from_rs1}
                                add rdx, {imm}
                                mov rax, 1
                                add r15, {block_instrs}
                                ret
                            "#, store_rd_from_rax = store_reg!(inst.rd, "rax"),
                                load_rax_from_rs1 = load_reg!("rax", inst.rs1),
                                load_rdx_from_rs1 = load_reg!("rdx", inst.rs1),
                                block_instrs = block_instrs,
                                imm = inst.imm, ret = ret, num_blocks = num_blocks);

                            break 'next_inst;
                        }
                        _ => unimplemented!("Unexpected 0b1100111"),
                    }
                }
                0b1100011 => {
                    // We know it's an Btype
                    let inst = Btype::from(inst);

                    match inst.funct3 {
                        0b000 | 0b001 | 0b100 | 0b101 | 0b110 | 0b111 => {
                            let cond = match inst.funct3 {
                                0b000 => /* BEQ */ "jne",
                                0b001 => /* BNE */ "je",
                                0b100 => /* BLT */ "jnl",
                                0b101 => /* BGE */ "jnge",
                                0b110 => /* BLTU */ "jnb",
                                0b111 => /* BGEU */ "jnae",
                                _ => unreachable!(),
                            };

                            let target = pc.wrapping_add(inst.imm as i64 as u64);

                            if (target / 4) >= num_blocks as u64 {
                                // Branch target is out of bounds
                                return Err(VmExit::JitOob);
                            }

                            asm += &format!(r#"
                                {load_rax_from_rs1}
                                {load_rdx_from_rs2}

                                cmp rax, rdx
                                {cond} .fallthrough

                                mov rax, [r14 + {target}]
                                test rax, rax
                                jz .jit_resolve

                                add r15, {block_instrs}
                                jmp rax

                                .jit_resolve:
                                mov rax, 1
                                mov rdx, {target_pc}
                                add r15, {block_instrs}
                                ret

                                .fallthrough:
                            "#, cond = cond,
                                load_rax_from_rs1 = load_reg!("rax", inst.rs1),
                                load_rdx_from_rs2 = load_reg!("rdx", inst.rs2),
                                target_pc = target,
                                target = (target / 4) * 8,
                                block_instrs = block_instrs,
                                );

                        }
                        _ => unimplemented!("Unexpected 0b1100011"),
                    }
                }
                0b0000011 => {
                    // We know it's an Itype
                    let inst = Itype::from(inst);

                    let (loadtyp, loadsz, regtyp, access_size) = match inst.funct3 {
                        0b000 => /* LB */ ("movsx", "byte", "rbx", 1),
                        0b001 => /* LH */ ("movsx", "word", "rbx", 2),
                        0b010 => /* LW */ ("movsx", "dword", "rbx", 4),
                        0b011 => /* LD */ ("mov", "qword", "rbx", 8),
                        0b100 => /* LBU */ ("movzx", "byte", "rbx", 1),
                        0b101 => /* LHU */ ("movzx", "word", "rbx", 2),
                        0b110 => /* LWU */ ("mov", "dword", "ebx", 4),
                        _ => unreachable!(),
                    };

                    // Compute the read permission mask
                    let mut perm_mask = 0u64;
                    for ii in 0..access_size {
                        perm_mask |= (PERM_READ as u64) << (ii * 8)
                    }

                    asm += &format!(r#"
                        {load_rax_from_rs1}
                        add rax, {imm}

                        ; Check if we are Out of bounds
                        cmp rax, {memory_len} - {access_size}
                        ja .fault

                        {loadtyp} {regtyp}, {loadsz} [r9 + rax]
                        mov rcx, {perm_mask}
                        and rbx, rcx
                        cmp rbx, rcx
                        je .nofault

                        .fault:
                        mov rcx, rax
                        mov rbx, {pc}
                        mov rax, 4
                        add r15, {block_instrs}
                        ret

                        .nofault:
                        {loadtyp} {regtyp}, {loadsz} [r8 + rax]
                        {store_rax_into_rd}
                    "#, load_rax_from_rs1 = load_reg!("rax", inst.rs1),
                        store_rax_into_rd = store_reg!(inst.rd, "rbx"),
                        imm = inst.imm,
                        loadtyp = loadtyp,
                        loadsz = loadsz,
                        regtyp = regtyp,
                        perm_mask= perm_mask,
                        pc = pc,
                        access_size = access_size,
                        block_instrs = block_instrs,
                        memory_len = self.memory.len(),
                    );
                }
                0b0100011 => {
                    // We know it's an Stype
                    let inst = Stype::from(inst);

                    let (storetyp, storesz, regtype, loadrt, access_size) = match inst.funct3 {
                        0b000 => /* SB */ ("movzx", "byte", "dl", "edx", 1),
                        0b001 => /* SH */ ("movzx", "word", "dx", "edx", 2),
                        0b010 => /* SW */ ("mov", "dword", "edx", "edx", 4),
                        0b011 => /* SD */ ("mov", "qword", "rdx", "rdx", 8),
                        _ => unreachable!(),
                    };

                    // Make sure the dirty block size is sane
                    assert!(DIRTY_BLOCK_SIZE.count_ones() == 1 &&
                            DIRTY_BLOCK_SIZE >= 8,
                        "Dirty block size must be a power of two and >= 8");

                    // Amount to shift to get the block from an address
                    let dirty_block_shift = DIRTY_BLOCK_SIZE.trailing_zeros();

                    // Compute the write permission mask
                    let mut write_mask = 0u64;
                    for ii in 0..access_size {
                        write_mask |= (PERM_WRITE as u64) << (ii * 8)
                    }

                    asm += &format!(r#"
                        {load_rax_from_rs1}
                        add rax, {imm}

                        ; Check if we are Out of bounds
                        cmp rax, {memory_len} - {access_size}
                        ja .fault

                        {storetyp} {loadrt}, {storesz} [r9 + rax]
                        mov rcx, {write_mask}
                        mov rdi, rbx
                        and rdx, rcx
                        cmp rdx, rcx
                        je .nofault

                        .fault:
                        mov rcx, rax
                        mov rdx, {pc}
                        mov rax, 5
                        add r15, {block_instrs}
                        ret

                        .nofault:
                        ; Get the raw bits and shift them into the read slot
                        ; Distance between the RAW bit and the not RAW bit
                        ; TODO
                        ; Convert the mask for the correct size into the RAW mask
                        shl rcx, 2
                        ; We know have the RAW bits in rdi
                        and rdi, rcx
                        shr rdi, 3
                        mov rdx, rdi
                        ; Update the permissions
                        or {storesz} [r9 + rax], {regtype}
                        mov rcx, rax

                        ; This gives us the dirty block index
                        shr rcx, {dirty_block_shift}
                        bts qword [r11], rcx
                        ; If its non-zero, continue
                        jc .continue

                        ; Write the block rcx at the dirty index
                        mov qword [r10 + r12 * 8], rcx
                        ; Increment the dirty index
                        add r12, 1

                        .continue:
                        {load_rdx_from_rs2}
                        mov {storesz} [r8 + rax], {regtype}
                    "#,
                        load_rax_from_rs1 = load_reg!("rax", inst.rs1),
                        load_rdx_from_rs2 = load_reg!("rdx", inst.rs2),
                        imm = inst.imm,
                        access_size = access_size,
                        block_instrs = block_instrs,
                        write_mask = write_mask,
                        memory_len = self.memory.len(),
                        loadrt = loadrt,
                        storetyp = storetyp,
                        pc = pc,
                        storesz = storesz,
                        regtype = regtype,
                        dirty_block_shift = dirty_block_shift,
                    );
                }
                0b0010011 => {
                    // We know it's an Itype
                    let inst = Itype::from(inst);

                    match inst.funct3 {
                        0b000 => {
                            // ADDI
                            asm += &format!(r#"
                                {load_rax_from_rs1}
                                add rax, {imm}
                                {store_rax_into_rd}
                                "#,
                                load_rax_from_rs1 = load_reg!("rax", inst.rs1),
                                imm = inst.imm,
                                store_rax_into_rd = store_reg!(inst.rd, "rax"),
                                );
                        }
                        0b010 => {
                            // SLTI
                            asm += &format!(r#"
                                {load_rax_from_rs1}
                                xor edx, edx
                                cmp rax, {imm}
                                setl bl
                                {store_rdx_into_rd}
                                "#,
                                load_rax_from_rs1 = load_reg!("rax", inst.rs1),
                                imm = inst.imm,
                                store_rdx_into_rd = store_reg!(inst.rd, "rdx"),
                                );
                        }
                        0b011 => {
                            // SLTIU
                            asm += &format!(r#"
                                {load_rax_from_rs1}
                                ; This zero extends the entire rdx register
                                xor edx, edx
                                cmp rax, {imm}
                                setb bl
                                {store_rdx_into_rd}
                                "#,
                                load_rax_from_rs1 = load_reg!("rax", inst.rs1),
                                imm = inst.imm,
                                store_rdx_into_rd = store_reg!(inst.rd, "rdx"),
                                );
                        }
                        0b100 => {
                            // XORI
                            asm += &format!(r#"
                                {load_rax_from_rs1}
                                xor rax, {imm}
                                {store_rax_into_rd}
                                "#,
                                load_rax_from_rs1 = load_reg!("rax", inst.rs1),
                                imm = inst.imm,
                                store_rax_into_rd = store_reg!(inst.rd, "rax"),
                                );
                        }
                        0b110 => {
                            // ORI
                            asm += &format!(r#"
                                {load_rax_from_rs1}
                                or rax, {imm}
                                {store_rax_into_rd}
                                "#,
                                load_rax_from_rs1 = load_reg!("rax", inst.rs1),
                                imm = inst.imm,
                                store_rax_into_rd = store_reg!(inst.rd, "rax"),
                                );
                        }
                        0b111 => {
                            // ANDI
                            asm += &format!(r#"
                                {load_rax_from_rs1}
                                and rax, {imm}
                                {store_rax_into_rd}
                                "#,
                                load_rax_from_rs1 = load_reg!("rax", inst.rs1),
                                imm = inst.imm,
                                store_rax_into_rd = store_reg!(inst.rd, "rax"),
                                );
                        }
                        0b001 => {
                            let mode = (inst.imm >> 6) & 0b111111;

                            match mode {
                                0b000000 => {
                                    // SLLI
                                    let shamt = inst.imm & 0b111111;
                                    asm += &format!(r#"
                                        {load_rax_from_rs1}
                                        shl rax, {imm}
                                        {store_rax_into_rd}
                                        "#,
                                        load_rax_from_rs1 = load_reg!("rax", inst.rs1),
                                        imm = shamt,
                                        store_rax_into_rd = store_reg!(inst.rd, "rax"),
                                    );
                                }
                                _ => unreachable!(),
                            }
                        }
                        0b101 => {
                            let mode = (inst.imm >> 6) & 0b111111;

                            match mode {
                                0b000000 => {
                                    // SRLI
                                    let shamt = inst.imm & 0b111111;
                                    asm += &format!(r#"
                                        {load_rax_from_rs1}
                                        shr rax, {imm}
                                        {store_rax_into_rd}
                                        "#,
                                        load_rax_from_rs1 = load_reg!("rax", inst.rs1),
                                        imm = shamt,
                                        store_rax_into_rd = store_reg!(inst.rd, "rax"),
                                    );
                                }
                                0b010000 => {
                                    // SRAI
                                    let shamt = inst.imm & 0b111111;
                                    asm += &format!(r#"
                                        {load_rax_from_rs1}
                                        sar rax, {imm}
                                        {store_rax_into_rd}
                                        "#,
                                        load_rax_from_rs1 = load_reg!("rax", inst.rs1),
                                        imm = shamt,
                                        store_rax_into_rd = store_reg!(inst.rd, "rax"),
                                    );
                                }
                                _ => unreachable!(),
                            }
                        }
                        _ => unreachable!(),
                    }
                }
                0b0110011 => {
                    // We know it's an Rtype
                    let inst = Rtype::from(inst);

                    match (inst.funct7, inst.funct3) {
                        (0b0000000, 0b000) => {
                            // ADD
                            asm += &format!(r#"
                                {load_rax_from_rs1}
                                {load_rdx_from_rs2}
                                add rax, rdx
                                {store_rax_into_rd}
                                "#,
                                load_rax_from_rs1 = load_reg!("rax", inst.rs1),
                                load_rdx_from_rs2 = load_reg!("rdx", inst.rs2),
                                store_rax_into_rd = store_reg!(inst.rd, "rax"),
                            );
                        }
                        (0b0100000, 0b000) => {
                            // SUB
                            asm += &format!(r#"
                                {load_rax_from_rs1}
                                {load_rdx_from_rs2}
                                sub rax, rdx
                                {store_rax_into_rd}
                                "#,
                                load_rax_from_rs1 = load_reg!("rax", inst.rs1),
                                load_rdx_from_rs2 = load_reg!("rdx", inst.rs2),
                                store_rax_into_rd = store_reg!(inst.rd, "rax"),
                            );
                        }
                        (0b0000000, 0b001) => {
                            // SLL
                            asm += &format!(r#"
                                {load_rax_from_rs1}
                                {load_rcx_from_rs2}
                                shl rax, cl
                                {store_rax_into_rd}
                                "#,
                                load_rax_from_rs1 = load_reg!("rax", inst.rs1),
                                load_rcx_from_rs2 = load_reg!("rcx", inst.rs2),
                                store_rax_into_rd = store_reg!(inst.rd, "rax"),
                            );
                        }
                        (0b0000000, 0b010) => {
                            // SLT
                            asm += &format!(r#"
                                {load_rax_from_rs1}
                                {load_rdx_from_rs2}
                                xor ecx, ecx
                                cmp rax, rdx
                                setl cl
                                {store_rcx_into_rd}
                                "#,
                                load_rax_from_rs1 = load_reg!("rax", inst.rs1),
                                load_rdx_from_rs2 = load_reg!("rdx", inst.rs2),
                                store_rcx_into_rd = store_reg!(inst.rd, "rcx"),
                            );
                        }
                        (0b0000000, 0b011) => {
                            // SLTU
                            asm += &format!(r#"
                                {load_rax_from_rs1}
                                {load_rdx_from_rs2}
                                xor ecx, ecx
                                cmp rax, rdx
                                setb cl
                                {store_rcx_into_rd}
                                "#,
                                load_rax_from_rs1 = load_reg!("rax", inst.rs1),
                                load_rdx_from_rs2 = load_reg!("rdx", inst.rs2),
                                store_rcx_into_rd = store_reg!(inst.rd, "rcx"),
                            );
                        }
                        (0b0000000, 0b100) => {
                            // XOR
                            asm += &format!(r#"
                                {load_rax_from_rs1}
                                {load_rdx_from_rs2}
                                xor rax, rdx
                                {store_rax_into_rd}
                                "#,
                                load_rax_from_rs1 = load_reg!("rax", inst.rs1),
                                load_rdx_from_rs2 = load_reg!("rdx", inst.rs2),
                                store_rax_into_rd = store_reg!(inst.rd, "rax"),
                            );
                        }
                        (0b0000000, 0b101) => {
                            // SRL
                            asm += &format!(r#"
                                {load_rax_from_rs1}
                                {load_rcx_from_rs2}
                                shr rax, cl
                                {store_rax_into_rd}
                                "#,
                                load_rax_from_rs1 = load_reg!("rax", inst.rs1),
                                load_rcx_from_rs2 = load_reg!("rcx", inst.rs2),
                                store_rax_into_rd = store_reg!(inst.rd, "rax"),
                            );
                        }
                        (0b0100000, 0b101) => {
                            // SRA
                            asm += &format!(r#"
                                {load_rax_from_rs1}
                                {load_rcx_from_rs2}
                                sar rax, cl
                                {store_rax_into_rd}
                                "#,
                                load_rax_from_rs1 = load_reg!("rax", inst.rs1),
                                load_rcx_from_rs2 = load_reg!("rcx", inst.rs2),
                                store_rax_into_rd = store_reg!(inst.rd, "rax"),
                            );
                        }
                        (0b0000000, 0b110) => {
                            // OR
                            asm += &format!(r#"
                                {load_rax_from_rs1}
                                {load_rdx_from_rs2}
                                or rax, rdx
                                {store_rax_into_rd}
                                "#,
                                load_rax_from_rs1 = load_reg!("rax", inst.rs1),
                                load_rdx_from_rs2 = load_reg!("rdx", inst.rs2),
                                store_rax_into_rd = store_reg!(inst.rd, "rax"),
                            );
                        }
                        (0b0000000, 0b111) => {
                            // AND
                            asm += &format!(r#"
                                {load_rax_from_rs1}
                                {load_rdx_from_rs2}
                                and rax, rdx
                                {store_rax_into_rd}
                                "#,
                                load_rax_from_rs1 = load_reg!("rax", inst.rs1),
                                load_rdx_from_rs2 = load_reg!("rdx", inst.rs2),
                                store_rax_into_rd = store_reg!(inst.rd, "rax"),
                            );
                        }
                        _ => unreachable!(),
                    }
                }
                0b0111011 => {
                    // We know it's an Rtype
                    let inst = Rtype::from(inst);

                    match (inst.funct7, inst.funct3) {
                        (0b0000000, 0b000) => {
                            // ADDW
                            asm += &format!(r#"
                                {load_rax_from_rs1}
                                {load_rdx_from_rs2}
                                add eax, edx
                                ; Sign extend
                                movsx rax, eax
                                {store_rax_into_rd}
                                "#,
                                load_rax_from_rs1 = load_reg!("rax", inst.rs1),
                                load_rdx_from_rs2 = load_reg!("rdx", inst.rs2),
                                store_rax_into_rd = store_reg!(inst.rd, "rax"),
                            );
                        }
                        (0b0100000, 0b000) => {
                            // SUBW
                            asm += &format!(r#"
                                {load_rax_from_rs1}
                                {load_rdx_from_rs2}
                                sub eax, edx
                                ; Sign extend
                                movsx rax, eax
                                {store_rax_into_rd}
                                "#,
                                load_rax_from_rs1 = load_reg!("rax", inst.rs1),
                                load_rdx_from_rs2 = load_reg!("rdx", inst.rs2),
                                store_rax_into_rd = store_reg!(inst.rd, "rax"),
                            );
                        }
                        (0b0000000, 0b001) => {
                            // SLLW
                            asm += &format!(r#"
                                {load_rax_from_rs1}
                                {load_rcx_from_rs2}
                                shl eax, cl
                                movsx rax, eax
                                {store_rax_into_rd}
                                "#,
                                load_rax_from_rs1 = load_reg!("rax", inst.rs1),
                                load_rcx_from_rs2 = load_reg!("rcx", inst.rs2),
                                store_rax_into_rd = store_reg!(inst.rd, "rax"),
                            );
                        }
                        (0b0000000, 0b101) => {
                            // SRLW
                            asm += &format!(r#"
                                {load_rax_from_rs1}
                                {load_rcx_from_rs2}
                                shr eax, cl
                                movsx rax, eax
                                {store_rax_into_rd}
                                "#,
                                load_rax_from_rs1 = load_reg!("rax", inst.rs1),
                                load_rcx_from_rs2 = load_reg!("rcx", inst.rs2),
                                store_rax_into_rd = store_reg!(inst.rd, "rax"),
                            );
                        }
                        (0b0100000, 0b101) => {
                            // SRAW
                            asm += &format!(r#"
                                {load_rax_from_rs1}
                                {load_rcx_from_rs2}
                                sar eax, cl
                                movsx rax, eax
                                {store_rax_into_rd}
                                "#,
                                load_rax_from_rs1 = load_reg!("rax", inst.rs1),
                                load_rcx_from_rs2 = load_reg!("rcx", inst.rs2),
                                store_rax_into_rd = store_reg!(inst.rd, "rax"),
                            );
                        }
                        _ => unreachable!(),
                    }
                }
                0b0001111 => {
                    let inst = Itype::from(inst);

                    match inst.funct3 {
                        0b000 => {
                            // FENCE
                        }
                        _ => unreachable!(),
                    }
                }
                0b1110011 => {
                    if inst == 0b00000000000000000000000001110011 {
                        // ECALL
                        asm += &format!(r#"
                            mov rax, 2
                            mov rdx, {pc}
                            add r15, {block_instrs}
                            ret
                        "#, pc = pc,
                            block_instrs = block_instrs,
                        );
                    } else if inst == 0b00000000000100000000000001110011 {
                        // EBREAK
                        asm += &format!(r#"
                            mov rax, 3
                            mov rdx, {pc}
                            add r15, {block_instrs}
                            ret
                        "#, pc = pc,
                            block_instrs = block_instrs,
                        );
                    } else {
                        unreachable!();
                    }
                }
                0b0011011 => {
                    // We know it's an Itype
                    let inst = Itype::from(inst);

                    match inst.funct3 {
                        0b000 => {
                            // ADDIW
                            asm += &format!(r#"
                                {load_rax_from_rs1}
                                add eax, {imm}
                                movsx rax, eax
                                {store_rax_into_rd}
                                "#,
                                load_rax_from_rs1 = load_reg!("rax", inst.rs1),
                                store_rax_into_rd = store_reg!(inst.rd, "rax"),
                                imm = inst.imm,
                            );
                        }
                        0b001 => {
                            let mode = (inst.imm >> 5) & 0b1111111;

                            match mode {
                                0b0000000 => {
                                    // SLLIW
                                    let shamt = inst.imm & 0b11111;
                                    asm += &format!(r#"
                                        {load_rax_from_rs1}
                                        shl eax, {imm}
                                        movsx rax, eax
                                        {store_rax_into_rd}
                                        "#,
                                        load_rax_from_rs1 = load_reg!("rax", inst.rs1),
                                        imm = shamt,
                                        store_rax_into_rd = store_reg!(inst.rd, "rax"),
                                    );
                                }
                                _ => unreachable!(),
                            }
                        }
                        0b101 => {
                            let mode = (inst.imm >> 5) & 0b1111111;

                            match mode {
                                0b0000000 => {
                                    // SRLIW
                                    let shamt = inst.imm & 0b11111;
                                    asm += &format!(r#"
                                        {load_rax_from_rs1}
                                        shr eax, {imm}
                                        movsx rax, eax
                                        {store_rax_into_rd}
                                        "#,
                                        load_rax_from_rs1 = load_reg!("rax", inst.rs1),
                                        imm = shamt,
                                        store_rax_into_rd = store_reg!(inst.rd, "rax"),
                                    );
                                }
                                0b0100000 => {
                                    // SRAIW
                                    let shamt = inst.imm & 0b11111;
                                    asm += &format!(r#"
                                        {load_rax_from_rs1}
                                        sar eax, {imm}
                                        movsx rax, eax
                                        {store_rax_into_rd}
                                        "#,
                                        load_rax_from_rs1 = load_reg!("rax", inst.rs1),
                                        imm = shamt,
                                        store_rax_into_rd = store_reg!(inst.rd, "rax"),
                                    );
                                }
                                _ => unreachable!(),
                            }
                        }
                        _ => unreachable!(),
                    }
                }
                _ => unimplemented!("Unhandled opcode {:#09b}\n", opcode),
            }
            pc += 4;
        }
        Ok(asm)
    }

    /// Run the VM using the emulator
    pub fn run_emu(&mut self, instrs_execed: &mut u64) -> Result<(), VmExit> {
        'next_inst: loop {
            // Get the current program counter
            let pc = self.reg(Register::Pc);
            let inst: u32 = self
                .memory
                .read_perms(VirtAddr(pc as usize), Perm(PERM_EXEC))?;

            // Update number of instructions executed
            *instrs_execed += 1;

            // Extract the opcode from the instruction
            let opcode = inst & 0b1111111;
            //print!("{}\n\n", self);

            match opcode {
                0b0110111 => {
                    // LUI
                    let inst = Utype::from(inst);
                    self.set_reg(inst.rd, inst.imm as i64 as u64);
                }
                0b0010111 => {
                    // AUIPC
                    let inst = Utype::from(inst);
                    self.set_reg(inst.rd, (inst.imm as i64 as u64).wrapping_add(pc));
                }
                0b1101111 => {
                    // JAL
                    let inst = Jtype::from(inst);
                    self.set_reg(inst.rd, pc.wrapping_add(4));
                    self.set_reg(Register::Pc, pc.wrapping_add(inst.imm as i64 as u64));
                    continue 'next_inst;
                }
                0b1100111 => {
                    // We know it's an Itype
                    let inst = Itype::from(inst);

                    match inst.funct3 {
                        0b000 => {
                            // JALR
                            let target = self.reg(inst.rs1).wrapping_add(inst.imm as i64 as u64);
                            self.set_reg(inst.rd, pc.wrapping_add(4));
                            self.set_reg(Register::Pc, target);
                            continue 'next_inst;
                        }
                        _ => unimplemented!("Unexpected 0b1100111"),
                    }
                }
                0b1100011 => {
                    // We know it's an Btype
                    let inst = Btype::from(inst);

                    let rs1 = self.reg(inst.rs1);
                    let rs2 = self.reg(inst.rs2);

                    match inst.funct3 {
                        0b000 => {
                            // BEQ
                            if rs1 == rs2 {
                                self.set_reg(Register::Pc, pc.wrapping_add(inst.imm as i64 as u64));
                                continue 'next_inst;
                            }
                        }
                        0b001 => {
                            // BNE
                            if rs1 != rs2 {
                                self.set_reg(Register::Pc, pc.wrapping_add(inst.imm as i64 as u64));
                                continue 'next_inst;
                            }
                        }
                        0b100 => {
                            // BLT
                            if (rs1 as i64) < (rs2 as i64) {
                                self.set_reg(Register::Pc, pc.wrapping_add(inst.imm as i64 as u64));
                                continue 'next_inst;
                            }
                        }
                        0b101 => {
                            // BGE
                            if (rs1 as i64) >= (rs2 as i64) {
                                self.set_reg(Register::Pc, pc.wrapping_add(inst.imm as i64 as u64));
                                continue 'next_inst;
                            }
                        }
                        0b110 => {
                            // BLTU
                            if (rs1 as u64) < (rs2 as u64) {
                                self.set_reg(Register::Pc, pc.wrapping_add(inst.imm as i64 as u64));
                                continue 'next_inst;
                            }
                        }
                        0b111 => {
                            // BGEU
                            if (rs1 as u64) >= (rs2 as u64) {
                                self.set_reg(Register::Pc, pc.wrapping_add(inst.imm as i64 as u64));
                                continue 'next_inst;
                            }
                        }
                        _ => unimplemented!("Unexpected 0b1100011"),
                    }
                }
                0b0000011 => {
                    // We know it's an Itype
                    let inst = Itype::from(inst);

                    // Compute the address
                    let addr =
                        VirtAddr(self.reg(inst.rs1).wrapping_add(inst.imm as i64 as u64) as usize);

                    match inst.funct3 {
                        0b000 => {
                            // LB
                            let mut tmp = [0u8; 1];
                            self.memory.read_into(addr, &mut tmp)?;
                            self.set_reg(inst.rd, i8::from_le_bytes(tmp) as i64 as u64);
                        }
                        0b001 => {
                            // LH
                            let mut tmp = [0u8; 2];
                            self.memory.read_into(addr, &mut tmp)?;
                            self.set_reg(inst.rd, i16::from_le_bytes(tmp) as i64 as u64);
                        }
                        0b010 => {
                            // LW
                            let mut tmp = [0u8; 4];
                            self.memory.read_into(addr, &mut tmp)?;
                            self.set_reg(inst.rd, i32::from_le_bytes(tmp) as i64 as u64);
                        }
                        0b011 => {
                            // LD
                            let mut tmp = [0u8; 8];
                            self.memory.read_into(addr, &mut tmp)?;
                            self.set_reg(inst.rd, i64::from_le_bytes(tmp) as i64 as u64);
                        }
                        0b100 => {
                            // LBU
                            let mut tmp = [0u8; 1];
                            self.memory.read_into(addr, &mut tmp)?;
                            self.set_reg(inst.rd, u8::from_le_bytes(tmp) as u64);
                        }
                        0b101 => {
                            // LHU
                            let mut tmp = [0u8; 2];
                            self.memory.read_into(addr, &mut tmp)?;
                            self.set_reg(inst.rd, u16::from_le_bytes(tmp) as u64);
                        }
                        0b110 => {
                            // LWU
                            let mut tmp = [0u8; 4];
                            self.memory.read_into(addr, &mut tmp)?;
                            self.set_reg(inst.rd, u32::from_le_bytes(tmp) as u64);
                        }
                        _ => unimplemented!("Unexpected 0b0000011"),
                    }
                }
                0b0100011 => {
                    // We know it's an Stype
                    let inst = Stype::from(inst);

                    // Compute the address
                    let addr =
                        VirtAddr(self.reg(inst.rs1).wrapping_add(inst.imm as i64 as u64) as usize);

                    match inst.funct3 {
                        0b000 => {
                            // SB
                            let val = self.reg(inst.rs2) as u8;
                            self.memory.write(addr, val)?;
                        }
                        0b001 => {
                            // SH
                            let val = self.reg(inst.rs2) as u16;
                            self.memory.write(addr, val)?;
                        }
                        0b010 => {
                            // SW
                            let val = self.reg(inst.rs2) as u32;
                            self.memory.write(addr, val)?;
                        }
                        0b011 => {
                            // SD
                            let val = self.reg(inst.rs2) as u64;
                            self.memory.write(addr, val)?;
                        }
                        _ => unimplemented!("Unexpected 0b0100011"),
                    }
                }
                0b0010011 => {
                    // We know it's an Itype
                    let inst = Itype::from(inst);

                    let rs1 = self.reg(inst.rs1);
                    let imm = inst.imm as i64 as u64;

                    match inst.funct3 {
                        0b000 => {
                            // ADDI
                            self.set_reg(inst.rd, rs1.wrapping_add(imm));
                        }
                        0b010 => {
                            // SLTI
                            if (rs1 as i64) < (imm as i64) {
                                self.set_reg(inst.rd, 1);
                            } else {
                                self.set_reg(inst.rd, 0);
                            }
                        }
                        0b011 => {
                            // SLTIU
                            if (rs1 as u64) < (imm as u64) {
                                self.set_reg(inst.rd, 1);
                            } else {
                                self.set_reg(inst.rd, 0);
                            }
                        }
                        0b100 => {
                            // XORI
                            self.set_reg(inst.rd, rs1 ^ imm);
                        }
                        0b110 => {
                            // ORI
                            self.set_reg(inst.rd, rs1 | imm);
                        }
                        0b111 => {
                            // ANDI
                            self.set_reg(inst.rd, rs1 & imm);
                        }
                        0b001 => {
                            let mode = (inst.imm >> 6) & 0b111111;

                            match mode {
                                0b000000 => {
                                    // SLLI
                                    let shamt = inst.imm & 0b111111;
                                    self.set_reg(inst.rd, rs1 << shamt);
                                }
                                _ => unreachable!(),
                            }
                        }
                        0b101 => {
                            let mode = (inst.imm >> 6) & 0b111111;

                            match mode {
                                0b000000 => {
                                    // SRLI
                                    let shamt = inst.imm & 0b111111;
                                    self.set_reg(inst.rd, rs1 >> shamt);
                                }
                                0b010000 => {
                                    // SRAI
                                    let shamt = inst.imm & 0b111111;
                                    self.set_reg(inst.rd, ((rs1 as i64) >> shamt) as u64);
                                }
                                _ => unreachable!(),
                            }
                        }
                        _ => unreachable!(),
                    }
                }
                0b0110011 => {
                    // We know it's an Rtype
                    let inst = Rtype::from(inst);

                    let rs1 = self.reg(inst.rs1);
                    let rs2 = self.reg(inst.rs2);

                    match (inst.funct7, inst.funct3) {
                        (0b0000000, 0b000) => {
                            // ADD
                            self.set_reg(inst.rd, rs1.wrapping_add(rs2));
                        }
                        (0b0100000, 0b000) => {
                            // SUB
                            self.set_reg(inst.rd, rs1.wrapping_sub(rs2));
                        }
                        (0b0000000, 0b001) => {
                            // SLL
                            let shamt = rs2 & 0b111111;
                            self.set_reg(inst.rd, rs1 << shamt);
                        }
                        (0b0000000, 0b010) => {
                            // SLT
                            if (rs1 as i64) < (rs2 as i64) {
                                self.set_reg(inst.rd, 1);
                            } else {
                                self.set_reg(inst.rd, 0);
                            }
                        }
                        (0b0000000, 0b011) => {
                            // SLTU
                            if (rs1 as u64) < (rs2 as u64) {
                                self.set_reg(inst.rd, 1);
                            } else {
                                self.set_reg(inst.rd, 0);
                            }
                        }
                        (0b0000000, 0b100) => {
                            // XOR
                            self.set_reg(inst.rd, rs1 ^ rs2);
                        }
                        (0b0000000, 0b101) => {
                            // SRL
                            let shamt = rs2 & 0b111111;
                            self.set_reg(inst.rd, rs1 >> shamt);
                        }
                        (0b0100000, 0b101) => {
                            // SRA
                            let shamt = rs2 & 0b111111;
                            self.set_reg(inst.rd, ((rs1 as i64) >> shamt) as u64);
                        }
                        (0b0000000, 0b110) => {
                            // OR
                            self.set_reg(inst.rd, rs1 | rs2);
                        }
                        (0b0000000, 0b111) => {
                            // AND
                            self.set_reg(inst.rd, rs1 & rs2);
                        }
                        _ => unreachable!(),
                    }
                }
                0b0111011 => {
                    // We know it's an Rtype
                    let inst = Rtype::from(inst);

                    let rs1 = self.reg(inst.rs1) as u32;
                    let rs2 = self.reg(inst.rs2) as u32;

                    match (inst.funct7, inst.funct3) {
                        (0b0000000, 0b000) => {
                            // ADDW
                            self.set_reg(inst.rd, rs1.wrapping_add(rs2) as i32 as i64 as u64);
                        }
                        (0b0100000, 0b000) => {
                            // SUBW
                            self.set_reg(inst.rd, rs1.wrapping_sub(rs2) as i32 as i64 as u64);
                        }
                        (0b0000000, 0b001) => {
                            // SLLW
                            let shamt = rs2 & 0b11111;
                            self.set_reg(inst.rd, (rs1 << shamt) as i32 as i64 as u64);
                        }
                        (0b0000000, 0b101) => {
                            // SRLW
                            let shamt = rs2 & 0b11111;
                            self.set_reg(inst.rd, (rs1 >> shamt) as i32 as i64 as u64);
                        }
                        (0b0100000, 0b101) => {
                            // SRAW
                            let shamt = rs2 & 0b11111;
                            self.set_reg(inst.rd, ((rs1 as i32) >> shamt) as i64 as u64);
                        }
                        _ => unreachable!(),
                    }
                }
                0b0001111 => {
                    let inst = Itype::from(inst);

                    match inst.funct3 {
                        0b000 => {
                            // FENCE
                        }
                        _ => unreachable!(),
                    }
                }
                0b1110011 => {
                    if inst == 0b00000000000000000000000001110011 {
                        // ECALL
                        return Err(VmExit::Syscall);
                    } else if inst == 0b00000000000100000000000001110011 {
                        // EBREAK
                        panic!("EBREAK");
                    } else {
                        unreachable!();
                    }
                }
                0b0011011 => {
                    // We know it's an Itype
                    let inst = Itype::from(inst);

                    let rs1 = self.reg(inst.rs1) as u32;
                    let imm = inst.imm as u32;

                    match inst.funct3 {
                        0b000 => {
                            // ADDIW
                            self.set_reg(inst.rd, rs1.wrapping_add(imm) as i32 as i64 as u64);
                        }
                        0b001 => {
                            let mode = (inst.imm >> 5) & 0b1111111;

                            match mode {
                                0b0000000 => {
                                    // SLLIW
                                    let shamt = inst.imm & 0b11111;
                                    self.set_reg(inst.rd, (rs1 << shamt) as i32 as i64 as u64);
                                }
                                _ => unreachable!(),
                            }
                        }
                        0b101 => {
                            let mode = (inst.imm >> 5) & 0b1111111;

                            match mode {
                                0b0000000 => {
                                    // SRLIW
                                    let shamt = inst.imm & 0b11111;
                                    self.set_reg(inst.rd, (rs1 >> shamt) as i32 as i64 as u64)
                                }
                                0b0100000 => {
                                    // SRAIW
                                    let shamt = inst.imm & 0b11111;
                                    self.set_reg(inst.rd, ((rs1 as i32) >> shamt) as i64 as u64);
                                }
                                _ => unreachable!(),
                            }
                        }
                        _ => unreachable!(),
                    }
                }
                _ => unimplemented!("Unhandled opcode {:#09b}\n", opcode),
            }

            // Update PC to the next instruction
            self.set_reg(Register::Pc, pc.wrapping_add(4));
        }
    }
}
