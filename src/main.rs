const PERM_READ: u8 = 1 << 0;
const PERM_WRITE: u8 = 1 << 1;
const PERM_EXEC: u8 = 1 << 2;
const PERM_RAW: u8 = 1 << 3;

const DIRTY_BLOCK_SIZE: usize = 4096;

/// A permissions byte which corresponds to a memory byte and defines the permissions is has
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Perm(u8);

/// A guest virtual address
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct VirtAddr(usize);

/// An isolated memory space
struct Mmu {
    /// Block of memory for this address space
    /// Offset 0 corresponds to address 0 in the guest address space
    memory: Vec<u8>,

    /// Holds the permission bytes for the corresponding byte in memory
    permissions: Vec<Perm>,

    /// Current base address of the next allocation
    curr_base: VirtAddr,
}

impl Mmu {
    /// Create a new memory space which can hold `size` bytes
    pub fn new(size: usize) -> Self {
        Mmu {
            memory: vec![0; size],
            permissions: vec![Perm(0); size],
            curr_base: VirtAddr(0x10000),
        }
    }

    /// Allocate a region of memory as RW in the address space
    pub fn allocate(&mut self, size: usize) -> Option<VirtAddr> {
        // 16-byte align the allocation
        let size_aligned = (size + 0xf) & !0xf;

        // Get the current allocation base
        let base = self.curr_base;

        // Get the current allocation base
        let new_base = VirtAddr(self.curr_base.0.checked_add(size_aligned)?);

        // Check if we go OOM by allocating
        if new_base.0 > self.memory.len() {
            return None;
        }

        // Mark the memory as uninitialized and writable
        self.set_permissions(base, size, Perm(PERM_RAW | PERM_WRITE));

        // Update the new alloc
        self.curr_base = new_base;

        Some(base)
    }

    /// Apply permissions to a region of memory
    pub fn set_permissions(&mut self, addr: VirtAddr, size: usize, perm: Perm) -> Option<()> {
        // Apply permissions
        self.permissions.get_mut(addr.0..addr.0.checked_add(size)?)?
            .iter_mut().for_each(|x| *x = perm);
        Some(())
    }

    /// Write the bytes from `buf` into `addr`
    pub fn write_from(&mut self, addr: VirtAddr, buf: &[u8]) -> Option<()> {
        // Get permissions for the desired address space
        let perms = self.permissions.get_mut(addr.0..addr.0.checked_add(buf.len())?)?;

        // Presume that no bits are read-after-write
        let mut has_raw = false;
        // Check if the bytes are writable, if not, issue permission denied
        if !perms.iter().all(|x| {
            // Check if any of the bits has read-after-write permission
            has_raw |= (x.0 & PERM_RAW) != 0;
            (x.0 & PERM_WRITE) != 0
        }) {
            return None;
        }

        // Write the address space with the `buf` content
        self.memory.get_mut(addr.0..addr.0.checked_add(buf.len())?)?.copy_from_slice(buf);

        // Update RaW bits
        if has_raw {
            perms.iter_mut().for_each(|x| {
                if (x.0 & PERM_RAW) != 0 {
                    // Mark memory as readable
                    // We also have to remove the RaW bit
                    *x = Perm((x.0 | PERM_READ) ^ PERM_RAW);
                }
            });
        }

        Some(())
    }

    /// Read the bytes at `addr` into `buf`
    pub fn read_into(&self, addr: VirtAddr, buf: &mut [u8]) -> Option<()> {
        // Get permissions for the address space
        let perms = self.permissions.get(addr.0..addr.0.checked_add(buf.len())?)?;

        // Check that all the bytes we want to read, are readable
        if !perms.iter().all(|x| (x.0 & PERM_READ) != 0) {
            return None;
        }

        // Apply permissions
        buf.copy_from_slice(self.memory.get(addr.0..addr.0.checked_add(buf.len())?)?);
        Some(())
    }
}

/// All the states of the emulated system
struct Emulator {
    /// Memory of the emulator
    pub memory: Mmu,
}

impl Emulator {
    // Creates a new emulator with `size` bytes of memory
    pub fn new(size: usize) -> Self {
        Emulator {
            memory: Mmu::new(size),
        }
    }
}

fn main() {
    // Make an emulator with 1 Meg of memory
    let mut emu = Emulator::new(1024 * 1024);

    let tmp = emu.memory.allocate(4096).unwrap();
    emu.memory.write_from(VirtAddr(tmp.0 + 0), b"asdf").unwrap();

    let mut bytes = [0u8; 4];
    emu.memory.read_into(tmp, &mut bytes).unwrap();

    print!("{:x?}\n", bytes);
}
