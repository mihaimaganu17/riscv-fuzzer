//! A software MMU with byte level permissions

use crate::primitive::Primitive;
use std::path::Path;

pub const PERM_READ: u8 = 1 << 0;
pub const PERM_WRITE: u8 = 1 << 1;
pub const PERM_EXEC: u8 = 1 << 2;
pub const PERM_RAW: u8 = 1 << 3;


/// Block size used for resettin and tracking memory which has been modified. The larger this is,
/// the fewer but more expensive memcpys() need to occur, the smaller, the greater but less
/// expensive memcpys() need to occur
pub const DIRTY_BLOCK_SIZE: usize = 4096;

/// A permissions byte which corresponds to a memory byte and defines the permissions is has
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Perm(pub u8);

/// A guest virtual address
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct VirtAddr(pub usize);

/// An isolated memory space
pub struct Mmu {
    /// Block of memory for this address space
    /// Offset 0 corresponds to address 0 in the guest address space
    memory: Vec<u8>,

    /// Holds the permission bytes for the corresponding byte in memory
    permissions: Vec<Perm>,

    /// Tracks block indicies in memory, which are dirty
    dirty: Vec<usize>,

    /// Tracks which parts of memory have been dirtied
    dirty_bitmap: Vec<u64>,

    /// Current base address of the next allocation
    curr_base: VirtAddr,
}

impl Mmu {
    /// Create a new memory space which can hold `size` bytes
    pub fn new(size: usize) -> Self {
        Mmu {
            memory: vec![0; size],
            permissions: vec![Perm(0); size],
            dirty: Vec::with_capacity(size / DIRTY_BLOCK_SIZE + 1),
            dirty_bitmap: vec![0u64; size / DIRTY_BLOCK_SIZE / 64 + 1],
            curr_base: VirtAddr(0x10000),
        }
    }

    /// Fork from an existing MMU
    pub fn fork(&self) -> Self {
        let size = self.memory.len();

        Mmu {
            memory: self.memory.clone(),
            permissions: self.permissions.clone(),
            dirty: Vec::with_capacity(size / DIRTY_BLOCK_SIZE + 1),
            dirty_bitmap: vec![0u64; size / DIRTY_BLOCK_SIZE / 64 + 1],
            curr_base: self.curr_base.clone(),
        }
    }

    /// Restores memory back to the original state (eg. restores all dirty blocks to the state of
    /// other)
    pub fn reset(&mut self, other: &Mmu) {
        for block in &self.dirty {
            // Get the start and the end addresses of the dirtied memory
            let start = block * DIRTY_BLOCK_SIZE;
            let end = (block + 1) * DIRTY_BLOCK_SIZE;

            // Zero the bitmap. Easier to clean all the bits
            self.dirty_bitmap[block / 64] = 0;

            // Restor memory state
            self.memory[start..end].copy_from_slice(&other.memory[start..end]);

            // Restore permissions
            self.permissions[start..end].copy_from_slice(&other.permissions[start..end]);
        }

        // Clear the dirty list
        self.dirty.clear();
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

        // Compute dirty bit blocks
        let block_start = addr.0 / DIRTY_BLOCK_SIZE;
        let block_end = addr.0 + buf.len() / DIRTY_BLOCK_SIZE;

        for block in block_start..=block_end {
            // Determine the bitmap position of the dirty block
            let idx = block_start / 64;
            let bit = block_start % 64;

            // Check if the block is not dirty
            if self.dirty_bitmap[idx] & (1 << bit) == 0 {
                // block is not diry, add it to the dirty list
                self.dirty.push(block);

                // Update the dirty bitmap
                self.dirty_bitmap[idx] |= 1 << bit;
            }
        }

        // Write the address space with the `buf` content
        self.memory.get_mut(addr.0..addr.0.checked_add(buf.len())?)?.copy_from_slice(buf);

        // Update RaW bits
        if has_raw {
            perms.iter_mut().for_each(|x| {
                if (x.0 & PERM_RAW) != 0 {
                    // Mark memory as readable
                    // We also have to remove the RaW bit
                    *x = Perm(x.0 | PERM_READ);
                }
            });
        }

        Some(())
    }

    /// Read the bytes at `addr` into `buf` assuming all `exp_perms` bits are set in the
    /// permission bytes. If this is zero, we ignore permissions entirely
    pub fn read_into_perms(&self, addr: VirtAddr, buf: &mut [u8], exp_perms: Perm) -> Option<()> {
        // Get permissions for the address space
        let perms = self
            .permissions
            .get(addr.0..addr.0.checked_add(buf.len())?)?;

        // Check that all the bytes we want to read have the expected permissions
        if exp_perms.0 != 0 && !perms.iter().all(|x| (x.0 & exp_perms.0) == exp_perms.0) {
            return None;
        }

        // Apply permissions
        buf.copy_from_slice(self.memory.get(addr.0..addr.0.checked_add(buf.len())?)?);
        Some(())
    }

    pub fn read_into(&self, addr: VirtAddr, buf: &mut [u8]) -> Option<()> {
        self.read_into_perms(addr, buf, Perm(PERM_READ))
    }

    /// Read a type `T` as `vaddr`, expecting `perms`
    pub fn read_perms<T: Primitive>(&mut self, addr: VirtAddr, exp_perms: Perm) -> Option<T> {
        let mut tmp = [0u8; 16];
        self.read_into_perms(addr, &mut tmp[..core::mem::size_of::<T>()], exp_perms)?;
        Some(unsafe { core::ptr::read_unaligned(tmp.as_ptr() as *const T) })
    }

    pub fn read<T: Primitive>(&mut self, addr: VirtAddr) -> Option<T> {
        self.read_perms(addr, Perm(PERM_READ))
    }

    pub fn write<T: Primitive>(&mut self, addr: VirtAddr, val: T) -> Option<()> {
        let val = unsafe {
            core::slice::from_raw_parts(&val as *const T as *const u8, core::mem::size_of::<T>())
        };

        self.write_from(addr, val)
    }

    /// Load a file into the emulators address space using the sections as described
    pub fn load<P: AsRef<Path>>(&mut self, filename: P, sections: &[Section]) -> Option<()> {
        let bytes = std::fs::read(filename).ok()?;

        for section in sections {
            // Set memory to writable
            self
                .set_permissions(section.virt_addr, section.mem_size, Perm(PERM_WRITE))?;

            // Write in the original file contents
            self.write_from(
                section.virt_addr,
                bytes.get(section.file_off..section.file_off.checked_add(section.file_size)?)?,
            )?;

            // Write in any padding with zeros
            if section.mem_size > section.file_size {
                let padding = vec![0u8; section.mem_size - section.file_size];
                self.write_from(
                    VirtAddr(section.virt_addr.0.checked_add(section.file_size)?),
                    &padding
                )?;
            }

            // Demote permissions to originals
            self.set_permissions(section.virt_addr, section.mem_size, section.permissions)?;

            // Update the allocator beyond any sections we load
            self.curr_base = VirtAddr(std::cmp::max(
                self.curr_base.0,
                (section.virt_addr.0 + section.mem_size + 0xf) & !0xf
            ));
        }

        Some(())
    }
}

pub struct Section {
    pub file_off: usize,
    pub virt_addr: VirtAddr,
    pub file_size: usize,
    pub mem_size: usize,
    pub permissions: Perm,
}
