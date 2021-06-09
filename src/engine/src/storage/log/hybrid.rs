use storage::kv::{Range, Scan, Store};
use utility::error::{Error, Result};

use std::cmp::{max, min};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::fmt::Display;
use std::fs::{create_dir_all, File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Seek as _, SeekFrom, Write};
use std::ops::Bound;
use std::path::Path;
use std::sync::{Mutex, MutexGuard};

pub struct Hybrid {
    file: Mutex<File>,
    // TODO: This index can be replaced by a learned index.
    index: BTreeMap<u64, (u64, u32)>,
    uncommitted: VecDeque<Vec<u8>>,
    metadata: HashMap<Vec<u8>, Vec<u8>>,
    metadata_file: File,
    sync: bool,
}

impl Display for Hybrid {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Hybrid")
    }
}

impl Hybrid {
    pub fn new(dir: &Path, sync: bool) -> Result<Self> {
        create_dir_all(dir)?;

        let file =
            OpenOptions::new().read(true).write(true).create(true).open(dir.join("raft-log"))?;
        let metadata_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(dir.join("raft-metadata"))?;
        Ok(Self {
            index: Self::build_index(&file)?,
        })
    }
}
