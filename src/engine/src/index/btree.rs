// Copyright (c) 2021 Xiaozhe Yao et al.
// 
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

use storage::kv::{Range, Scan, Store};
use utility::error::{Result};
use std::collections::BTreeMap;
use std::fmt::Display;

pub struct BTreeIndex {
    data: BTreeMap<Vec<u8>, Vec<u8>>,
}

impl BTreeIndex {
    pub fn new() -> Self {
        Self {
            data: BTreeMap::new()
        }
    }
}

impl Display for BTreeIndex {
    fn fmt(&self, f:&mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "B-Tree Index")
    }
}

impl Store for BTreeIndex {
    fn flush(&mut self) -> Result<()> {
        Ok(())
    }

    fn delete(&mut self, key: &[u8]) -> Result<()> {
        self.data.remove(key);
        Ok(())
    }

    fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        Ok(self.data.get(key).cloned())
    }
    
    fn scan(&self, range: Range) -> Scan {
        Box::new(
            self.data.range(range)
            .map(|(k, v)| Ok((k.clone(), v.clone())))
            .collect::<Vec<_>>()
            .into_iter(),
        )
    }

    fn set(&mut self, key:&[u8], value: Vec<u8>) -> Result<()> {
        self.data.insert(key.to_vec(), value);;
        Ok(())
    }
}