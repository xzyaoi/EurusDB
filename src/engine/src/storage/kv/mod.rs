// Copyright (c) 2021 Xiaozhe Yao et al.
// 
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

use crate::utility::error::Result;
use crate::sql::types;

#[cfg(test)]
pub use test::Test;

use crate::error::Result;
use std::fmt::Display;
use std::ops::{Bound, RangeBounds};

// A key/value store
pub trait Store: Display + Send + Sync {
    fn delete(&mut self, key: &[u8]) -> Result<()>;
    fn flush(&mut self) -> Result<()>;
    fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>>;
    fn set(&mut self, key: &[u8], value: Vec<u8>) -> Result<()>;
}

pub struct Range {
    start: Bound<Vec<u8>>,
    end: Bound<Vec<u8>>,
}

impl Range {
    pub fn from<R: RangeBounds<Vec<u8>>>(range: R) -> Self {
        Self {
            start: match range.start_bound() {
                Bound::Included(v) => Bound::Included(v.to_vec()),
                Bound::Excluded(v) => Bound::Excluded(v.to_vec()),
                Bound::Unbounded => Bound::Unbounded
            },
            end: match range.end_bound() {
                Bound::Included(v) => Bound::Included(v.to_vec()),
                Bound::Excluded(v) => Bound::Excluded(v.to_vec()),
                Bound::Unbounded => Bound::Unbounded
            }
        }
    }
    pub fn contains(&self, v: &[u8]) -> bool {
        (match &self.start {
            Bound::Included(start) => &**start <=v,
            Bound::Excluded(start) => &**start < v,
            Bound::Unbounded => true,
        }) && (match &self.end {
            Bound::Included(end) => v <= &** end,
            Bound::Excluded(end) => v <  &** end,
            Bound::Unbounded => true,
        })
    }
}

