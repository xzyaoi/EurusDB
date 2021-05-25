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
    fn flush(&mutself) -> Result<()>;
    
}