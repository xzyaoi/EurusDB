#![warn(clippy::all)]
#![allow(clippy::new_without_default)]
#![allow(clippy::unneeded_field_pattern)]

pub mod storage;
pub mod utility;

pub use utility::client;