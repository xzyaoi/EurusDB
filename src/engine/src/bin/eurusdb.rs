#![warn(clippy::all)]

use clap::{app_from_crate, crate_authors, crate_description, crate_name, crate_version};
use eurusdb::utility::error::{Error, Result};
use serde_derive::Deserialize;
use std::collections::HashMap;

#[tokio::main]
async fn main() -> Result<()> {}
