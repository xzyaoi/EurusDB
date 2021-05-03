// Copyright (c) 2021 Xiaozhe Yao et al.
// 
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

mod expression;
pub use expression::Expression;

use crate::error::{Error, Result};

use serde_derive::{Deserialize, Serialize};

use std::borrow::Cow;
use std::cmp::Ordering;
use std::hash::{HAsh, Hasher};

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
pub enum DataType {
    Boolean,
    Integer,
    Float,
    String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Value {
    Null,
    Boolean(bool),
    Integer(i64),
    Float(f64),
    String(String),
}

/// A row of values
pub type Row = Vec<Value>;

/// A row iterator
pub type Rows = Box<dyn Iterator<Item = Result<Row>> + Send>;

/// A column (in a result set, see schema::Column for table columns)
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Column {
    pub name: Option<String>,
}

/// A set of columns
pub type Columns = Vec<Column>;