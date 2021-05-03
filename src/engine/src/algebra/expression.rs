// Copyright (c) 2021 Xiaozhe Yao et al.
// 
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

use super::{Row, Value};
use crate::error::{Error, Result};

use regex::Regex;
use serde_derive::{Deserialize, Serialize};
use std::fmt::{self, Display};
use std::mem::replace;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Expression {
    // Values
    Constant(Value),
    Field(usize, Option<(Option<String>, String)>),
    
    // Logical Operations
    And(Box<Expression>, Box<Expression>),
    Not(Box<Expression>),
    Or(Box<Expression>, Box<Expression>),

    // Comparison Operations
    Equal(Box<Expression>, Box<Expression>),
    GreaterThan(Box<Expression>, Box<Expression>),
    LessThan(Box<Expression>, Box<Expression>),
    IsNull(Box<Expression>),
    Assert(Box<Expression>),

    // Mathematical Operations
    Add(Box<Expression>, Box<Expression>),
    Divide(Box<Expression>, Box<Expression>),
    Exponentiate(Box<Expression>, Box<Expression>),
    Factorial(Box<Expression>),
    Modulo(Box<Expression>, Box<Expression>),
    Multiply(Box<Expression>, Box<Expression>),
    Negate(Box<Expression>),
    Subtract(Box<Expression>, Box<Expression>),

    // String Operations
    Like(Box<Expression>, Box<Expression>),
}

impl Expression {
    pub fn evaluate(&self, row: Option<&Row>) -> Result<Value> {
        use Value::*;
        Ok(match self {
            // Constants
            Self::Constance(c) => c.clone(),
            Self::Field(i,_) => row.and_then(|row| row.get(*i).cloned()).unwrap_or(Null),

            // Logical operations
            Self::And(lhs, rhs)=> match(lhs.evaluate(row)?, rhs.evaluate(row)?){
                (Boolean(lhs), Boolean(rhs)) => Boolean(lhs && rhs),
                (Boolean(lhs), Null) if !lhs => Boolean(false),
            }
        })
    }
}

