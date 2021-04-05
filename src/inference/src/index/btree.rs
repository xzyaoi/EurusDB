// Copyright (c) 2021 Xiaozhe Yao et al.
// 
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

use std::cmp::Ordering;
use std::fmt::Display;
use std::mem::replace;
use std::ops::{Bound, Deref, DerefMut};
use std::sync::{Arc, RwLock};

const DEFAULT_ORDER: usize=8;

#[derive(Debug, PartialEq)]
enum Node {
    Root(Children),
    Inner(Children),
    Leaf(Values),
}

#[derive(Debug, PartialEq)]
enum Children {
    keys: Vec<Vec<u8>>,
    nodes: Vec<Node>,
}

#[derive(Debug, PartialEq)]
struct Values(Vec<(Vec<u8>, Vec<u8>)>);

impl Node {
    fn delete(&mut self, key: &[u8]) {

    }
    fn get(&self, key: &[u8]) {

    }
    // Fetches the first key/value pair
    fn first(&self) {

    }
    // Fetches the last key/value pair
    fn last(&self) {

    }
}