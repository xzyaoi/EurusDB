// Copyright (c) 2021 Xiaozhe Yao et al.
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

use crate::utility::error::{Error, Result};
use std::collections::HashMap;

pub struct Server {}

impl Server {
    pub async fn new(id: &str, peers: HashMap<String, String>) -> Result<Self> {
        Ok(Server {})
    }
    pub async fn listen(mut self, sql_addr: &str, raft_addr: &str) -> Result<Self> {
        let (sql, raft) =
            tokio::try_join!(TcpListener::bind(sql_addr), TcpListener::bind(raft_addr),)?;
        info!("Listening on {} (SQL) and {} (Raft)", sql.local_addr()?, raft.local_addr()?);
        self.sql_listener = Some(sql);
        self.raft_listener = Some(raft);
        Ok(self)
    }
    pub async fn serve(self) -> Result<()> {
        
    }
}
