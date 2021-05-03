# Anatomy

In the architecture, there are essentially four layers in EurusDB, from top to bottom:

* Query Layer: Processes SQL and programmatic queries.
* Consensus Layer: Handles cluster coordination and state machine replication.
* Storage Layer: Stores the data, manages transactions, controls the indexes.
* Computing Layer: Performs matrix multiplication, gradient descent and other machine learning algorithms.

Surrounding all these components is a server application to perform network communication, configurations, logging and other chores.

## Storage Layer 

### Key/Value Storage

In order to support the iteration over a range in order, we apply several implications:

* The data are stored in the order of the key.
* The keys use an order-preserving encoding to allow range scans.

The storage uses composited keys made up of several possibly variable-length values.

* `bool`: `0x00` for `false`, `0x01` for `true`.
* `Vec<u8>`: terminated with `0x0000`, `0x00` escaped as `0x00ff`.
* `String`:  like `Vec<u8>`.
* `u64`: Big-endian binary encoding.
* `i64`: Big-endian binary encoding, sign bit flipped.
* `f64`: Big-endian binary encoding, sign bit flipped if `+`, all flipped if `-`.
* `sql::Value`: As above, with type prefix `0x00`=`Null`, `0x01`=`Boolean`, `0x02`=`Float`,
  `0x03`=`Integer`, `0x04`=`String`

### MVCC Transactions



## Consensus Layer

### Raft Consensus Protocol

## Query Layer

### SQL

## Server and Clients

