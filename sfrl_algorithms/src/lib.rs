//! Algorithms crate: PPO, SAC, and shared training abstractions.
//! Keep algorithms independent of any specific engine.
//!
//! Features
//! - `ppo` (default): enables the `ppo` module and `Ppo` stub.
//! - `sac` (default): enables the `sac` module and `Sac` stub.
//!
//! Downstream crates can disable defaults and enable only what they need:
//! ```toml
//! # Enable only PPO
//! sfrl_algorithms = { version = "0.1", default-features = false, features = ["ppo"] }
//!
//! # Or only SAC
//! sfrl_algorithms = { version = "0.1", default-features = false, features = ["sac"] }
//! ```
//!
//! TODOs:
//! - Implement proper trajectory collection and update rules.
//! - Add Model/Policy traits once a specific ML backend is chosen.
//! - Add config structs and serialization.

use sfrl_core::Env;


// Feature-gated algorithm modules
#[cfg(feature = "ppo")]
pub mod ppo;

pub mod definitions;
#[cfg(feature = "sac")]
pub mod sac;
