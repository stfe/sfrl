//! Environment adapters that expose engines via the `sfrl_core::Env` trait.
//! Engines stay clean and readable; envs do the minimal glue.
//!
//! TODO:
//! - Add adapters for more engines.
//! - Provide wrappers for continuous action spaces.
//! - Consider feature flags per engine.

pub mod tictactoe_env;
