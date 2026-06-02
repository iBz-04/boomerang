//! Boomerang core domain types and shared traits.
//!
//! This crate defines the fundamental data structures used across the
//! boomerang workspace: video chunks, embeddings, search results, and
//! the trait interfaces for embedders and vector stores.

pub mod chunk;
pub mod embedding;
pub mod error;
pub mod search;
pub mod store;
pub mod types;
