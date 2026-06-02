//! Qdrant collection naming for embedding-space isolation.

use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use boomerang_core::error::CoreError;
use boomerang_core::types::{EmbeddingBackend, EmbeddingSpace};

const COLLECTION_PREFIX: &str = "boomerang_chunks";

pub(crate) fn collection_name(space: &EmbeddingSpace) -> String {
    match &space.model {
        Some(model) => format!(
            "{COLLECTION_PREFIX}__{}__{}",
            space.backend.as_str(),
            URL_SAFE_NO_PAD.encode(model),
        ),
        None => format!("{COLLECTION_PREFIX}__{}", space.backend.as_str()),
    }
}

pub(crate) fn parse_collection_name(
    name: &str,
    dimensions: usize,
) -> Result<Option<EmbeddingSpace>, CoreError> {
    if !name.starts_with(COLLECTION_PREFIX) {
        return Ok(None);
    }

    let suffix = match name
        .strip_prefix(COLLECTION_PREFIX)
        .and_then(|value| value.strip_prefix("__"))
    {
        Some(s) => s,
        None => return Ok(None),
    };

    let mut parts = suffix.splitn(2, "__");
    let backend = parts
        .next()
        .ok_or_else(|| CoreError::Store(format!("missing backend in collection name: {name}")))?
        .parse::<EmbeddingBackend>()
        .map_err(CoreError::Config)?;
    let model = parts
        .next()
        .map(|encoded| {
            URL_SAFE_NO_PAD
                .decode(encoded)
                .map_err(|error| {
                    CoreError::Store(format!(
                        "invalid model encoding in collection name {name}: {error}"
                    ))
                })
                .and_then(|bytes| {
                    String::from_utf8(bytes).map_err(|error| {
                        CoreError::Store(format!(
                            "invalid utf-8 model encoding in collection name {name}: {error}"
                        ))
                    })
                })
        })
        .transpose()?;

    Ok(Some(EmbeddingSpace::new(backend, model, dimensions)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_collection_name_round_trip_with_model() {
        let space = EmbeddingSpace::new(
            EmbeddingBackend::QwenCloud,
            Some("org/model:v1".to_string()),
            768,
        );
        let name = collection_name(&space);
        let decoded = parse_collection_name(&name, 768).unwrap().unwrap();
        assert_eq!(decoded, space);
    }

    #[test]
    fn test_collection_name_round_trip_without_model() {
        let space = EmbeddingSpace::new(EmbeddingBackend::Gemini, None, 3072);
        let name = collection_name(&space);
        let decoded = parse_collection_name(&name, 3072).unwrap().unwrap();
        assert_eq!(decoded, space);
    }
}
