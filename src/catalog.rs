//! The catalog of CLIP models clipper knows how to load.

use crate::clip::ClipConfig;
use crate::clip::text_model::{Activation, ClipTextConfig};
use crate::clip::vision_model::ClipVisionConfig;

/// Public, lightweight description of a supported model (name + embedding dim).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelInfo {
    pub name: String,
    pub dim: usize,
}

/// Internal full spec used to actually load a model.
pub(crate) struct ModelSpec {
    pub name: &'static str,
    pub hf_repo: &'static str,
    pub revision: &'static str,
    pub weights_file: &'static str,
    pub tokenizer_file: &'static str,
    pub dim: usize,
    pub config: fn() -> ClipConfig,
}

pub(crate) const DEFAULT_MODEL: &str = "openai/clip-vit-base-patch32";

pub(crate) fn specs() -> Vec<ModelSpec> {
    vec![
        ModelSpec {
            name: "openai/clip-vit-base-patch32",
            hf_repo: "openai/clip-vit-base-patch32",
            revision: "refs/pr/15",
            weights_file: "model.safetensors",
            tokenizer_file: "tokenizer.json",
            dim: 512,
            config: clip_vit_base_patch32,
        },
        ModelSpec {
            name: "laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
            hf_repo: "laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
            revision: "main",
            weights_file: "model.safetensors",
            tokenizer_file: "tokenizer.json",
            dim: 768,
            config: laion_vit_l14_224,
        },
    ]
}

pub(crate) fn find_spec(name: &str) -> Option<ModelSpec> {
    specs().into_iter().find(|s| s.name == name)
}

/// All models clipper can load, as `(name, dim)`.
pub fn supported_models() -> Vec<ModelInfo> {
    specs()
        .into_iter()
        .map(|s| ModelInfo { name: s.name.to_string(), dim: s.dim })
        .collect()
}

fn clip_vit_base_patch32() -> ClipConfig {
    ClipConfig {
        text_config: ClipTextConfig {
            vocab_size: 49408,
            embed_dim: 512,
            activation: Activation::QuickGelu,
            intermediate_size: 2048,
            max_position_embeddings: 77,
            pad_with: None,
            num_hidden_layers: 12,
            num_attention_heads: 8,
            projection_dim: 512,
        },
        vision_config: ClipVisionConfig {
            embed_dim: 768,
            activation: Activation::QuickGelu,
            intermediate_size: 3072,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            projection_dim: 512,
            num_channels: 3,
            image_size: 224,
            patch_size: 32,
        },
        logit_scale_init_value: 2.6592,
        image_size: 224,
    }
}

fn laion_vit_l14_224() -> ClipConfig {
    ClipConfig {
        text_config: ClipTextConfig {
            vocab_size: 49408,
            embed_dim: 768,
            activation: Activation::Gelu,
            intermediate_size: 3072,
            max_position_embeddings: 77,
            pad_with: None,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            projection_dim: 768,
        },
        vision_config: ClipVisionConfig {
            embed_dim: 1024,
            activation: Activation::Gelu,
            intermediate_size: 4096,
            num_hidden_layers: 24,
            num_attention_heads: 16,
            projection_dim: 768,
            num_channels: 3,
            image_size: 224,
            patch_size: 14,
        },
        logit_scale_init_value: 2.6592,
        image_size: 224,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_lists_both_models_with_dims() {
        let models = supported_models();
        let by_name = |n: &str| models.iter().find(|m| m.name == n).map(|m| m.dim);
        assert_eq!(by_name("openai/clip-vit-base-patch32"), Some(512));
        assert_eq!(by_name("laion/CLIP-ViT-L-14-laion2B-s32B-b82K"), Some(768));
    }

    #[test]
    fn laion_config_has_l14_shape_and_gelu() {
        let c = laion_vit_l14_224();
        assert_eq!(c.vision_config.embed_dim, 1024);
        assert_eq!(c.vision_config.num_hidden_layers, 24);
        assert_eq!(c.vision_config.patch_size, 14);
        assert_eq!(c.text_config.projection_dim, 768);
        assert!(matches!(c.text_config.activation, Activation::Gelu));
        assert!(matches!(c.vision_config.activation, Activation::Gelu));
    }
}
