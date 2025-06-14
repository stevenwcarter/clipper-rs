# CLIP Embedder Library

A Rust library for generating CLIP embeddings from images and text using the Candle framework.

## Features

- **Easy-to-use API**: Simple struct-based interface with `new()` constructor and embedding methods
- **Image embeddings**: Generate 512-dimensional embeddings from image files
- **Text embeddings**: Generate 512-dimensional embeddings from text strings
- **GPU acceleration**: Automatic Metal (macOS) or CUDA support with CPU fallback
- **Model management**: Automatic download and caching of CLIP models from HuggingFace

## Quick Start

### Basic Usage

```rust
use anyhow::Result;
use clipper::ClipEmbedder;

fn main() -> Result<()> {
    // Initialize the CLIP embedder (downloads model on first run)
    let embedder = ClipEmbedder::new(None, None, false)?;
    
    // Get image embedding
    let image_embedding = embedder.get_image_embedding("path/to/image.jpg")?;
    println!("Image embedding length: {}", image_embedding.len()); // 512
    
    // Get text embedding  
    let text_embedding = embedder.get_text_embedding("a photo of a cat")?;
    println!("Text embedding length: {}", text_embedding.len()); // 512
    
    Ok(())
}
```

## API Reference

### `ClipEmbedder`

The main struct that provides access to CLIP embeddings.

#### Constructor

```rust
ClipEmbedder::new(
    model_path: Option<String>,      // Optional custom model path
    tokenizer_path: Option<String>,  // Optional custom tokenizer path  
    use_cpu: bool                    // Force CPU usage if true
) -> Result<ClipEmbedder>
```

**Parameters:**
- `model_path`: Path to a local model file. If `None`, downloads from HuggingFace.
- `tokenizer_path`: Path to a local tokenizer file. If `None`, downloads from HuggingFace.
- `use_cpu`: Set to `true` to force CPU usage, `false` to use GPU if available.

#### Methods

##### `get_image_embedding()`

```rust
fn get_image_embedding(&self, image_path: &str) -> Result<Vec<f32>>
```

Generates a 512-dimensional embedding vector for an image file.

**Parameters:**
- `image_path`: Path to the image file (supports common formats: JPG, PNG, etc.)

**Returns:** `Vec<f32>` with 512 elements representing the image embedding.

##### `get_text_embedding()`

```rust
fn get_text_embedding(&self, text: &str) -> Result<Vec<f32>>
```

Generates a 512-dimensional embedding vector for a text string.

**Parameters:**
- `text`: The input text string to encode

**Returns:** `Vec<f32>` with 512 elements representing the text embedding.

## Example: Computing Similarity

```rust
use anyhow::Result;
use clipper::ClipEmbedder;

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot_product / (norm_a * norm_b)
    }
}

fn main() -> Result<()> {
    let embedder = ClipEmbedder::new(None, None, false)?;
    
    // Compare image and text
    let image_embedding = embedder.get_image_embedding("assets/cat.jpg")?;
    let text_embedding = embedder.get_text_embedding("a photo of a cat")?;
    
    let similarity = cosine_similarity(&image_embedding, &text_embedding);
    println!("Image-text similarity: {:.4}", similarity);
    
    Ok(())
}
```

## Command Line Interface

The library also includes a CLI tool for testing:

```bash
# Use default images and text
cargo run

# Use custom images
cargo run -- --images image1.jpg,image2.jpg

# Use custom text sequences
cargo run -- --sequences "a cat","a dog","a bird"

# Force CPU usage
cargo run -- --cpu

# Use custom model files
cargo run -- --model /path/to/model.safetensors --tokenizer /path/to/tokenizer.json
```

## Model Information

This library uses the **CLIP ViT-Base-Patch32** model by default:
- **Model**: `openai/clip-vit-base-patch32`
- **Embedding size**: 512 dimensions
- **Image input size**: 224x224 pixels (automatically resized)
- **Text context length**: Up to 77 tokens

## Performance Notes

- **First run**: Downloads ~400MB model files from HuggingFace (cached locally)
- **GPU acceleration**: Automatically uses Metal (macOS) or CUDA if available
- **Memory usage**: ~2GB GPU memory for inference
- **Speed**: ~10-50ms per embedding depending on hardware

## Dependencies

- `candle-core`: Tensor operations and model loading
- `candle-transformers`: CLIP model implementation  
- `tokenizers`: Text tokenization
- `image`: Image loading and preprocessing
- `hf-hub`: HuggingFace model downloading

## License

This project uses the same license as the underlying Candle framework.
