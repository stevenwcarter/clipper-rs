use anyhow::Result;
use clipper::ClipEmbedder;

fn main() -> Result<()> {
    // Initialize the CLIP embedder
    // This handles all the model downloading and setup
    println!("Initializing CLIP embedder...");
    let embedder = ClipEmbedder::new(
        None,    // model_path: Use default (downloads from HuggingFace)
        None,    // tokenizer_path: Use default (downloads from HuggingFace)  
        false,   // use_cpu: Use GPU if available, otherwise CPU
    )?;
    
    println!("CLIP embedder initialized successfully!\n");

    // Example 1: Get image embedding
    println!("=== Image Embedding Example ===");
    let image_path = "assets/stable-diffusion-xl.jpg";
    let image_embedding = embedder.get_image_embedding(image_path)?;
    
    println!("Image: {}", image_path);
    println!("Embedding length: {}", image_embedding.len());
    println!("First 8 dimensions: {:?}", &image_embedding[..8]);
    println!();

    // Example 2: Get text embeddings
    println!("=== Text Embedding Examples ===");
    let texts = vec![
        "a photo of a cat",
        "a dog running in the park", 
        "a beautiful sunset over mountains",
        "a person riding a bicycle",
    ];

    for text in texts {
        let text_embedding = embedder.get_text_embedding(text)?;
        println!("Text: '{}'", text);
        println!("Embedding length: {}", text_embedding.len());
        println!("First 8 dimensions: {:?}", &text_embedding[..8]);
        println!();
    }

    // Example 3: Compare embeddings (compute similarity)
    println!("=== Similarity Example ===");
    let text1 = "a cat sitting on a couch";
    let text2 = "a dog playing in the yard";
    
    let embedding1 = embedder.get_text_embedding(text1)?;
    let embedding2 = embedder.get_text_embedding(text2)?;
    
    // Compute cosine similarity
    let similarity = cosine_similarity(&embedding1, &embedding2);
    println!("Text 1: '{}'", text1);
    println!("Text 2: '{}'", text2);
    println!("Cosine similarity: {:.4}", similarity);

    Ok(())
}

// Helper function to compute cosine similarity between two embeddings
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
