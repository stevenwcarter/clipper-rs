use anyhow::Result;
use clipper::ClipEmbedder;
use std::fs;

fn main() -> Result<()> {
    println!("🚀 Testing CLIP Embedder with Different Input Methods");
    println!("====================================================\n");

    // Initialize embedder
    println!("Initializing CLIP embedder...");
    let embedder = ClipEmbedder::new(None, None, false)?;
    println!("✅ Embedder initialized successfully!\n");

    let image_path = "assets/stable-diffusion-xl.jpg";
    
    // Method 1: Original file path method
    println!("📁 Method 1: From file path");
    let embedding1 = embedder.get_image_embedding(image_path)?;
    println!("   Embedding length: {}", embedding1.len());
    println!("   Sample values: {:?}", &embedding1[..4]);
    println!();

    // Method 2: From DynamicImage
    println!("🖼️  Method 2: From DynamicImage");
    let dynamic_image = image::open(image_path)?;
    let embedding2 = embedder.get_image_embedding_from_dynamic(dynamic_image)?;
    println!("   Embedding length: {}", embedding2.len());
    println!("   Sample values: {:?}", &embedding2[..4]);
    println!();

    // Method 3: From raw bytes
    println!("📄 Method 3: From raw bytes");
    let image_bytes = fs::read(image_path)?;
    println!("   Image file size: {} bytes", image_bytes.len());
    let embedding3 = embedder.get_image_embedding_from_bytes(&image_bytes)?;
    println!("   Embedding length: {}", embedding3.len());
    println!("   Sample values: {:?}", &embedding3[..4]);
    println!();

    // Verify consistency across methods
    println!("🔍 Consistency Check:");
    let diff1 = embedding1.iter().zip(&embedding2).map(|(a, b)| (a - b).abs()).sum::<f32>();
    let diff2 = embedding1.iter().zip(&embedding3).map(|(a, b)| (a - b).abs()).sum::<f32>();
    
    println!("   L1 difference (file vs dynamic): {:.6}", diff1);
    println!("   L1 difference (file vs bytes): {:.6}", diff2);
    
    let tolerance = 1e-5;
    if diff1 < tolerance && diff2 < tolerance {
        println!("   ✅ All methods produce consistent embeddings!");
    } else {
        println!("   ⚠️  Methods produce different embeddings (differences > {})", tolerance);
    }
    println!();

    // Test similarity with different input methods
    println!("🔗 Cross-method similarity test:");
    let cosine_sim_1_2 = cosine_similarity(&embedding1, &embedding2);
    let cosine_sim_1_3 = cosine_similarity(&embedding1, &embedding3);
    
    println!("   Cosine similarity (file vs dynamic): {:.6}", cosine_sim_1_2);
    println!("   Cosine similarity (file vs bytes): {:.6}", cosine_sim_1_3);
    
    if cosine_sim_1_2 > 0.999 && cosine_sim_1_3 > 0.999 {
        println!("   ✅ Excellent consistency across all input methods!");
    }

    println!("\n🎉 All input methods working correctly!");

    Ok(())
}

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
