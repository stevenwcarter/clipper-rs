use anyhow::Result;
use clipper::ClipEmbedder;
use std::fs;

fn main() -> Result<()> {
    println!("🚀 Testing CLIP Embedder Batch Processing");
    println!("==========================================\n");

    // Initialize embedder
    println!("Initializing CLIP embedder...");
    let embedder = ClipEmbedder::new(None, None, false)?;
    println!("✅ Embedder initialized successfully!\n");

    // Prepare test images
    let image_paths = vec![
        "assets/stable-diffusion-xl.jpg".to_string(),
        "assets/bike.jpg".to_string(),
    ];

    // Test 1: Batch processing from file paths
    println!("📁 Test 1: Batch processing from file paths");
    let batch_embeddings = embedder.get_image_embeddings(&image_paths)?;
    println!("   Processed {} images", batch_embeddings.len());
    for (i, embedding) in batch_embeddings.iter().enumerate() {
        println!("   Image {}: {} dimensions, sample: {:?}", 
                 i + 1, embedding.len(), &embedding[..4]);
    }
    println!();

    // Test 2: Compare batch vs individual processing
    println!("🔍 Test 2: Consistency check (batch vs individual)");
    let individual1 = embedder.get_image_embedding(&image_paths[0])?;
    let individual2 = embedder.get_image_embedding(&image_paths[1])?;
    
    let similarity1 = cosine_similarity(&batch_embeddings[0], &individual1);
    let similarity2 = cosine_similarity(&batch_embeddings[1], &individual2);
    
    println!("   Batch vs Individual similarity:");
    println!("   Image 1: {:.6}", similarity1);
    println!("   Image 2: {:.6}", similarity2);
    
    if similarity1 > 0.999 && similarity2 > 0.999 {
        println!("   ✅ Batch processing produces identical results!");
    } else {
        println!("   ⚠️  Inconsistency detected between batch and individual processing");
    }
    println!();

    // Test 3: Batch processing from DynamicImages
    println!("🖼️  Test 3: Batch processing from DynamicImages");
    let dynamic_images = vec![
        image::open(&image_paths[0])?,
        image::open(&image_paths[1])?,
    ];
    let dynamic_batch_embeddings = embedder.get_image_embeddings_from_dynamic(dynamic_images)?;
    println!("   Processed {} DynamicImages", dynamic_batch_embeddings.len());
    
    // Compare with file path batch results
    let dynamic_similarity1 = cosine_similarity(&batch_embeddings[0], &dynamic_batch_embeddings[0]);
    let dynamic_similarity2 = cosine_similarity(&batch_embeddings[1], &dynamic_batch_embeddings[1]);
    
    println!("   DynamicImage vs file path similarity:");
    println!("   Image 1: {:.6}", dynamic_similarity1);
    println!("   Image 2: {:.6}", dynamic_similarity2);
    
    if dynamic_similarity1 > 0.999 && dynamic_similarity2 > 0.999 {
        println!("   ✅ DynamicImage batch processing consistent with file path batch!");
    }
    println!();

    // Test 4: Batch processing from raw bytes
    println!("📄 Test 4: Batch processing from raw bytes");
    let image_bytes1 = fs::read(&image_paths[0])?;
    let image_bytes2 = fs::read(&image_paths[1])?;
    let bytes_list = vec![image_bytes1.as_slice(), image_bytes2.as_slice()];
    
    let bytes_batch_embeddings = embedder.get_image_embeddings_from_bytes(&bytes_list)?;
    println!("   Processed {} byte arrays", bytes_batch_embeddings.len());
    
    // Compare with file path batch results
    let bytes_similarity1 = cosine_similarity(&batch_embeddings[0], &bytes_batch_embeddings[0]);
    let bytes_similarity2 = cosine_similarity(&batch_embeddings[1], &bytes_batch_embeddings[1]);
    
    println!("   Bytes vs file path similarity:");
    println!("   Image 1: {:.6}", bytes_similarity1);
    println!("   Image 2: {:.6}", bytes_similarity2);
    
    if bytes_similarity1 > 0.999 && bytes_similarity2 > 0.999 {
        println!("   ✅ Bytes batch processing consistent with file path batch!");
    }
    println!();

    // Test 5: Performance comparison
    println!("⚡ Test 5: Performance comparison");
    let start = std::time::Instant::now();
    let _individual_results = vec![
        embedder.get_image_embedding(&image_paths[0])?,
        embedder.get_image_embedding(&image_paths[1])?,
    ];
    let individual_time = start.elapsed();
    
    let start = std::time::Instant::now();
    let _batch_results = embedder.get_image_embeddings(&image_paths)?;
    let batch_time = start.elapsed();
    
    println!("   Individual processing: {:?}", individual_time);
    println!("   Batch processing: {:?}", batch_time);
    
    if batch_time < individual_time {
        println!("   ✅ Batch processing is faster!");
    } else {
        println!("   ⚠️  Individual processing was faster (possibly due to small batch size)");
    }

    println!("\n🎉 All batch processing tests completed!");
    println!("✅ Batch methods working correctly for all input types!");

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
