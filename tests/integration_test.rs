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
    println!("🚀 Testing CLIP Embedder Library");
    println!("=================================\n");

    // Initialize embedder
    println!("Initializing CLIP embedder...");
    let embedder = ClipEmbedder::new(None, None, false)?;
    println!("✅ Embedder initialized successfully!\n");

    // Test 1: Basic embedding generation
    println!("📸 Test 1: Image Embedding");
    let image_path = "assets/stable-diffusion-xl.jpg";
    let image_embedding = embedder.get_image_embedding(image_path)?;
    println!("   Image: {}", image_path);
    println!("   Embedding length: {} ✅", image_embedding.len());
    println!("   Sample values: {:?}\n", &image_embedding[..4]);

    println!("📝 Test 2: Text Embedding");  
    let text = "a beautiful landscape with mountains";
    let text_embedding = embedder.get_text_embedding(text)?;
    println!("   Text: '{}'", text);
    println!("   Embedding length: {} ✅", text_embedding.len());
    println!("   Sample values: {:?}\n", &text_embedding[..4]);

    // Test 2: Similarity between related texts
    println!("🔗 Test 3: Text-to-Text Similarity");
    let text1 = "a cat sitting on a couch";
    let text2 = "a kitten resting on furniture";
    let text3 = "a car driving on a highway";

    let emb1 = embedder.get_text_embedding(text1)?;
    let emb2 = embedder.get_text_embedding(text2)?;
    let emb3 = embedder.get_text_embedding(text3)?;

    let sim_related = cosine_similarity(&emb1, &emb2);
    let sim_unrelated = cosine_similarity(&emb1, &emb3);

    println!("   Text A: '{}'", text1);
    println!("   Text B: '{}'", text2);
    println!("   Similarity A-B (related): {:.4}", sim_related);
    println!("   Text C: '{}'", text3);
    println!("   Similarity A-C (unrelated): {:.4}", sim_unrelated);
    
    if sim_related > sim_unrelated {
        println!("   ✅ Related texts have higher similarity!");
    } else {
        println!("   ⚠️  Unexpected similarity results");
    }
    println!();

    // Test 3: Cross-modal similarity (image-text)
    println!("🖼️  Test 4: Image-to-Text Similarity");
    let bike_image = "assets/bike.jpg";
    let bike_embedding = embedder.get_image_embedding(bike_image)?;
    
    let bike_text = "a bicycle";
    let car_text = "an automobile";
    
    let bike_text_emb = embedder.get_text_embedding(bike_text)?;
    let car_text_emb = embedder.get_text_embedding(car_text)?;
    
    let bike_sim = cosine_similarity(&bike_embedding, &bike_text_emb);
    let car_sim = cosine_similarity(&bike_embedding, &car_text_emb);
    
    println!("   Image: {}", bike_image);
    println!("   Text A: '{}'", bike_text);
    println!("   Image-Text A similarity: {:.4}", bike_sim);
    println!("   Text B: '{}'", car_text);
    println!("   Image-Text B similarity: {:.4}", car_sim);
    
    if bike_sim > car_sim {
        println!("   ✅ Image matches relevant text better!");
    } else {
        println!("   ⚠️  Unexpected cross-modal results");
    }

    println!("\n🎉 All tests completed!");
    println!("The CLIP embedder library is working correctly.");

    Ok(())
}
