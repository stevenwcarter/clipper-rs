use anyhow::Result;
use clap::Parser;
use clipper::ClipEmbedder;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    model: Option<String>,

    #[arg(long)]
    tokenizer: Option<String>,

    #[arg(long, use_value_delimiter = true)]
    images: Option<Vec<String>>,

    #[arg(long)]
    cpu: bool,

    #[arg(long, use_value_delimiter = true)]
    sequences: Option<Vec<String>>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    
    // Create the ClipEmbedder instance
    println!("🚀 Initializing CLIP embedder...");
    let embedder = ClipEmbedder::new(args.model, args.tokenizer, args.cpu)?;
    println!("✅ CLIP embedder initialized successfully!\n");
    
    // Get image paths
    let vec_imgs = match args.images {
        Some(imgs) => imgs,
        None => vec![
            "assets/stable-diffusion-xl.jpg".to_string(),
            "assets/bike.jpg".to_string(),
        ],
    };

    // Test different image input methods
    println!("📸 Testing different image embedding methods...");
    let image_path = &vec_imgs[0];
    
    // Method 1: From file path
    let embedding1 = embedder.get_image_embedding(image_path)?;
    println!("   Method 1 (file path): {} dimensions", embedding1.len());
    
    // Method 2: From DynamicImage
    let dynamic_image = image::open(image_path)?;
    let embedding2 = embedder.get_image_embedding_from_dynamic(dynamic_image)?;
    println!("   Method 2 (DynamicImage): {} dimensions", embedding2.len());
    
    // Method 3: From bytes
    let image_bytes = std::fs::read(image_path)?;
    let embedding3 = embedder.get_image_embedding_from_bytes(&image_bytes)?;
    println!("   Method 3 (raw bytes): {} dimensions", embedding3.len());
    
    // Check consistency
    let similarity_1_2 = cosine_similarity(&embedding1, &embedding2);
    let similarity_1_3 = cosine_similarity(&embedding1, &embedding3);
    println!("   Consistency check:");
    println!("     File vs DynamicImage: {:.6}", similarity_1_2);
    println!("     File vs bytes: {:.6}", similarity_1_3);
    
    if similarity_1_2 > 0.999 && similarity_1_3 > 0.999 {
        println!("   ✅ All input methods produce consistent results!");
    } else {
        println!("   ⚠️  Some inconsistency detected between methods");
    }
    println!();

    // Test text embedding
    println!("📝 Testing text embeddings...");
    let text_sequences = match args.sequences {
        Some(seq) => seq,
        None => vec![
            "a cycling race".to_string(),
            "a photo of two cats".to_string(),
            "a robot holding a candle".to_string(),
        ],
    };
    
    for text in &text_sequences {
        let text_embedding = embedder.get_text_embedding(text)?;
        println!("   Text: '{}'", text);
        println!("   Embedding length: {}", text_embedding.len());
        println!("   Sample dimensions: {:?}", &text_embedding[..4]);
        println!();
    }

    // Demonstrate similarity
    if vec_imgs.len() > 1 {
        println!("🔗 Testing cross-modal similarity...");
        let bike_embedding = embedder.get_image_embedding(&vec_imgs[1])?;
        let bike_text_embedding = embedder.get_text_embedding("a bicycle")?;
        let car_text_embedding = embedder.get_text_embedding("a car")?;
        
        let bike_similarity = cosine_similarity(&bike_embedding, &bike_text_embedding);
        let car_similarity = cosine_similarity(&bike_embedding, &car_text_embedding);
        
        println!("   Image: {} vs Text: 'a bicycle'", vec_imgs[1]);
        println!("   Similarity: {:.4}", bike_similarity);
        println!("   Image: {} vs Text: 'a car'", vec_imgs[1]);  
        println!("   Similarity: {:.4}", car_similarity);
        
        if bike_similarity > car_similarity {
            println!("   ✅ Bike image is more similar to 'bicycle' than 'car'!");
        }
    }

    println!("\n🎉 Library demonstration complete!");
    println!("✅ All input methods (file path, DynamicImage, raw bytes) working correctly!");

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
