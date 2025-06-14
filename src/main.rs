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

    // Test image embedding
    println!("📸 Testing image embedding...");
    let image_embedding = embedder.get_image_embedding(&vec_imgs[0])?;
    println!("   Image: {}", vec_imgs[0]);
    println!("   Embedding length: {}", image_embedding.len());
    println!("   Sample dimensions: {:?}", &image_embedding[..8]);
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
        println!("   Sample dimensions: {:?}", &text_embedding[..8]);
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
