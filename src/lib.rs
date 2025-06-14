use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_core::utils::{cuda_is_available, metal_is_available};
use candle_nn::VarBuilder;
use candle_transformers::models::clip;
use tokenizers::Tokenizer;

/// A CLIP model wrapper that provides easy access to image and text embeddings
pub struct ClipEmbedder {
    model: clip::ClipModel,
    tokenizer: Tokenizer,
    config: clip::ClipConfig,
    device: Device,
}

impl ClipEmbedder {
    /// Create a new ClipEmbedder instance
    /// 
    /// # Arguments
    /// * `model_path` - Optional path to the model file. If None, downloads from HuggingFace
    /// * `tokenizer_path` - Optional path to the tokenizer file. If None, downloads from HuggingFace
    /// * `use_cpu` - Whether to force CPU usage instead of GPU
    pub fn new(model_path: Option<String>, tokenizer_path: Option<String>, use_cpu: bool) -> Result<Self> {
        let device = get_device(use_cpu)?;
        
        let model_file = match model_path {
            None => {
                let api = hf_hub::api::sync::Api::new()?;
                let api = api.repo(hf_hub::Repo::with_revision(
                    "openai/clip-vit-base-patch32".to_string(),
                    hf_hub::RepoType::Model,
                    "refs/pr/15".to_string(),
                ));
                api.get("model.safetensors")?
            }
            Some(model) => model.into(),
        };
        
        let tokenizer = get_tokenizer(tokenizer_path)?;
        let config = clip::ClipConfig::vit_base_patch32();
        
        let vb = unsafe { 
            VarBuilder::from_mmaped_safetensors(&[model_file], DType::F32, &device)? 
        };
        let model = clip::ClipModel::new(vb, &config)?;
        
        Ok(ClipEmbedder {
            model,
            tokenizer,
            config,
            device,
        })
    }
    
    /// Generate a 512-dimensional embedding for an image
    /// 
    /// # Arguments
    /// * `image_path` - Path to the image file
    /// 
    /// # Returns
    /// A vector of 512 floating point values representing the image embedding
    pub fn get_image_embedding(&self, image_path: &str) -> Result<Vec<f32>> {
        let img = load_image(image_path, self.config.image_size)?;
        let img = img.unsqueeze(0)?.to_device(&self.device)?;
        let image_features = self.model.get_image_features(&img)?;
        let embedding = image_features.squeeze(0)?.to_vec1::<f32>()?;
        Ok(embedding)
    }
    
    /// Generate a 512-dimensional embedding for a text string
    /// 
    /// # Arguments
    /// * `text` - The text string to encode
    /// 
    /// # Returns
    /// A vector of 512 floating point values representing the text embedding
    pub fn get_text_embedding(&self, text: &str) -> Result<Vec<f32>> {
        let encoding = self.tokenizer.encode(text, true)
            .map_err(anyhow::Error::msg)?;
        let tokens = encoding.get_ids().to_vec();
        
        // Create input tensor with batch dimension
        let input_ids = Tensor::new(vec![tokens], &self.device)?;
        let text_features = self.model.get_text_features(&input_ids)?;
        let embedding = text_features.squeeze(0)?.to_vec1::<f32>()?;
        Ok(embedding)
    }
}

fn get_device(cpu: bool) -> Result<Device> {
    if cpu {
        Ok(Device::Cpu)
    } else if cuda_is_available() {
        Ok(Device::new_cuda(0)?)
    } else if metal_is_available() {
        Ok(Device::new_metal(0)?)
    } else {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            println!(
                "Running on CPU, to run on GPU(metal), build this example with `--features metal`"
            );
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            println!("Running on CPU, to run on GPU, build this example with `--features cuda`");
        }
        Ok(Device::Cpu)
    }
}

fn load_image<T: AsRef<std::path::Path>>(path: T, image_size: usize) -> Result<Tensor> {
    let img = image::ImageReader::open(path)?.decode()?;
    let (height, width) = (image_size, image_size);
    let img = img.resize_to_fill(
        width as u32,
        height as u32,
        image::imageops::FilterType::Triangle,
    );
    let img = img.to_rgb8();
    let img = img.into_raw();
    let img = Tensor::from_vec(img, (height, width, 3), &Device::Cpu)?
        .permute((2, 0, 1))?
        .to_dtype(DType::F32)?
        .affine(2. / 255., -1.)?;
    Ok(img)
}

fn get_tokenizer(tokenizer: Option<String>) -> Result<Tokenizer> {
    let tokenizer_file = match tokenizer {
        None => {
            let api = hf_hub::api::sync::Api::new()?;
            let api = api.repo(hf_hub::Repo::with_revision(
                "openai/clip-vit-base-patch32".to_string(),
                hf_hub::RepoType::Model,
                "refs/pr/15".to_string(),
            ));
            api.get("tokenizer.json")?
        }
        Some(file) => file.into(),
    };
    Tokenizer::from_file(tokenizer_file).map_err(anyhow::Error::msg)
}
