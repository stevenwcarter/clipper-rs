use clipper::ClipEmbedder;

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb)
}

#[test]
#[ignore = "downloads ~1.7GB LAION ViT-L/14 weights; run manually"]
fn laion_l14_ranks_matching_caption_higher() {
    let m = ClipEmbedder::from_model("laion/CLIP-ViT-L-14-laion2B-s32B-b82K", true)
        .expect("load L/14");
    assert_eq!(m.dim(), 768);

    let img = m.get_image_embedding("assets/bike.jpg").expect("image embed");
    assert_eq!(img.len(), 768);

    let matching = m
        .get_text_embedding("a photograph of a bicycle")
        .expect("text embed");
    let wrong = m
        .get_text_embedding("a bowl of fruit on a table")
        .expect("text embed");

    let s_match = cosine(&img, &matching);
    let s_wrong = cosine(&img, &wrong);
    eprintln!("match={s_match:.4} wrong={s_wrong:.4}");
    assert!(
        s_match > s_wrong,
        "expected matching caption to score higher: match={s_match} wrong={s_wrong}"
    );
}
