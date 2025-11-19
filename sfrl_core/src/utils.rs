use std::cmp::Ordering::Less;

pub fn argmax(data: &[f32]) -> usize {
    data.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Less))
        .map(|(index, _)| index)
        .unwrap_or(0)
}

pub fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }

    let max_logit = logits
        .iter()
        .fold(f32::NEG_INFINITY, |a, &b| f32::max(a, b));

    let mut exponents: Vec<f32> = logits.iter().map(|&l| (l - max_logit).exp()).collect();
    let sum_exponents: f32 = exponents.iter().sum();
    if sum_exponents == 0.0 {
        return vec![0.0; logits.len()];
    }
    exponents.into_iter().map(|p| p / sum_exponents).collect()
}

pub fn categorical_sample(logits: &[f32], rng: &impl Fn() -> f32) -> usize {
    let probs = softmax(logits);
    let rng_value: f32 = rng();
    let mut acc_prob = 0.0;
    for (index, &prob) in probs.iter().enumerate() {
        acc_prob += prob;
        if rng_value < acc_prob {
            return index;
        }
    }
    logits.len() - 1
}

pub fn gaussian_sample(mean: f32, log_sigma: f32, rng: &impl Fn() -> f32) -> f32 {
    let sigma = log_sigma.exp();
    let noise = rng() * 2.0 - 1.0; // rng should be normally distributed mean 0 and std 1
    let sample = mean + sigma * noise;
    sample
}
