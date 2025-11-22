/// Soft Actor-Critic (stub).
pub struct Sac {
    pub temperature: f32,
    // TODO: add learning rates, target update rates, etc.
}

impl Sac {
    pub fn new() -> Self {
        Self { temperature: 0.1 }
    }
}
