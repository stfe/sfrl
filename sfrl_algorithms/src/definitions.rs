use sfrl_core::{Actions, Env, Observation, Observations, StepResults};

pub type ActionLogits = Vec<f32>;
pub type ValueLogit = f32;
pub type LogProb = f32;

pub trait Model {
    fn infer(&self, observation: &Observation) -> (ActionLogits, ValueLogit);
}

pub trait Agent {
    fn sample(
        &self,
        env: &mut dyn Env,
        model: &mut dyn Model,
        observations: Observations,
    ) -> Actions;
    fn step(&self, env: &mut dyn Env, actions: &Actions) -> StepResults;
}

pub struct Experience {
    pub observation: Observation,
    pub action: Actions,
    pub reward: f32,
    pub log_prob: f32,
    pub advantage: f32,
}

/// Common algorithm interface.
pub trait Algorithm {
    /// Train the algorithm on the given environment for `steps` time steps.
    /// A policy function must be provided to map observations to actions.
    fn train<E: Env>(
        &mut self,
        env: &mut E,
        model: &mut dyn Model,
        steps: usize,
        agent: &mut dyn Agent,
    );
}
