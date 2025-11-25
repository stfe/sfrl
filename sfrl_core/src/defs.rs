use burn::prelude::Backend;
use std::collections::HashMap;

pub type ActorId = usize;
pub type ModelId = usize;
#[derive(Clone, Debug, Default)]
pub struct Observation {
    pub data: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct ActionSpace {
    pub discrete: Vec<usize>,
    pub continuous: Vec<(f32, f32)>, // TODO use structure to indicate which value is min and which one is max
}

pub struct Action {
    pub discrete: Vec<u32>,
    pub continuous: Vec<f32>,
}

pub struct StepResult {
    pub observations: HashMap<ActorId, Observation>,
    pub rewards: HashMap<ActorId, f32>,
    pub terminated: HashMap<ActorId, bool>,
    pub truncated: HashMap<ActorId, bool>,
}

/// Environment is the mapper between real physics and model
pub trait Environment {
    fn reset(&mut self) -> HashMap<ActorId, Observation>;

    fn step(&mut self, actions: HashMap<ActorId, Action>) -> StepResult;

    fn model_id(&self, actor_id: ActorId) -> ModelId;

    fn action_space(&self, actor_id: ActorId) -> &ActionSpace;
}

pub trait Model {
    fn model_id(&self) -> usize;
}

pub trait ModelManager<B: Backend, M: Model> {
    fn get_model_by_id(&self, model_id: ModelId) -> &mut M;
    fn register(&mut self, model: M);
}
