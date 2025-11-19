pub mod utils;

pub trait HasActorId {
    fn actor_id(&self) -> ActorId;
}

#[derive(Clone, Debug)]
pub enum Action {
    Continuous { actor_id: ActorId, value: f32 },
    Discrete { actor_id: ActorId, value: u32 },
}

impl HasActorId for Action {
    fn actor_id(&self) -> ActorId {
        match self {
            Action::Continuous { actor_id, .. } => *actor_id,
            Action::Discrete { actor_id, .. } => *actor_id,
        }
    }
}

pub type ActorId = usize;

pub type Actions = Vec<Action>;

#[derive(Clone, Debug, Default)]
pub struct Observation {
    pub actor_id: ActorId,
    pub data: Vec<f32>,
}

pub type Observations = Vec<Observation>;

#[derive(Debug, Clone, Default)]
pub struct StepResult {
    pub actor_id: ActorId,
    pub reward: f32,
    pub done: bool,
}

pub struct StepResults {
    pub results: Vec<StepResult>,
}

impl StepResults {
    pub fn new(results: Vec<StepResult>) -> Self {
        Self { results }
    }

    pub fn is_done(&self) -> bool {
        for result in &self.results {
            if !result.done {
                return false;
            }
        }
        true
    }
}

pub trait Env {
    fn reset(&mut self) -> Vec<Observation>;

    fn observations(&self) -> Vec<Observation>;

    fn sample_from_model_output(&mut self, actor_id: ActorId, logits: &[f32]) -> Actions;

    fn step(&mut self, action: &Actions) -> StepResults;

    fn number_of_actors(&self) -> usize;
}
