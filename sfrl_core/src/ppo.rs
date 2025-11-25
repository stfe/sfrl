use crate::defs::{Action, ActorId, Environment, Model, ModelManager, Observation, StepResult};
use burn::nn::{Linear, LinearConfig, Relu};
use burn::optim::Adam;
use burn::prelude::*;
use burn::tensor::activation::{log_softmax, softmax};
use burn::tensor::backend::AutodiffBackend;
use rand::distr::weighted::WeightedIndex;
use rand::distr::Distribution;
use rand::rngs::ThreadRng;
use std::collections::HashMap;
use std::marker::PhantomData;

#[derive(Config)]
pub struct PPOConfig {
    pub clip_ratio: f64,   // Usually 0.2
    pub gamma: f64,        // Discount factor (0.99)
    pub gae_lambda: f64,   // GAE smoothing (0.95)
    pub value_coef: f64,   // Weight for value loss (0.5)
    pub entropy_coef: f64, // Weight for entropy bonus (0.01)
    pub learning_rate: f64,
    pub batch_size: usize,
    pub epochs_per_rollout: usize,
}

struct PPOModelLogits<B: Backend> {
    discrete: Vec<Tensor<B, 2>>,
    continuous_mean: Tensor<B, 2>,
    continuous_log_std: Tensor<B, 2>,
    values: Tensor<B, 2>, // TODO add validation to values size
}

pub trait PPOModel<B: Backend>: Model {
    fn forward(&self, input: Tensor<B, 2>) -> PPOModelLogits<B>;
}

/// A concrete Neural Network module that implements your `Model` trait
/// but also exposes Actor and Critic heads for PPO.
#[derive(Module, Debug)]
pub struct PpoNet<B: Backend> {
    id: usize,
    input_dim: usize,
    fc1: Linear<B>,
    activation: Relu,
    actor_mean: Linear<B>,
    actor_log_std: Linear<B>,
    critic: Linear<B>,
}

impl<B: Backend> PpoNet<B> {
    pub fn new(
        id: usize,
        input_dim: usize,
        hidden_dim: usize,
        action_dim: usize,
        device: &B::Device,
    ) -> Self {
        Self {
            id,
            input_dim,
            fc1: LinearConfig::new(input_dim, hidden_dim).init(device),
            activation: Relu::new(),
            actor_mean: LinearConfig::new(hidden_dim, action_dim).init(device),
            actor_log_std: LinearConfig::new(hidden_dim, action_dim).init(device),
            critic: LinearConfig::new(hidden_dim, 1).init(device),
        }
    }

    /// Returns (Action Mean, Action Log Std, Value)
    pub fn forward(&self, input: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>) {
        let x = self.fc1.forward(input);
        let x = self.activation.forward(x);

        let mean = self.actor_mean.forward(x.clone());
        let log_std = self.actor_log_std.forward(x.clone()); // probably it can be independent layer which does not depend on input parameters
        let value = self.critic.forward(x);

        (mean, log_std, value)
    }
}

// Implementing your Trait
impl<B: Backend> Model for PpoNet<B> {
    fn model_id(&self) -> usize {
        self.id
    }
}

struct TrajectoryPoint {
    observation: Observation,
    action: Action,
    log_prob: f32,
    value: f32,
    reward: f32,
    done: f32, // 1.0 if done, 0.0 otherwise
}

impl TrajectoryPoint {
    fn new(observation: Observation, value: f32) -> Self {
        Self {
            observation,
            action: Action {
                discrete: vec![],
                continuous: vec![],
            },
            log_prob: 0.0,
            value,
            reward: 0.0,
            done: 0.0,
        }
    }
}

struct Trajectory {
    paths: HashMap<ActorId, Vec<TrajectoryPoint>>,
}

impl Trajectory {
    fn new() -> Self {
        Self {
            paths: Default::default(),
        }
    }

    fn push(&mut self, actor_id: ActorId, trajectory_point: TrajectoryPoint) {
        self.paths.insert(actor_id, vec![trajectory_point]);
    }

    fn get_last_mut(&mut self, actor_id: ActorId) -> Option<&mut TrajectoryPoint> {
        match self.paths.get_mut(&actor_id) {
            None => None,
            Some(path) => path.last_mut(),
        }
    }
}

pub struct PPOTrainer<B, ModelType, Manager>
where
    B: AutodiffBackend,
    ModelType: PPOModel<B>,
    Manager: ModelManager<B, ModelType>,
{
    config: PPOConfig,
    model_manager: Manager,
    optimizer: Adam,
    device: B::Device,
    rng: ThreadRng,
    _phantom: PhantomData<ModelType>,
}

impl<B, ModelType, Manager> PPOTrainer<B, ModelType, Manager>
where
    B: AutodiffBackend,
    ModelType: PPOModel<B>,
    Manager: ModelManager<B, ModelType>,
{
    /// Main entry point. Runs one full PPO iteration (Rollout -> GAE -> Update).
    pub fn train_step(&mut self, env: &mut dyn Environment) {
        // 1. Vectorized Data Collection
        let buffer = self.collect_trajectory(env);

        // 2. Generalized Advantage Estimation (GAE)
        let (advantages, returns) = self.compute_gae(&buffer);

        // 3. Policy & Value Update
        self.update_policy(buffer, advantages, returns);
    }

    fn collect_trajectory(&mut self, env: &mut dyn Environment) -> Trajectory {
        let mut obs_map = env.reset();
        let mut trajectory = Trajectory::new();

        for _ in 0..self.config.batch_size {
            let mut model_to_actor = HashMap::new();

            // group by model_id
            for (actor_id, observation) in obs_map {
                model_to_actor
                    .entry(env.model_id(actor_id))
                    .or_insert(Vec::new())
                    .push((actor_id, observation));
            }

            let mut actions = HashMap::new();

            for (model_id, obs) in model_to_actor {
                let model = self.model_manager.get_model_by_id(model_id);
                let obs_tensor = self.obs_map_to_tensor(&obs);
                let inference_result = model.forward(obs_tensor);
                actions.extend(self.sample(env, inference_result, &obs, &mut trajectory));
            }

            // C. Action Sampling & Step

            // D. Handle Next Step. Step is done when all actions are sampled for each ActorId
            let step_result = env.step(actions);

            // E. Store Rewards/Dones and update current observations
            self.process_step_result(&step_result, &mut trajectory);

            // For PPO, if an actor is done, we usually reset it immediately or the env handles it.
            // Here we assume `step_result.observations` contains the *new* state (reset or next).
            obs_map = step_result.observations;
        }

        trajectory
    }

    fn obs_map_to_tensor(&self, obs_map: &Vec<(ActorId, Observation)>) -> Tensor<B, 2> {
        let num_of_actors = obs_map.len();
        let obs_dim = obs_map[0].1.data.len();
        let mut flat_data = Vec::with_capacity(num_of_actors * obs_dim);
        for (actor_id, observation) in obs_map.iter() {
            flat_data.extend_from_slice(&observation.data);
        }
        Tensor::<B, 2>::from_floats(flat_data.as_slice(), &self.device)
            .reshape([num_of_actors, obs_dim])
    }

    /// Helper: Samples actions, calculates log_probs, and prepares Action HashMap
    fn sample(
        &mut self,
        env: &dyn Environment, // TODO Unused in this snippet but often needed for context
        logits: PPOModelLogits<B>,
        obs: &Vec<(ActorId, Observation)>,
        trajectory: &mut Trajectory,
    ) -> HashMap<ActorId, Action> {
        // TODO add panic statements to verify that all dimensions are correct
        // PLAN:
        // * Calculate discrete actions parameters
        // * Calculate continuous actions parameters
        // * Return results mapped to actor id

        let values_vec = Self::tensor_to_vec1d(logits.values);
        for (i, (actor_id, observation)) in obs.iter().enumerate() {
            let trajectory_point = TrajectoryPoint::new(observation.clone(), values_vec[i]);
            trajectory.push(*actor_id, trajectory_point);
        }

        // sample discrete actions
        if !logits.discrete.is_empty() {
            for single_action_tensor in logits.discrete.into_iter() {
                let probs = softmax(single_action_tensor.clone(), 1);
                let all_log_probs = log_softmax(single_action_tensor, 1);
                let probs_vec = Self::tensor_to_vec2d(probs);
                let all_log_probs_vec = Self::tensor_to_vec2d(all_log_probs);
                for (i, (actor_id, obs)) in obs.iter().enumerate() {
                    let actor_probs = &probs_vec[i];
                    let dist = WeightedIndex::new(actor_probs).unwrap();
                    let action_index = dist.sample(&mut self.rng) as u32;
                    let log_prob = all_log_probs_vec[i][action_index as usize];
                    let tr = trajectory.get_last_mut(*actor_id).unwrap();
                    tr.action.discrete.push(action_index);
                    tr.log_prob += log_prob;
                }
            }
        }

        // sample continuous actions
        if logits.continuous_mean.dims()[0] > 0 {
            if logits.continuous_mean.dims()[0] != logits.continuous_log_std.dims()[0] {
                panic!(
                    "Continuous mean and log_std arrays should have the same size. Mean size: {}, Std size: {}",
                    logits.continuous_mean.dims()[0],
                    logits.continuous_log_std.dims()[0]
                );
            }

            let std = logits.continuous_log_std.exp();
            // Vectorized sampling: [num_actors, 1]
            let noise = Tensor::<B, 2>::random_like(
                &logits.continuous_mean,
                burn::tensor::Distribution::Normal(0.0, 1.0),
            );
            let action_tensor = logits.continuous_mean.clone() + (noise * std.clone());

            // Calculate Log Probs immediately (Vectorized)
            // log_prob = -0.5 * ((x - mu)/sigma)^2 - log(sigma) - 0.5 * log(2pi)
            let var = std.clone().powf_scalar(2.0);
            let diff = action_tensor.clone() - logits.continuous_mean;
            let log_probs_tensor =
                diff.powf_scalar(2.0).neg().div(var.mul_scalar(2.0)) - std.log() - 0.9189385;

            let mut actions_vec = Self::tensor_to_vec2d(action_tensor);
            let log_probs_vec = Self::tensor_to_vec1d(log_probs_tensor.sum_dim(1));

            for (i, (actor_id, _)) in obs.iter().enumerate() {
                let action_limits = env.action_space(*actor_id);
                let actions = &mut actions_vec[i];
                for ai in 0..action_limits.continuous.len() {
                    let (min_value, max_value) =
                        (action_limits.continuous[i].0, action_limits.continuous[i].1);
                    actions[ai] = actions[ai].clamp(min_value, max_value);
                }
                let tr = trajectory.get_last_mut(*actor_id).unwrap();
                tr.action.continuous = actions.clone();
                tr.log_prob += log_probs_vec[i];
            }
        }

        // Prepare Actions HashMap for the Environment
        let mut actions_map = HashMap::new();
        // TODO fill out action map from Trajectory

        actions_map
    }

    /// Helper: Processes StepResult to store rewards and done flags
    fn process_step_result(&self, result: &StepResult, buffer: &mut RolloutBuffer) {
        for &id in &self.sorted_actor_ids {
            let r = *result.rewards.get(&id).unwrap_or(&0.0);
            let term = *result.terminated.get(&id).unwrap_or(&false);
            let trunc = *result.truncated.get(&id).unwrap_or(&false);

            buffer.rewards.push(r);
            buffer.dones.push(if term || trunc { 1.0 } else { 0.0 });
        }
    }

    // -------------------------------------------------------------------------
    // Phase 2: GAE Calculation
    // -------------------------------------------------------------------------

    fn compute_gae(&self, buffer: &RolloutBuffer) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let total_steps = buffer.num_steps;
        let num_actors = buffer.num_actors;
        let len = total_steps * num_actors;

        let mut advantages = vec![0.0; len];
        let mut returns = vec![0.0; len];

        // We compute GAE per actor column
        // Data is stored row-major: [Step 0 (All Actors), Step 1 (All Actors)...]
        // Index = step * num_actors + actor_idx

        for actor_idx in 0..num_actors {
            let mut gae = 0.0;
            let mut next_value = 0.0; // Value of state beyond the horizon (approx 0 or bootstrap)

            for step in (0..total_steps).rev() {
                let idx = step * num_actors + actor_idx;

                let reward = buffer.rewards[idx];
                let value = buffer.values[idx];
                let done = buffer.dones[idx];

                // If done, next_value is irrelevant (masked by 1 - done)
                let mask = 1.0 - done;

                let delta = reward + (self.config.gamma as f32 * next_value * mask) - value;
                gae =
                    delta + (self.config.gamma as f32 * self.config.gae_lambda as f32 * mask * gae);

                advantages[idx] = gae;
                returns[idx] = gae + value;

                next_value = value;
            }
        }

        let adv_tensor = Tensor::from_floats(advantages.as_slice(), &self.device).reshape([len, 1]);
        let ret_tensor = Tensor::from_floats(returns.as_slice(), &self.device).reshape([len, 1]);

        // Normalize Advantages (Critical for PPO stability)
        let adv_mean = adv_tensor.clone().mean();
        let adv_std = adv_tensor.clone().std(0) + 1e-8;
        let adv_normalized = (adv_tensor - adv_mean) / adv_std;

        (adv_normalized, ret_tensor)
    }

    // -------------------------------------------------------------------------
    // Phase 3: Optimization Update
    // -------------------------------------------------------------------------

    fn update_policy(
        &mut self,
        buffer: RolloutBuffer,
        advantages: Tensor<B, 2>,
        returns: Tensor<B, 2>,
    ) {
        // 1. Create Full Batch Tensors
        // These are massive tensors containing (Steps * Actors) rows
        let obs_tensor = Tensor::from_floats(buffer.obs.as_slice(), &self.device)
            .reshape([buffer.obs.len() / buffer.obs_dim, buffer.obs_dim]);

        let acts_tensor = Tensor::from_floats(buffer.actions.as_slice(), &self.device)
            .reshape([buffer.actions.len(), 1]);

        let old_log_probs = Tensor::from_floats(buffer.log_probs.as_slice(), &self.device)
            .reshape([buffer.log_probs.len(), 1])
            .detach(); // Important: Detach old policies

        let advantages = advantages.detach();
        let returns = returns.detach();

        // 2. Epoch Loop
        for _ in 0..self.config.epochs_per_rollout {
            // In a real implementation, you would shuffle indices here and
            // loop over mini-batches to save VRAM and improve convergence.
            // For brevity, we do a full-batch update here.

            let loss = self.compute_loss(
                obs_tensor.clone(),
                acts_tensor.clone(),
                old_log_probs.clone(),
                advantages.clone(),
                returns.clone(),
            );

            let grads = Grads::from_loss(loss);
            self.model_manager =
                self.optimizer
                    .step(self.config.learning_rate, self.model_manager, grads);
        }
    }

    fn compute_loss(
        &self,
        obs: Tensor<B, 2>,
        actions: Tensor<B, 2>,
        old_log_probs: Tensor<B, 2>,
        advantages: Tensor<B, 2>,
        returns: Tensor<B, 2>,
    ) -> Tensor<B, 1> {
        let (mean, log_std, values) = self.model_manager.forward_full(obs);
        let std = log_std.exp();

        // 1. Calculate New Log Probs
        let var = std.clone().powf(2.0);
        let diff = actions - mean;
        let new_log_probs =
            diff.powf(2.0).neg().div(var.mul_scalar(2.0)) - std.clone().ln() - 0.9189385;

        // 2. Ratio
        let ratio = (new_log_probs - old_log_probs).exp();

        // 3. Surrogate Loss
        let surr1 = ratio.clone() * advantages.clone();
        let eps = self.config.clip_ratio;
        let ratio_clipped = ratio.clamp(1.0 - eps, 1.0 + eps);
        let surr2 = ratio_clipped * advantages;
        let policy_loss = Tensor::min_pair(surr1, surr2).mean().neg();

        // 4. Value Loss
        let value_loss = (values - returns).powf(2.0).mean();

        // 5. Entropy Loss
        let entropy = (std.ln() + 0.5 + 0.9189385).mean();
        let entropy_loss = entropy.mul_scalar(self.config.entropy_coef).neg();

        // Sum
        policy_loss + value_loss.mul_scalar(self.config.value_coef) + entropy_loss
    }

    fn tensor_to_vec2d(tensor: Tensor<B, 2>) -> Vec<Vec<f32>> {
        let dims = tensor.dims();
        let cols = dims[1]; // Number of columns (items per row)

        let flat_data = tensor
            .into_data() // Syncs from GPU/Device to CPU
            .convert::<f32>() // Ensures data is f32 (casts if necessary)
            .into_vec::<f32>() // Returns Result<Vec<f32>>
            .unwrap(); // Unwraps the result

        flat_data.chunks(cols).map(|chunk| chunk.to_vec()).collect()
    }

    fn tensor_to_vec1d(tensor: Tensor<B, 2>) -> Vec<f32> {
        tensor
            .into_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .unwrap()
    }
}
