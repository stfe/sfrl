use crate::defs::{
    Action, ActionSpace, ActorId, Environment, Model, ModelId, ModelManager, Observation,
    StepResult,
};
use burn::module::AutodiffModule;
use burn::nn::{Linear, LinearConfig, Relu};
use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::{Adam, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::activation::{log_softmax, softmax};
use burn::tensor::backend::AutodiffBackend;
use rand::distr::weighted::WeightedIndex;
use rand::distr::Distribution;
use rand::rngs::ThreadRng;
use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;

#[derive(Config, Debug)]
pub struct PPOConfig {
    pub clip_ratio: f64,   // Usually 0.2
    pub gamma: f64,        // Discount factor (0.99)
    pub gae_lambda: f64,   // GAE smoothing (0.95)
    pub value_coef: f64,   // Weight for value loss (0.5)
    pub entropy_coef: f64, // Weight for entropy bonus (0.01)
    pub learning_rate: f64,
    pub batch_size: usize, // TODO implement update policy only when batch size is accumulated
    pub max_steps: usize,  // TODO restrict number of environment steps
    pub epochs_per_rollout: usize,
}

struct PPOModelLogits<B: Backend> {
    discrete: Vec<Tensor<B, 2>>,
    continuous_mean: Tensor<B, 2>,
    continuous_log_std: Tensor<B, 2>,
    values: Tensor<B, 2>, // TODO add validation to values size
}

struct TrainingData<B: Backend> {
    action_space: ActionSpace,
    obs: Tensor<B, 2>,
    discrete_actions: Tensor<B, 2, Int>,
    continuous_actions: Tensor<B, 2>,
    log_probs: Tensor<B, 2>,
    advantages: Tensor<B, 2>,
    returns: Tensor<B, 2>,
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
    Manager: ModelManager<B, ModelType> + AutodiffModule<B>,
{
    config: PPOConfig,
    model_manager: Manager,
    optimizer: OptimizerAdaptor<Adam, Manager, B>,
    device: B::Device,
    rng: ThreadRng,
    _phantom: PhantomData<ModelType>,
}

impl<B, ModelType, Manager> PPOTrainer<B, ModelType, Manager>
where
    B: AutodiffBackend,
    ModelType: PPOModel<B>,
    Manager: ModelManager<B, ModelType> + AutodiffModule<B>,
{
    /// Main entry point. Runs one full PPO iteration (Rollout -> GAE -> Update).
    pub fn train_step(&mut self, env: &mut dyn Environment) {
        // 1. Vectorized Data Collection
        let trajectory = self.collect_trajectory(env);

        // 2. Generalized Advantage Estimation (GAE)
        let training_data = self.build_training_data(env, &trajectory);

        // 3. Policy & Value Update
        self.update_policy(training_data);
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
            let step_result = env.step(actions);
            self.process_step_result(&step_result, &mut trajectory);
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

        let mut actions = HashMap::new();
        for (actor_id, _) in obs {
            let tr = trajectory.get_last_mut(*actor_id).unwrap();
            actions.insert(*actor_id, tr.action.clone());
        }
        actions
    }

    fn process_step_result(&self, result: &StepResult, trajectory: &mut Trajectory) {
        for (actor_id, r) in result.rewards.iter() {
            let term = *result.terminated.get(&actor_id).unwrap_or(&false);
            let trunc = *result.truncated.get(&actor_id).unwrap_or(&false);
            let tr = trajectory.get_last_mut(*actor_id).unwrap();
            tr.reward = *r;
            tr.done = if term || trunc { 1.0 } else { 0.0 };
        }
    }

    // -------------------------------------------------------------------------
    // Phase 2: Preparing training data for Neural Network Models
    // -------------------------------------------------------------------------

    fn build_training_data(
        &self,
        env: &dyn Environment,
        trajectory: &Trajectory, // TODO support multiple trajectories
    ) -> HashMap<ModelId, TrainingData<B>> {
        let mut all_advantages: HashMap<ModelId, Vec<f32>> = HashMap::new();
        let mut all_returns: HashMap<ModelId, Vec<f32>> = HashMap::new();
        let mut all_observations_flat: HashMap<ModelId, Vec<f32>> = HashMap::new();
        let mut all_discrete_actions_flat: HashMap<ModelId, Vec<u32>> = HashMap::new();
        let mut all_continuous_actions_flat: HashMap<ModelId, Vec<f32>> = HashMap::new();
        let mut all_log_probs: HashMap<ModelId, Vec<f32>> = HashMap::new();
        let mut all_action_spaces: HashMap<ModelId, ActionSpace> = HashMap::new();
        let mut model_id_to_action_space = HashMap::new();

        for (actor_id, points) in trajectory.paths.iter() {
            let model_id = env.model_id(*actor_id);
            model_id_to_action_space
                .entry(model_id)
                .or_insert(env.action_space(*actor_id));

            let mut gae = 0.0;
            let mut next_value = 0.0; // Bootstrap value (0.0 if done)

            let mut actor_advantages = vec![0.0; points.len()];
            let mut actor_returns = vec![0.0; points.len()];

            for t in (0..points.len()).rev() {
                // iterate backwards
                let point = &points[t];

                // If done=1.0, we mask out the next_value
                let mask = 1.0 - point.done;

                let delta =
                    point.reward + (self.config.gamma as f32 * next_value * mask) - point.value;
                gae =
                    delta + (self.config.gamma as f32 * self.config.gae_lambda as f32 * mask * gae);

                actor_advantages[t] = gae;
                actor_returns[t] = gae + point.value;

                next_value = point.value;
            }

            all_advantages
                .entry(model_id)
                .or_insert(vec![])
                .extend(actor_advantages);
            all_returns
                .entry(model_id)
                .or_insert(vec![])
                .extend(actor_returns);
            all_observations_flat
                .entry(model_id)
                .or_insert(vec![])
                .extend(points.iter().flat_map(|tr| tr.observation.data.clone()));
            all_continuous_actions_flat
                .entry(model_id)
                .or_insert(vec![])
                .extend(points.iter().flat_map(|tr| tr.action.continuous.clone()));
            all_discrete_actions_flat
                .entry(model_id)
                .or_insert(vec![])
                .extend(points.iter().flat_map(|tr| tr.action.discrete.clone()));
            all_log_probs
                .entry(model_id)
                .or_insert(vec![])
                .extend(points.iter().map(|tr| tr.log_prob));
            all_action_spaces.insert(model_id, env.action_space(*actor_id).clone());
        }

        // Convert arrays to tensors
        let mut model_training_data = HashMap::new();
        let model_ids = all_advantages.keys().cloned().collect::<HashSet<_>>();
        for model_id in model_ids {
            let size = all_advantages.get(&model_id).unwrap().len();

            let adv_tensor = Tensor::<B, 2>::from_floats(
                all_advantages.get(&model_id).unwrap().as_slice(),
                &self.device,
            )
            .reshape([size, 1]);

            let adv_mean = adv_tensor.clone().mean().reshape([1, 1]);
            let adv_std = adv_tensor.clone().var(0).sqrt().add_scalar(1e-8);
            let adv_normalized = (adv_tensor - adv_mean) / adv_std;

            let ret_tensor = Tensor::<B, 2>::from_floats(
                all_returns.get(&model_id).unwrap().as_slice(),
                &self.device,
            )
            .reshape([size, 1]);

            let obs_len = all_observations_flat[&model_id].len();
            let obs_dim = obs_len / size;
            let obs_tensor = Tensor::<B, 2>::from_floats(
                all_observations_flat.get(&model_id).unwrap().as_slice(),
                &self.device,
            )
            .reshape([size, obs_dim]);

            let continuous_act_len = all_continuous_actions_flat[&model_id].len();
            let continuous_act_dim = continuous_act_len / size;
            let continuous_actions_tensor = Tensor::<B, 2>::from_floats(
                all_continuous_actions_flat
                    .get(&model_id)
                    .unwrap()
                    .as_slice(),
                &self.device,
            )
            .reshape([size, continuous_act_dim]);

            let discrete_act_len = all_discrete_actions_flat[&model_id].len();
            let discrete_act_dim = discrete_act_len / size;
            let discrete_act_tensor = Tensor::<B, 2, Int>::from_ints(
                all_discrete_actions_flat.get(&model_id).unwrap().as_slice(),
                &self.device,
            )
            .reshape([size, discrete_act_dim]);

            // let discrete_act_tensor = self.discrete_actions_to_one_hot_tensor(
            //     discrete_act_tensor,
            //     all_action_spaces.get(&model_id).unwrap(),
            // );

            let log_probs_tensor = Tensor::<B, 2>::from_floats(
                all_log_probs.get(&model_id).unwrap().as_slice(),
                &self.device,
            );

            model_training_data.insert(
                model_id,
                TrainingData {
                    action_space: (**model_id_to_action_space.get(&model_id).unwrap()).clone(),
                    obs: obs_tensor,
                    discrete_actions: discrete_act_tensor,
                    continuous_actions: continuous_actions_tensor,
                    log_probs: log_probs_tensor,
                    advantages: adv_normalized,
                    returns: ret_tensor,
                },
            );
        }

        // TODO verify sizes of Tensors

        // (adv_normalized, ret_tensor)
        model_training_data
    }

    // -------------------------------------------------------------------------
    // Phase 3: Optimization Update
    // -------------------------------------------------------------------------

    fn update_policy(&mut self, training_data: HashMap<ModelId, TrainingData<B>>) {
        for (
            model_id,
            TrainingData {
                action_space,
                obs,
                discrete_actions,
                continuous_actions,
                log_probs,
                advantages,
                returns,
            },
        ) in training_data
        {
            for _ in 0..self.config.epochs_per_rollout {
                // 1. Calculate the loss (Forward Pass)
                let loss = self.compute_loss(
                    model_id,
                    action_space.clone(),
                    obs.clone(),
                    discrete_actions.clone(),
                    continuous_actions.clone(),
                    log_probs.clone(),
                    advantages.clone(),
                    returns.clone(),
                );

                // 2. Calculate Gradients (Backward Pass)
                // and returns the gradients for all parameters tracked in the graph.
                let grads = loss.backward();
                let grads_params = GradientsParams::from_grads(grads, &self.model_manager);

                // TODO propagate gradients
                // This works because Manager implements AutodiffModule<B>
                self.model_manager = self.optimizer.step(
                    self.config.learning_rate,
                    self.model_manager.clone(),
                    grads_params,
                );
            }
        }
    }

    fn compute_loss(
        &mut self,
        model_id: ModelId,
        action_space: ActionSpace, // Unused for discrete slicing now, kept for signature compatibility
        obs: Tensor<B, 2>,
        discrete_actions: Tensor<B, 2, Int>,
        continuous_actions: Tensor<B, 2>,
        old_log_probs: Tensor<B, 2>,
        advantages: Tensor<B, 2>,
        returns: Tensor<B, 2>,
    ) -> Tensor<B, 1> {
        // Check if we actually have continuous dimensions
        let [batch_size, _] = obs.dims();

        // 1. Get Model and Forward Pass
        let model = self.model_manager.get_model_by_id(model_id);
        let logits: PPOModelLogits<B> = model.forward(obs);

        // Initialize accumulators
        let mut new_log_probs = Tensor::zeros([batch_size, 1], &self.device);
        let mut total_entropy = Tensor::zeros([batch_size, 1], &self.device);

        // =========================================================
        // 2. Discrete Actions Handling (Iterate the Vec)
        // =========================================================
        Self::discrete_actions_loss_calc(
            action_space.discrete,
            discrete_actions,
            batch_size,
            &logits,
            &mut new_log_probs,
            &mut total_entropy,
        );

        // =========================================================
        // 3. Continuous Actions Handling
        // =========================================================
        Self::continuous_actions_loss_calc(
            continuous_actions,
            &logits,
            &mut new_log_probs,
            &mut total_entropy,
        );

        // =========================================================
        // 4. PPO Loss Calculation
        // =========================================================

        // Entropy Loss
        let entropy_loss = total_entropy
            .mean()
            .mul_scalar(self.config.entropy_coef)
            .neg();

        // Ratio
        let ratio = (new_log_probs - old_log_probs).exp();

        // Surrogate Loss
        let surr1 = ratio.clone() * advantages.clone();
        let eps = self.config.clip_ratio;
        let ratio_clipped = ratio.clamp(1.0 - eps, 1.0 + eps);
        let surr2 = ratio_clipped * advantages;
        let policy_loss = Tensor::min_pair(surr1, surr2).mean().neg();

        // Value Loss
        let value_loss = (logits.values - returns).powf_scalar(2.0).mean();

        // Sum
        policy_loss + value_loss.mul_scalar(self.config.value_coef) + entropy_loss
    }

    fn continuous_actions_loss_calc(
        continuous_actions: Tensor<B, 2>,
        logits: &PPOModelLogits<B>,
        new_log_probs: &mut Tensor<B, 2>,
        total_entropy: &mut Tensor<B, 2>,
    ) {
        // 1. Check if continuous dimensions exist
        if logits.continuous_mean.dims()[1] > 0 {
            let mean = logits.continuous_mean.clone();

            // We calculate std from log_std (neural networks usually output log_std for stability)
            let std = logits.continuous_log_std.clone().exp();
            let var = std.clone().powf_scalar(2.0);

            // A. Gaussian Log Prob (Likelihood)
            // Formula: -0.5 * ((x - mu)^2 / var) - log(std) - 0.5 * log(2pi)
            let diff = continuous_actions - mean;

            let log_std = logits.continuous_log_std.clone();

            let gauss_log_probs =
                diff.powf_scalar(2.0).neg().div(var.mul_scalar(2.0)) - log_std - 0.9189385; // 0.5 * ln(2 * pi)

            // We sum_dim(1) because probability of a vector is the product of elements (sum in log-space).
            *new_log_probs = new_log_probs.clone() + gauss_log_probs.sum_dim(1);

            // B. Gaussian Entropy
            // Formula: Sum(log(std) + 0.5 + 0.5 * log(2pi))
            let gauss_entropy = (logits.continuous_log_std.clone() + 0.5 + 0.9189385).sum_dim(1);

            // ERROR FIX: Dereference (*) to update accumulator
            *total_entropy = total_entropy.clone() + gauss_entropy;
        }
    }

    fn discrete_actions_loss_calc(
        discrete_action_space: Vec<usize>,
        discrete_actions: Tensor<B, 2, Int>, // Input is now explicitly Int
        batch_size: usize,
        logits: &PPOModelLogits<B>,
        new_log_probs: &mut Tensor<B, 2>,
        total_entropy: &mut Tensor<B, 2>,
    ) {
        // VALIDATION: Ensure model heads match the action space definition
        assert_eq!(
            logits.discrete.len(),
            discrete_action_space.len(),
            "Model discrete heads count does not match action space length"
        );

        // ZIP: Iterate through Logits and ActionSpace sizes together
        for (head_idx, (head_logits, &space_size)) in logits
            .discrete
            .iter()
            .zip(discrete_action_space.iter())
            .enumerate()
        {
            let [_batch, dim] = head_logits.dims();
            assert_eq!(dim, space_size, "Dimension mismatch in head {}", head_idx);

            let head_log_probs_all = log_softmax(head_logits.clone(), 1);
            let head_probs = softmax(head_logits.clone(), 1);
            let action_col_indices = discrete_actions
                .clone()
                .slice([0..batch_size, head_idx..(head_idx + 1)]);

            let selected_log_prob = head_log_probs_all.clone().gather(1, action_col_indices);

            *new_log_probs = new_log_probs.clone() + selected_log_prob;

            let head_entropy = (head_probs * head_log_probs_all).sum_dim(1).neg();
            *total_entropy = total_entropy.clone() + head_entropy;
        }
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

    fn discrete_actions_to_one_hot_tensor(
        &self,
        // Input: [Batch_Size, Num_Heads] (e.g., indices of selected actions)
        tensor: Tensor<B, 2>,
        // From ActionSpace.discrete: sizes of each head (e.g. [5, 2])
        action_spaces: &ActionSpace,
    ) -> Tensor<B, 2> {
        // Output: [Batch_Size, Sum_Of_Limits]
        let [batch_size, num_heads] = tensor.dims();
        if num_heads != action_spaces.discrete.len() {
            panic!(
                "Discrete Action dimension is not correct. Tensor heads size should match to discrete action space size. Tensor heads: {}, Discrete Action Space: {}",
                num_heads,
                action_spaces.discrete.len()
            );
        }

        let mut one_hot_parts = Vec::new();

        for (head_idx, &class_count) in action_spaces.discrete.iter().enumerate() {
            // Shape: [Batch_Size, 1]
            let head_indices = tensor
                .clone()
                .slice([0..batch_size, head_idx..head_idx + 1]);

            // [Batch_Size, 1, Class_Count]
            let one_hot: Tensor<B, 3> = Tensor::<B, 2>::one_hot(head_indices, class_count);

            // Squeeze the middle dimension to get [Batch_Size, Class_Count]
            let one_hot_flat = one_hot.squeeze::<2>();
            one_hot_parts.push(one_hot_flat);
        }

        // Concatenate all heads along dimension 1
        // Result Shape: [Batch_Size, Sum(class_count)]
        Tensor::cat(one_hot_parts, 1)
    }
}
