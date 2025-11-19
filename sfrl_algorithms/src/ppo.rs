use crate::definitions::{Agent, Algorithm, LogProb, Model, ValueLogit};
use sfrl_core::{Actions, Env, HasActorId, Observation, Observations, StepResult, StepResults};

#[derive(Clone, Debug)]
struct TrajectoryPoint {
    observation: Observation,
    actions: Actions, // one actor can have multiple actions
    result: StepResult,
    value: ValueLogit,
    log_prob: LogProb,
}

struct Trajectory {
    paths: Vec<Vec<TrajectoryPoint>>,
}

impl Trajectory {
    fn new(max_num_of_actors: usize) -> Self {
        let mut paths = Vec::with_capacity(max_num_of_actors);
        for _ in 0..max_num_of_actors {
            paths.push(Vec::new())
        }
        Self { paths }
    }

    fn add_point(&mut self, observations: Observations, actions: Actions, results: StepResults) {
        let mut acts = vec![Actions::default(); self.paths.len()];
        let mut obs = vec![Observation::default(); self.paths.len()];
        let mut ress = vec![StepResult::default(); self.paths.len()];
        for Observation { actor_id, data } in observations {
            // TODO make code safer
            obs[actor_id].actor_id = actor_id;
            obs[actor_id].data = data;
        }
        for StepResult {
            actor_id,
            reward,
            done,
        } in results.results
        {
            // TODO make code safer
            ress[actor_id].actor_id = actor_id;
            ress[actor_id].reward = reward;
            ress[actor_id].done = done;
        }
        for action in actions {
            let index = action.actor_id();
            acts[index].push(action);
        }
        for i in 0..self.paths.len() {
            let (actions, observation, result) =
                (acts.pop().unwrap(), obs.pop().unwrap(), ress.pop().unwrap());
            if actions.is_empty() {
                continue;
            }
            self.paths[i].push(TrajectoryPoint {
                observation,
                actions,
                result,
                value: 0.0,    // TODO pass this parameter for training
                log_prob: 0.0, // TODO pass this parameter for training
            });
        }
    }
}

pub struct Ppo {
    pub epochs: usize,
}

impl Ppo {
    pub fn new(epochs: usize) -> Self {
        Self { epochs }
    }
}

pub struct PPOAgent {}

impl Agent for PPOAgent {
    fn sample(
        &self,
        env: &mut dyn Env,
        model: &mut dyn Model,
        observations: Observations,
    ) -> Actions {
        let mut actions = Actions::default();
        for observation in &observations {
            let (action_logits, value_logits) = model.infer(observation);
            // TODO mask actions logits if actions are not available
            actions.append(
                &mut env.sample_from_model_output(observation.actor_id, action_logits.as_slice()),
            );
        }
        actions
    }

    fn step(&self, env: &mut dyn Env, actions: &Actions) -> StepResults {
        env.step(actions)
    }
}

impl Algorithm for Ppo {
    fn train<E: Env>(
        &mut self,
        env: &mut E,
        model: &mut dyn Model,
        steps: usize,
        agent: &mut dyn Agent,
    ) {
        env.reset();
        for t in 0..steps {
            let mut trajectory = Trajectory::new(env.number_of_actors());
            let observations = env.observations();
            let mut actions = Actions::default();
            for observation in observations {
                let (action_logits, value_logit) = model.infer(&observation);
                actions.append(
                    &mut env.sample_from_model_output(observation.actor_id, &action_logits),
                );
            }
            let res = agent.step(env, &actions);

            // TODO implement logic if trajectory limit achieved
            if res.is_done() {
                // TODO convert trajectory to experiences

                env.reset();
            }
        }
        // TODO: real PPO: collect rollouts, compute advantages, update policy/value networks.
    }
}
