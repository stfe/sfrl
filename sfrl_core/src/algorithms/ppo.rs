use super::definitions::{Agent, Algorithm, LogProb, Model, ValueLogit};
use crate::utils::calculate_log_probs;
use crate::{Actions, ActorId, Env, HasActorId, Observation, Observations, StepResult, StepResults};

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

    fn add_point(
        &mut self,
        observations: Observations,
        actions: Actions,
        log_probs: Vec<(ActorId, LogProb)>,
        values: Vec<(ActorId, ValueLogit)>,
        results: StepResults,
    ) {
        let mut acts = vec![Actions::default(); self.paths.len()];
        let mut lp = vec![LogProb::default(); self.paths.len()];
        let mut vs = vec![ValueLogit::default(); self.paths.len()];
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
            let actor_id = action.actor_id();
            acts[actor_id].push(action);
        }

        log_probs
            .into_iter()
            .for_each(|(actor_id, log_prob)| lp[actor_id] = log_prob);

        values
            .into_iter()
            .for_each(|(actor_id, value)| vs[actor_id] = value);

        for i in 0..self.paths.len() {
            let (actions, observation, result, value, log_prob) = (
                acts.pop().unwrap(),
                obs.pop().unwrap(),
                ress.pop().unwrap(),
                vs.pop().unwrap(),
                lp.pop().unwrap(),
            );
            if actions.is_empty() {
                continue;
            }
            self.paths[i].push(TrajectoryPoint {
                observation,
                actions,
                result,
                value,
                log_prob,
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
        for _t in 0..steps {
            let mut trajectory = Trajectory::new(env.number_of_actors());
            let observations = env.observations();
            let mut actions = Actions::default();
            let mut values: Vec<(ActorId, ValueLogit)> = Vec::with_capacity(observations.len());
            let mut log_probs: Vec<(ActorId, ValueLogit)> = Vec::with_capacity(observations.len());

            for observation in &observations {
                let (action_logits, value_logit) = model.infer(&observation);
                values.push((observation.actor_id, value_logit));
                let mut single_actor_actions =
                    env.sample_from_model_output(observation.actor_id, &action_logits);
                log_probs.push((
                    observation.actor_id,
                    calculate_log_probs(&action_logits, &actions),
                ));
                actions.append(&mut single_actor_actions);
            }
            let res = agent.step(env, &actions);
            let done = res.is_done();
            trajectory.add_point(observations, actions, log_probs, values, res);

            // TODO implement logic if trajectory limit achieved
            if done {
                // TODO convert trajectory to experiences

                env.reset();
            }
        }
        // TODO: real PPO: collect rollouts, compute advantages, update policy/value networks.
    }
}
