// use crate::Algorithm;
// use sfrl_core::{ActionSpace, Env, Logger, Observation};

/// Soft Actor-Critic (stub).
pub struct Sac {
    pub temperature: f32,
    // TODO: add learning rates, target update rates, etc.
}

impl Sac {
    pub fn new() -> Self { Self { temperature: 0.1 } }
}

// impl Algorithm for Sac {
//     fn train<E: Env>(
//         &mut self,
//         env: &mut E,
//         steps: usize,
//         policy: &mut dyn FnMut(&Observation, &ActionSpace) -> E::Action,
//         logger: &mut dyn Logger,
//     ) {
//         let mut obs = env.reset();
//         let action_space = env.action_space();
//         for t in 0..steps {
//             let action = policy(&obs, &action_space);
//             let res = env.step(&action);
//             logger.log(&format!(
//                 "[SAC] t={t}, r={:.3}, done={}, obs_len={}",
//                 res.reward,
//                 res.done,
//                 res.observation.len()
//             ));
//             if res.done {
//                 obs = env.reset();
//             } else {
//                 obs = res.observation;
//             }
//         }
//         // TODO: real SAC: entropy-regularized actor/critic updates with replay buffer.
//     }
// }
