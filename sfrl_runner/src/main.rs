use rand::Rng;
use sfrl_algorithms::{ppo::Ppo, Algorithm, StdoutLogger};
use sfrl_core::{ActionSpace, Observation};
use sfrl_envs::gridworld_env::{GridWorldEnv, GridWorldMove};

fn main() {
    // Runner: wire env + algorithm, provide a policy, run training.
    // This is intentionally lightweight and for demonstration only.

    // Create environment (adapts the engine to core::Env)
    let mut env = GridWorldEnv::new(5, 5);

    // Choose algorithm (PPO stub)
    let mut algo = Ppo::new();

    // Logger
    let mut logger = StdoutLogger;

    // Define a dummy policy: pick a random action from the discrete space.
    let mut rng = rand::thread_rng();
    let mut policy = move |_obs: &Observation, action_space: &ActionSpace| -> GridWorldMove {
        match action_space {
            ActionSpace::Discrete { n } => {
                let i = rng.gen_range(0..*n);
                match i {
                    0 => GridWorldMove::Up,
                    1 => GridWorldMove::Down,
                    2 => GridWorldMove::Left,
                    3 => GridWorldMove::Right,
                    _ => GridWorldMove::Stay,
                }
            }
            ActionSpace::Continuous { .. } => {
                // Not supported in this demo env; default to Stay.
                GridWorldMove::Stay
            }
        }
    };

    // Run a short training loop
    algo.train(&mut env, 50, &mut policy, &mut logger);

    // TODOs for runner:
    // - Parse CLI args for algorithm selection and config.
    // - Load/save experiment configs and checkpoints.
}
