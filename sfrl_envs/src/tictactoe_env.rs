use rand::rngs::ThreadRng;
use rand::Rng;
use rand_distr::StandardNormal;
use sfrl_core::utils::softmax;
use sfrl_core::{Action, Actions, ActorId, Env, Observation, StepResult, StepResults};
use sfrl_engines::tictactoe::{Mark, TicTacToe};

struct TicTacToeEnv {
    rng: ThreadRng,
    game: TicTacToe,
}

impl TicTacToeEnv {
    fn new(game: TicTacToe) -> TicTacToeEnv {
        TicTacToeEnv {
            rng: rand::rng(),
            game,
        }
    }

    fn game_to_observations(&self) -> Vec<Observation> {
        let mut data = vec![0f32; 18];
        let actor_mark = self.game.current_player();
        for i in 0..self.game.board().len() {
            let value = self.game.board()[i];
            if value == actor_mark {
                data[i] = 1.0;
            } else {
                data[i + 9] = 1.0;
            }
        }
        vec![Observation {
            actor_id: self.actor_id(),
            data,
        }]
    }

    fn actor_id(&self) -> ActorId {
        if self.game.current_player() == Mark::X {
            0
        } else {
            1
        }
    }
}

impl Env for TicTacToeEnv {
    fn reset(&mut self) -> Vec<Observation> {
        self.game.reset();
        self.game_to_observations()
    }

    fn observations(&self) -> Vec<Observation> {
        self.game_to_observations()
    }

    fn sample_from_model_output(&mut self, actor_id: ActorId, logits: &[f32]) -> Actions {
        let rng_value: f32 = self.rng.sample(StandardNormal);
        let probs = softmax(logits);
        let mut acc_prob = 0.0;
        for (index, &prob) in probs.iter().enumerate() {
            acc_prob += prob;
            if rng_value < acc_prob && self.game.is_empty(index) {
                return vec![Action::Discrete {
                    actor_id,
                    value: index as u32,
                }];
            }
        }
        vec![Action::Discrete {
            actor_id,
            value: logits.len() as u32 - 1,
        }]
    }

    fn step(&mut self, actions: &Actions) -> StepResults {
        let current_player = self.game.current_player();
        let actor_id = self.actor_id();
        match actions[0] {
            Action::Discrete {
                actor_id: _,
                value: index,
            } => {
                if !self.game.place(index as usize) {
                    panic!("Incorrect action position");
                }
            }
            _ => {
                panic!("Incorrect action type")
            }
        }
        let done = self.game.is_game_over();
        let winner = if let Some(mark) = self.game.winner() {
            current_player == mark
        } else {
            false
        };

        StepResults {
            results: vec![StepResult {
                actor_id,
                reward: if winner { 1.0 } else { 0.0 },
                done,
            }],
        }
    }

    fn number_of_actors(&self) -> usize {
        2
    }
}
