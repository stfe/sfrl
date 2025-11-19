//! A minimal Tic-Tac-Toe engine.
//! Mechanics-only: board representation and win/draw checks.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mark {
    Empty,
    X,
    O,
}

#[derive(Debug, Clone)]
pub struct TicTacToe {
    board: [Mark; 9],
    moves_made: u8,
}

impl TicTacToe {
    pub fn new() -> Self {
        Self {
            board: [Mark::Empty; 9],
            moves_made: 0,
        }
    }

    pub fn reset(&mut self) {
        self.board = [Mark::Empty; 9];
        self.moves_made = 0;
    }

    /// Attempts to place `mark` at `idx` (0..9). Returns `true` if successful.
    pub fn place(&mut self, idx: usize) -> bool {
        let mark = self.current_player();
        if idx >= 9 {
            return false;
        }
        if !matches!(mark, Mark::X | Mark::O) {
            return false;
        }
        if self.board[idx] != Mark::Empty {
            return false;
        }
        self.board[idx] = mark;
        self.moves_made = self.moves_made.saturating_add(1);
        true
    }

    pub fn is_empty(&self, index: usize) -> bool {
        self.board[index] == Mark::Empty
    }

    /// Returns Some(winner) if someone has won, None otherwise.
    pub fn winner(&self) -> Option<Mark> {
        const LINES: [[usize; 3]; 8] = [
            // rows
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
            // cols
            [0, 3, 6],
            [1, 4, 7],
            [2, 5, 8],
            // diags
            [0, 4, 8],
            [2, 4, 6],
        ];
        for line in &LINES {
            let [a, b, c] = *line;
            let m = self.board[a];
            if m != Mark::Empty && m == self.board[b] && m == self.board[c] {
                return Some(m);
            }
        }
        None
    }

    pub fn is_game_over(&self) -> bool {
        self.moves_made >= 9 || self.winner().is_some()
    }

    pub fn board(&self) -> &[Mark; 9] {
        &self.board
    }

    pub fn current_player(&self) -> Mark {
        if self.moves_made % 2 == 0 {
            Mark::X
        } else {
            Mark::O
        }
    }
}
