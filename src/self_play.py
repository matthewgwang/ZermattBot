"""
Self-play game generation for AlphaZero training
"""

import chess
import numpy as np
import torch
from typing import List, Tuple, Dict
from .utils.mcts import MCTS
from .utils.chess_bot import ChessNet1500, board_to_tensor, move_to_index


class SelfPlayGame:
    """Generate self-play games using MCTS"""

    def __init__(self, model: ChessNet1500, num_simulations: int = 100, temperature: float = 1.0):
        self.model = model
        self.mcts = MCTS(model, num_simulations=num_simulations, temperature=temperature)

    def play_game(self, max_moves: int = 200) -> Tuple[List[Tuple[np.ndarray, Dict[chess.Move, float], float]], str]:
        """
        Play a single self-play game.

        Returns:
            training_data: List of (board_state, mcts_probs, outcome) tuples
            result: Game result string ("1-0", "0-1", "1/2-1/2")
        """
        board = chess.Board()
        training_data = []

        move_count = 0
        temperature = self.mcts.temperature

        while not board.is_game_over() and move_count < max_moves:
            # Use high temperature for first 30 moves (exploration)
            # Then reduce to near-zero for exploitation
            if move_count < 30:
                self.mcts.temperature = 1.0
            else:
                self.mcts.temperature = 0.1

            # Get MCTS-improved probabilities
            action_probs = self.mcts.search(board)

            if not action_probs:
                break

            # Store training example (we'll set outcome later)
            board_state = self._board_to_planes(board)
            training_data.append((board_state, action_probs.copy(), None))

            # Sample move from MCTS probabilities
            moves = list(action_probs.keys())
            probs = list(action_probs.values())
            move = np.random.choice(moves, p=probs)

            # Make move
            board.push(move)
            move_count += 1

        # Restore temperature
        self.mcts.temperature = temperature

        # Get game result
        result = board.result()

        # Assign outcomes from each player's perspective
        if result == "1-0":
            winner_value = 1.0
        elif result == "0-1":
            winner_value = -1.0
        else:
            winner_value = 0.0

        # Update training data with outcomes
        final_training_data = []
        for i, (state, probs, _) in enumerate(training_data):
            # Flip value for each turn (white vs black)
            move_color = i % 2  # 0 = white, 1 = black
            if move_color == 0:  # White's perspective
                outcome = winner_value
            else:  # Black's perspective
                outcome = -winner_value

            final_training_data.append((state, probs, outcome))

        return final_training_data, result

    def _board_to_planes(self, board: chess.Board) -> np.ndarray:
        """Convert board to 15-plane representation"""
        planes = np.zeros((15, 8, 8), dtype=np.float32)

        piece_idx = {
            chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
            chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
        }

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                rank, file = divmod(square, 8)
                idx = piece_idx[piece.piece_type]
                if piece.color == chess.BLACK:
                    idx += 6
                planes[idx, rank, file] = 1.0

        # Castling rights
        if board.has_kingside_castling_rights(chess.WHITE):
            planes[12, 0, 4:8] = 1.0
        if board.has_queenside_castling_rights(chess.WHITE):
            planes[12, 0, 0:4] = 1.0
        if board.has_kingside_castling_rights(chess.BLACK):
            planes[12, 7, 4:8] = 1.0
        if board.has_queenside_castling_rights(chess.BLACK):
            planes[12, 7, 0:4] = 1.0

        # En passant
        if board.ep_square is not None:
            rank, file = divmod(board.ep_square, 8)
            planes[13, rank, file] = 1.0

        # Turn
        if board.turn == chess.WHITE:
            planes[14, :, :] = 1.0

        return planes


def generate_self_play_games(model: ChessNet1500, num_games: int = 100,
                              num_simulations: int = 100) -> List[Tuple[np.ndarray, Dict[chess.Move, float], float]]:
    """
    Generate multiple self-play games.

    Args:
        model: Chess neural network
        num_games: Number of games to generate
        num_simulations: MCTS simulations per move

    Returns:
        List of training examples (board_state, mcts_probs, outcome)
    """
    generator = SelfPlayGame(model, num_simulations=num_simulations)
    all_training_data = []

    for game_num in range(num_games):
        print(f"Playing self-play game {game_num + 1}/{num_games}...")

        training_data, result = generator.play_game()
        all_training_data.extend(training_data)

        print(f"  Game {game_num + 1}: {result}, {len(training_data)} positions")

    print(f"\nGenerated {len(all_training_data)} training positions from {num_games} games")
    return all_training_data
