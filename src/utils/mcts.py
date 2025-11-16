"""
Monte Carlo Tree Search for AlphaZero-style self-play training
"""

import numpy as np
import chess
import torch
from typing import Dict, Tuple, Optional
import math


class MCTSNode:
    """Node in the MCTS search tree"""

    def __init__(self, board: chess.Board, parent: Optional['MCTSNode'] = None,
                 move: Optional[chess.Move] = None, prior: float = 0.0):
        self.board = board.copy()
        self.parent = parent
        self.move = move  # Move that led to this node
        self.prior = prior  # Prior probability from policy network

        self.children: Dict[chess.Move, MCTSNode] = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.is_expanded = False

    def value(self) -> float:
        """Average value of this node"""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def is_leaf(self) -> bool:
        """Check if this is a leaf node"""
        return len(self.children) == 0

    def select_child(self, c_puct: float = 1.0) -> 'MCTSNode':
        """Select child with highest UCB score"""
        best_score = -float('inf')
        best_child = None

        for child in self.children.values():
            # UCB formula: Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
            q_value = child.value()
            u_value = c_puct * child.prior * math.sqrt(self.visit_count) / (1 + child.visit_count)
            score = q_value + u_value

            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def expand(self, policy_probs: Dict[chess.Move, float]):
        """Expand node with legal moves and their policy probabilities"""
        for move, prob in policy_probs.items():
            if move not in self.children:
                next_board = self.board.copy()
                next_board.push(move)
                self.children[move] = MCTSNode(next_board, parent=self, move=move, prior=prob)

        self.is_expanded = True

    def backup(self, value: float):
        """Backpropagate value up the tree"""
        node = self
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            value = -value  # Flip value for opponent
            node = node.parent


class MCTS:
    """Monte Carlo Tree Search for chess using neural network guidance"""

    def __init__(self, model, num_simulations: int = 100, c_puct: float = 1.0, temperature: float = 1.0):
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.temperature = temperature

    def search(self, board: chess.Board) -> Dict[chess.Move, float]:
        """
        Run MCTS from the given position.
        Returns move probabilities improved by search.
        """
        root = MCTSNode(board)

        # Run simulations
        for _ in range(self.num_simulations):
            node = root
            search_path = [node]

            # Selection: traverse tree until leaf
            while node.is_expanded and not node.board.is_game_over():
                node = node.select_child(self.c_puct)
                search_path.append(node)

            # Get value from network
            value = 0.0
            if node.board.is_game_over():
                # Terminal node
                result = node.board.result()
                if result == "1-0":
                    value = 1.0 if node.board.turn == chess.WHITE else -1.0
                elif result == "0-1":
                    value = -1.0 if node.board.turn == chess.WHITE else 1.0
                else:
                    value = 0.0
            else:
                # Expand and evaluate with network
                policy_probs, value = self._evaluate(node.board)
                node.expand(policy_probs)

            # Backpropagation
            for node_in_path in reversed(search_path):
                node_in_path.backup(value)
                value = -value

        # Return move probabilities based on visit counts
        return self._get_action_probs(root)

    def _evaluate(self, board: chess.Board) -> Tuple[Dict[chess.Move, float], float]:
        """
        Evaluate position with neural network.
        Returns (policy_probs_dict, value)
        """
        from .chess_bot import board_to_tensor, move_to_index

        # Get network output
        with torch.no_grad():
            board_tensor = board_to_tensor(board)
            policy_logits, value = self.model(board_tensor)

            # Convert to probabilities
            policy_probs = torch.softmax(policy_logits, dim=1).squeeze(0).numpy()
            value = value.item()

        # Map legal moves to their probabilities
        legal_moves = list(board.legal_moves)
        move_probs = {}

        total_prob = 0.0
        for move in legal_moves:
            idx = move_to_index(move)
            if idx < len(policy_probs):
                move_probs[move] = policy_probs[idx]
                total_prob += policy_probs[idx]

        # Normalize probabilities
        if total_prob > 0:
            move_probs = {m: p / total_prob for m, p in move_probs.items()}
        else:
            # Uniform if network gives no signal
            uniform_prob = 1.0 / len(legal_moves)
            move_probs = {m: uniform_prob for m in legal_moves}

        return move_probs, value

    def _get_action_probs(self, root: MCTSNode) -> Dict[chess.Move, float]:
        """
        Get action probabilities from visit counts.
        Uses temperature to control exploration.
        """
        if len(root.children) == 0:
            return {}

        moves = []
        visit_counts = []

        for move, child in root.children.items():
            moves.append(move)
            visit_counts.append(child.visit_count)

        visit_counts = np.array(visit_counts, dtype=np.float32)

        # Apply temperature
        if self.temperature == 0:
            # Greedy: pick move with most visits
            probs = np.zeros_like(visit_counts)
            probs[np.argmax(visit_counts)] = 1.0
        else:
            # Temperature scaling
            visit_counts = visit_counts ** (1.0 / self.temperature)
            probs = visit_counts / visit_counts.sum()

        # Create move -> probability mapping
        action_probs = {move: prob for move, prob in zip(moves, probs)}

        return action_probs

    def get_best_move(self, board: chess.Board) -> chess.Move:
        """Get the best move according to MCTS"""
        action_probs = self.search(board)
        if not action_probs:
            # Fallback to random legal move
            return np.random.choice(list(board.legal_moves))

        # Pick move with highest probability
        best_move = max(action_probs.items(), key=lambda x: x[1])[0]
        return best_move
