"""
AlphaZero-style self-play training on Modal GPU
"""

import modal

app = modal.App("chess-self-play-trainer")

# Create persistent volume for models
volume = modal.Volume.from_name("chess-selfplay-data", create_if_missing=True)

# Docker image with dependencies
image = (
    modal.Image.debian_slim()
    .pip_install("torch", "numpy", "python-chess")
)


@app.function(
    image=image,
    gpu="T4",
    volumes={"/data": volume},
    timeout=7200,  # 2 hours
)
def self_play_iteration(iteration: int, games_per_iteration: int = 100, num_simulations: int = 100):
    """
    Run one iteration of self-play training:
    1. Load current model
    2. Generate self-play games
    3. Train on self-play data
    4. Save improved model
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    import chess
    import numpy as np
    import os

    print(f"\n{'='*60}")
    print(f"SELF-PLAY ITERATION {iteration}")
    print(f"{'='*60}\n")

    # Check GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Warning: No GPU found, using CPU")

    # ===== MODEL DEFINITION =====
    class ChessNet1500(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(15, 64, 3, padding=1)
            self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
            self.conv3 = nn.Conv2d(128, 128, 3, padding=1)

            self.policy = nn.Sequential(
                nn.Conv2d(128, 64, 1),
                nn.Flatten(),
                nn.Linear(64 * 8 * 8, 1024),
                nn.ReLU(),
                nn.Linear(1024, 4096)
            )

            self.value = nn.Sequential(
                nn.Conv2d(128, 32, 1),
                nn.Flatten(),
                nn.Linear(32 * 8 * 8, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
                nn.Tanh()
            )

        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = torch.relu(self.conv3(x))
            return self.policy(x), self.value(x)

    # ===== HELPER FUNCTIONS =====
    def board_to_tensor(board):
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

        if board.has_kingside_castling_rights(chess.WHITE):
            planes[12, 0, 4:8] = 1.0
        if board.has_queenside_castling_rights(chess.WHITE):
            planes[12, 0, 0:4] = 1.0
        if board.has_kingside_castling_rights(chess.BLACK):
            planes[12, 7, 4:8] = 1.0
        if board.has_queenside_castling_rights(chess.BLACK):
            planes[12, 7, 0:4] = 1.0

        if board.ep_square is not None:
            rank, file = divmod(board.ep_square, 8)
            planes[13, rank, file] = 1.0

        if board.turn == chess.WHITE:
            planes[14, :, :] = 1.0

        return torch.FloatTensor(planes).unsqueeze(0)

    def move_to_index(move):
        return move.from_square * 64 + move.to_square

    # ===== MCTS (simplified inline version) =====
    class MCTSNode:
        def __init__(self, board, parent=None, move=None, prior=0.0):
            self.board = board.copy()
            self.parent = parent
            self.move = move
            self.prior = prior
            self.children = {}
            self.visit_count = 0
            self.value_sum = 0.0
            self.is_expanded = False

        def value(self):
            return self.value_sum / self.visit_count if self.visit_count > 0 else 0.0

        def select_child(self, c_puct=1.0):
            import math
            best_score = -float('inf')
            best_child = None
            for child in self.children.values():
                q_value = child.value()
                u_value = c_puct * child.prior * math.sqrt(self.visit_count) / (1 + child.visit_count)
                score = q_value + u_value
                if score > best_score:
                    best_score = score
                    best_child = child
            return best_child

        def expand(self, policy_probs):
            for move, prob in policy_probs.items():
                if move not in self.children:
                    next_board = self.board.copy()
                    next_board.push(move)
                    self.children[move] = MCTSNode(next_board, parent=self, move=move, prior=prob)
            self.is_expanded = True

        def backup(self, value):
            node = self
            while node is not None:
                node.visit_count += 1
                node.value_sum += value
                value = -value
                node = node.parent

    def mcts_search(model, board, num_simulations=100):
        """Simplified MCTS search"""
        root = MCTSNode(board)

        for _ in range(num_simulations):
            node = root
            search_path = [node]

            # Selection
            while node.is_expanded and not node.board.is_game_over():
                node = node.select_child()
                search_path.append(node)

            # Evaluation
            value = 0.0
            if node.board.is_game_over():
                result = node.board.result()
                if result == "1-0":
                    value = 1.0 if node.board.turn == chess.WHITE else -1.0
                elif result == "0-1":
                    value = -1.0 if node.board.turn == chess.WHITE else 1.0
            else:
                # Evaluate with network
                with torch.no_grad():
                    board_tensor = board_to_tensor(node.board).to(device)
                    policy_logits, value_pred = model(board_tensor)
                    policy_probs = torch.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()
                    value = value_pred.item()

                # Map to legal moves
                legal_moves = list(node.board.legal_moves)
                move_probs = {}
                total_prob = 0.0
                for move in legal_moves:
                    idx = move_to_index(move)
                    if idx < len(policy_probs):
                        move_probs[move] = policy_probs[idx]
                        total_prob += policy_probs[idx]

                if total_prob > 0:
                    move_probs = {m: p / total_prob for m, p in move_probs.items()}
                else:
                    uniform = 1.0 / len(legal_moves)
                    move_probs = {m: uniform for m in legal_moves}

                node.expand(move_probs)

            # Backpropagation
            for n in reversed(search_path):
                n.backup(value)
                value = -value

        # Get action probs
        moves = []
        visits = []
        for move, child in root.children.items():
            moves.append(move)
            visits.append(child.visit_count)

        visits = np.array(visits, dtype=np.float32)
        probs = visits / visits.sum()

        return {move: prob for move, prob in zip(moves, probs)}

    def play_self_play_game(model):
        """Play one self-play game"""
        board = chess.Board()
        training_data = []
        move_count = 0

        while not board.is_game_over() and move_count < 100:
            # Get MCTS probabilities
            action_probs = mcts_search(model, board, num_simulations=num_simulations)

            if not action_probs:
                break

            # Store state
            state = board_to_tensor(board).squeeze(0).numpy()
            training_data.append((state, action_probs.copy(), None))

            # Sample move
            moves = list(action_probs.keys())
            probs = np.array(list(action_probs.values()), dtype=np.float64)

            # Normalize to ensure probabilities sum to exactly 1.0
            probs = probs / probs.sum()

            move = np.random.choice(moves, p=probs)
            board.push(move)
            move_count += 1

        # Assign outcomes
        result = board.result()
        if result == "1-0":
            winner_value = 1.0
        elif result == "0-1":
            winner_value = -1.0
        else:
            winner_value = 0.0

        final_data = []
        for i, (state, probs, _) in enumerate(training_data):
            outcome = winner_value if i % 2 == 0 else -winner_value
            final_data.append((state, probs, outcome))

        return final_data, result

    # ===== LOAD MODEL =====
    model = ChessNet1500().to(device)

    model_path = f'/data/chess_model_iter{iteration-1}.pth' if iteration > 0 else '/data/chess_model.pth'
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    else:
        print(f"No existing model found, starting from scratch")

    model.eval()

    # ===== GENERATE SELF-PLAY GAMES =====
    print(f"\nGenerating {games_per_iteration} self-play games...")
    all_training_data = []

    for game_num in range(games_per_iteration):
        print(f"  Game {game_num + 1}/{games_per_iteration}...", end=" ")
        training_data, result = play_self_play_game(model)
        all_training_data.extend(training_data)
        print(f"{result} ({len(training_data)} positions)")

    print(f"\nTotal training positions: {len(all_training_data)}")

    # ===== PREPARE TRAINING DATA =====
    class SelfPlayDataset(Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            state, move_probs, outcome = self.data[idx]

            # Convert move probs to policy target
            policy_target = np.zeros(4096, dtype=np.float32)
            for move, prob in move_probs.items():
                policy_target[move_to_index(move)] = prob

            return (
                torch.FloatTensor(state),
                torch.FloatTensor(policy_target),
                torch.FloatTensor([outcome])
            )

    dataset = SelfPlayDataset(all_training_data)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    # ===== TRAIN MODEL =====
    print(f"\nTraining on self-play data...")
    model.train()

    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 5
    for epoch in range(epochs):
        total_p, total_v, num = 0, 0, 0

        for boards, policy_targets, value_targets in dataloader:
            boards = boards.to(device)
            policy_targets = policy_targets.to(device)
            value_targets = value_targets.to(device)

            optimizer.zero_grad()
            policy_pred, value_pred = model(boards)

            # Policy loss (cross-entropy)
            p_loss = -(policy_targets * torch.log_softmax(policy_pred, dim=1)).sum(dim=1).mean()
            v_loss = value_criterion(value_pred, value_targets)
            loss = p_loss + v_loss

            loss.backward()
            optimizer.step()

            total_p += p_loss.item()
            total_v += v_loss.item()
            num += 1

        avg_p = total_p / num
        avg_v = total_v / num
        print(f"Epoch {epoch+1}/{epochs}: Policy Loss={avg_p:.4f}, Value Loss={avg_v:.4f}")

    # ===== SAVE MODEL =====
    model_cpu = model.cpu()
    save_path = f'/data/chess_model_iter{iteration}.pth'
    torch.save(model_cpu.state_dict(), save_path)

    # Also save as base model for next run to continue from
    base_model_path = '/data/chess_model.pth'
    torch.save(model_cpu.state_dict(), base_model_path)
    volume.commit()

    print(f"\nModel saved to {save_path}")
    print(f"Iteration {iteration} complete!")

    # Return model as bytes
    with open(save_path, 'rb') as f:
        return f.read()


@app.local_entrypoint()
def main():
    """Run multiple iterations of self-play training"""
    import os

    iterations_per_run = 5  # Do 5 iterations each run
    games_per_iteration = 20  # Fast iterations
    num_simulations = 50  # Balanced quality/speed

    # Find the last completed iteration locally
    start_iteration = 0
    for i in range(1000):  # Check up to 1000 iterations
        if not os.path.exists(f'chess_model_iter{i}.pth'):
            start_iteration = i
            break

    num_iterations = start_iteration + iterations_per_run

    if start_iteration > 0:
        print(f"Found existing training up to iteration {start_iteration - 1}")
        print(f"Continuing with iterations {start_iteration} to {num_iterations - 1}")

    print(f"\n{'='*60}")
    print(f"ALPHAZERO SELF-PLAY TRAINING")
    print(f"{'='*60}")
    print(f"Running iterations: {start_iteration} to {num_iterations-1} ({iterations_per_run} iterations)")
    print(f"Games per iteration: {games_per_iteration}")
    print(f"MCTS simulations: {num_simulations}")
    print(f"Estimated time: ~{iterations_per_run * 20}-{iterations_per_run * 30} minutes")
    print(f"{'='*60}\n")

    for iteration in range(start_iteration, num_iterations):
        print(f"\n{'='*60}")
        print(f"STARTING ITERATION {iteration}")
        print(f"{'='*60}\n")

        model_bytes = self_play_iteration.remote(
            iteration,
            games_per_iteration=games_per_iteration,
            num_simulations=num_simulations
        )

        # Save latest model locally
        with open(f'chess_model_iter{iteration}.pth', 'wb') as f:
            f.write(model_bytes)

        print(f"\nIteration {iteration} complete! Model saved locally.")

    # Deploy final model to bot automatically
    final_iter = num_iterations - 1
    print(f"\n{'='*60}")
    print(f"TRAINING BATCH COMPLETE!")
    print(f"{'='*60}")
    print(f"Completed iterations: {start_iteration} to {final_iter}")
    print(f"Total iterations trained: {final_iter + 1}")

    # Auto-deploy to bot
    import shutil
    shutil.copy(f'chess_model_iter{final_iter}.pth', 'src/chess_model.pth')
    print(f"✓ Deployed to src/chess_model.pth")
    print(f"\nTo continue training: Run 'modal run train_self_play_modal.py' again")
    print(f"To use model: Restart your bot with 'python serve.py'")
