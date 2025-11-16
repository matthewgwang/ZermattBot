"""
Quick self-play training for fast iteration (~15-20 minutes)
Perfect for testing and rapid experimentation
"""

import modal

app = modal.App("chess-quick-selfplay")

volume = modal.Volume.from_name("chess-selfplay-data", create_if_missing=True)

image = (
    modal.Image.debian_slim()
    .pip_install("torch", "numpy", "python-chess")
)


@app.function(
    image=image,
    gpu="T4",
    volumes={"/data": volume},
    timeout=3600,  # 1 hour max
)
def quick_self_play(iteration: int):
    """
    Quick self-play iteration - optimized for speed
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    import chess
    import numpy as np
    import os

    print(f"\n{'='*60}")
    print(f"QUICK SELF-PLAY ITERATION {iteration}")
    print(f"{'='*60}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ===== MODEL DEFINITION (copied from chess_bot.py) =====
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

    # Load or create model
    model = ChessNet1500().to(device)

    model_files = [f for f in os.listdir('/data') if f.startswith('chess_model_iter') and f.endswith('.pth')]
    if model_files:
        model_files.sort()
        latest = model_files[-1]
        print(f"Loading {latest}")
        model.load_state_dict(torch.load(f'/data/{latest}', map_location=device, weights_only=True))
    else:
        print("Starting with fresh model")

    # ===== QUICK SELF-PLAY (simplified, no MCTS) =====
    print(f"\nPlaying 15 quick games...")
    all_training_data = []

    for game_num in range(15):  # Only 15 games for speed
        board = chess.Board()
        game_data = []
        move_count = 0

        while not board.is_game_over() and move_count < 100:
            state = board_to_tensor(board).to(device)

            with torch.no_grad():
                policy, value = model(state)

            # Get legal moves and their policy scores
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                break

            move_scores = {}
            for move in legal_moves:
                idx = move.from_square * 64 + move.to_square
                move_scores[move] = policy[0, idx].item()

            # Temperature-based selection
            temp = 1.0 if move_count < 30 else 0.1
            scores = np.array(list(move_scores.values()))
            scores = scores / temp
            scores = np.exp(scores - scores.max())
            probs = scores / scores.sum()

            # Store training data
            policy_target = np.zeros(4096, dtype=np.float32)
            for move, prob in zip(move_scores.keys(), probs):
                idx = move.from_square * 64 + move.to_square
                policy_target[idx] = prob

            game_data.append((state.cpu().squeeze(0).numpy(), policy_target, None))

            # Make move
            move = np.random.choice(legal_moves, p=probs)
            board.push(move)
            move_count += 1

        # Assign outcome
        result = board.result()
        outcome = 1.0 if result == "1-0" else (-1.0 if result == "0-1" else 0.0)

        for i, (state, policy, _) in enumerate(game_data):
            value = outcome if i % 2 == 0 else -outcome
            all_training_data.append((state, policy, value))

        if (game_num + 1) % 5 == 0:
            print(f"  Game {game_num + 1}/15 complete")

    print(f"Generated {len(all_training_data)} training positions")

    # ===== QUICK TRAINING =====
    class QuickDataset(Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            state, policy, value = self.data[idx]
            return (
                torch.FloatTensor(state),
                torch.FloatTensor(policy),
                torch.FloatTensor([value])
            )

    dataset = QuickDataset(all_training_data)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    print(f"\nTraining for 3 epochs...")
    model.train()

    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(3):  # Only 3 epochs for speed
        total_p, total_v, num = 0, 0, 0

        for boards, policy_targets, value_targets in dataloader:
            boards = boards.to(device)
            policy_targets = policy_targets.to(device)
            value_targets = value_targets.to(device)

            optimizer.zero_grad()
            policy_pred, value_pred = model(boards)

            p_loss = -(policy_targets * torch.log_softmax(policy_pred, dim=1)).sum(dim=1).mean()
            v_loss = value_criterion(value_pred, value_targets)
            loss = p_loss + v_loss

            loss.backward()
            optimizer.step()

            total_p += p_loss.item()
            total_v += v_loss.item()
            num += 1

        print(f"  Epoch {epoch+1}/3: P={total_p/num:.4f}, V={total_v/num:.4f}")

    # Save model
    model_cpu = model.cpu()
    save_path = f'/data/chess_model_iter{iteration}.pth'
    torch.save(model_cpu.state_dict(), save_path)
    volume.commit()

    print(f"\n✓ Model saved to {save_path}")

    with open(save_path, 'rb') as f:
        return f.read()


@app.local_entrypoint()
def main():
    """Run quick self-play training"""
    import os

    num_iterations = 2  # Just 2 iterations for quick test

    print(f"\n{'='*60}")
    print(f"QUICK SELF-PLAY TRAINING (Fast Mode)")
    print(f"{'='*60}")
    print(f"Iterations: {num_iterations}")
    print(f"Games per iteration: 15")
    print(f"Training epochs: 3")
    print(f"Estimated time: 15-20 minutes")
    print(f"{'='*60}\n")

    for iteration in range(num_iterations):
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration}")
        print(f"{'='*60}\n")

        model_bytes = quick_self_play.remote(iteration)

        with open(f'chess_model_iter{iteration}.pth', 'wb') as f:
            f.write(model_bytes)

        print(f"\n✓ Iteration {iteration} complete!")

    # Deploy final model to bot automatically
    final_iter = num_iterations - 1
    print(f"\n{'='*60}")
    print(f"QUICK TRAINING COMPLETE!")
    print(f"{'='*60}")

    # Auto-deploy to bot
    import shutil
    shutil.copy(f'chess_model_iter{final_iter}.pth', 'src/chess_model.pth')
    print(f"✓ Automatically deployed to src/chess_model.pth")
    print(f"\nReload your dev server to use the new model!")
    print(f"Or run multiple quick sessions to keep improving!")
