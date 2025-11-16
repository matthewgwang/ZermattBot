"""
Train chess bot on Modal GPU using Volume storage
"""

import modal

app = modal.App("chess-trainer")

# Create a persistent volume
volume = modal.Volume.from_name("chess-data", create_if_missing=True)

# Define Docker image
image = (
    modal.Image.debian_slim()
    .pip_install("torch", "numpy", "python-chess")
)

@app.function(
    image=image,
    volumes={"/data": volume},
    gpu="any",  # Request GPU
    timeout=3600,  # 1 hour
)
def train():
    """Train with PGN already in volume"""
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    import chess
    import chess.pgn
    import numpy as np
    import os
    
    # Check if PGN exists
    if not os.path.exists('/data/games.pgn'):
        print("❌ PGN file not found in volume!")
        print("   Run: modal volume put chess-data games.pgn /games.pgn")
        return None
    
    print("\n" + "="*60)
    print("CHESS BOT TRAINING ON MODAL GPU")
    print("="*60 + "\n")
    
    # Check GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠ No GPU found, using CPU")
    
    # Model definition
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
    
    def move_to_index(move):
        return move.from_square * 64 + move.to_square
    
    class ChessDataset(Dataset):
        def __init__(self, pgn_file, max_games=5000, min_elo=2200):
            print(f"Loading games from {pgn_file}...")
            self.data = []
            
            with open(pgn_file) as f:
                game_count = 0
                while game_count < max_games:
                    game = chess.pgn.read_game(f)
                    if game is None:
                        break
                    
                    try:
                        white_elo = int(game.headers.get("WhiteElo", 0))
                        black_elo = int(game.headers.get("BlackElo", 0))
                        if white_elo < min_elo or black_elo < min_elo:
                            continue
                    except:
                        continue
                    
                    result = game.headers.get("Result", "*")
                    outcome = 1.0 if result == "1-0" else (-1.0 if result == "0-1" else 0.0)
                    
                    board = game.board()
                    for i, move in enumerate(game.mainline_moves()):
                        self.data.append({
                            'board': board.copy(),
                            'move': move,
                            'outcome': outcome if i % 2 == 0 else -outcome
                        })
                        board.push(move)
                    
                    game_count += 1
                    if game_count % 100 == 0:
                        print(f"  {game_count} games, {len(self.data)} positions")
            
            print(f"✓ Dataset ready: {len(self.data)} positions")
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            item = self.data[idx]
            return (
                board_to_tensor(item['board']).squeeze(0),
                move_to_index(item['move']),
                torch.FloatTensor([item['outcome']])
            )
    
    # Training
    model = ChessNet1500().to(device)
    
    dataset = ChessDataset('/data/games.pgn', max_games=5000)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=0)
    
    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(10):
        model.train()
        total_p, total_v, num = 0, 0, 0
        
        for batch_idx, (boards, moves, outcomes) in enumerate(dataloader):
            boards = boards.to(device)
            moves = moves.to(device)
            outcomes = outcomes.to(device)
            
            optimizer.zero_grad()
            policy_pred, value_pred = model(boards)
            
            p_loss = policy_criterion(policy_pred, moves)
            v_loss = value_criterion(value_pred, outcomes)
            loss = p_loss + 0.5 * v_loss
            
            loss.backward()
            optimizer.step()
            
            total_p += p_loss.item()
            total_v += v_loss.item()
            num += 1
            
            if batch_idx % 20 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}: P={p_loss.item():.4f}, V={v_loss.item():.4f}")
        
        avg_p = total_p / num
        avg_v = total_v / num
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1} Complete: P={avg_p:.4f}, V={avg_v:.4f}")
        print(f"{'='*60}\n")
    
    # Save model to volume
    model_cpu = model.cpu()
    torch.save(model_cpu.state_dict(), '/data/chess_model.pth')
    volume.commit()
    
    print("✓ Model saved to volume!")
    
    # Also return model as bytes
    with open('/data/chess_model.pth', 'rb') as f:
        return f.read()


@app.local_entrypoint()
def main():
    """Just trigger training - PGN should already be in volume"""
    print("🚀 Starting GPU training on Modal...\n")
    
    model_bytes = train.remote()
    
    if model_bytes:
        print("\n📥 Saving trained model locally...")
        with open('src/chess_model.pth', 'wb') as f:
            f.write(model_bytes)
        
        print("✓ Model saved to src/chess_model.pth")
        print("\n🎉 Training complete!")
    else:
        print("❌ Training failed")