# chess-SEResNET

A chess evaluation engine powered by a Squeeze-Excitation Residual Neural Network (SE-ResNet). This project implements a neural network-based position evaluation for chess, trained on 35 million positions and deployed in C++ for efficient inference.

## Features

- **Neural Network Evaluation**: Small (300k parameters) SE-ResNet architecture providing accurate position assessment
- **Alpha-Beta Search**: Minimax search with alpha-beta pruning for move selection
- **Bitboard Representation**: Efficient board representation using 64-bit bitboards
- **Magic Bitboards**: Fast sliding piece attack generation using magic bitboard technique
- **Interactive Interface**: Simple command-line interface for position evaluation

## Project Structure

```
chess-seresnet/
├── core/                   # Core chess engine library
│   ├── types.h/cpp        # Type definitions and utilities
│   ├── tables.h/cpp       # Attack tables and magic bitboards
│   └── position.h/cpp     # Position representation and move generation
├── src/
│   └── main.cpp           # Main application with neural network evaluation
├── model_ser_mid.json     # Trained neural network model
├── CMakeLists.txt         # CMake build configuration
└── README.md              # This file
```

## Dependencies

This project requires the following header-only libraries:

1. **[frugally-deep](https://github.com/Dobiasd/frugally-deep)** - Keras model loader and inference
2. **[FunctionalPlus](https://github.com/Dobiasd/FunctionalPlus)** - Functional programming utilities (required by frugally-deep)
3. **[Eigen](https://eigen.tuxfamily.org/)** - Linear algebra library (required by frugally-deep)

### Installing Dependencies

```bash
# Clone dependencies into the project directory
git clone https://github.com/Dobiasd/frugally-deep.git
git clone https://github.com/Dobiasd/FunctionalPlus.git
git clone https://gitlab.com/libeigen/eigen.git
```

## Building

### Using CMake

```bash
# Create build directory
mkdir build && cd build

# Configure with CMake
cmake ..

# Build
cmake --build . --config Release

# Or simply:
make -j$(nproc)
```

### Custom Dependency Paths

If your dependencies are installed elsewhere, you can specify their locations:

```bash
cmake .. \
    -DFDEEP_INCLUDE_DIR=/path/to/frugally-deep/include \
    -DFUNCTIONALPLUS_INCLUDE_DIR=/path/to/FunctionalPlus/include \
    -DEIGEN_INCLUDE_DIR=/path/to/eigen
```

## Usage

Run the executable from the project root (where `model_ser_mid.json` is located):

```bash
./build/chess-seresnet
```

The program accepts FEN positions and search depth:

```
fen: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -
depth: 2
eval: 0.5
best move: e2e4
```

The evaluation score is from White's perspective: values closer to 1.0 favor White, values closer to 0.0 favor Black.

Enter `quit` to exit the program.

### Example Positions

```
# Starting position
fen: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -

# Sicilian Defense
fen: rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6

# Italian Game
fen: r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq -
```

## Performance Notes

- **Depth 0-2**: Fast evaluation, suitable for real-time play
- **Depth 3+**: Significantly slower due to exponential growth in positions to evaluate
- The neural network evaluation adds overhead compared to traditional evaluation functions

## Technical Details

### Neural Network Architecture

The SE-ResNet model uses:
- Input: 8×8×12 one-hot encoded board (12 channels for piece types)
- Output: Single value [0, 1] representing position evaluation (0 = Black winning, 1 = White winning)

### Board Encoding

Each position is encoded as a 768-element tensor:
- 8 rows × 8 columns × 12 piece types
- Piece types: White (P, N, B, R, Q, K) and Black (p, n, b, r, q, k)

### Move Generation

The engine uses bitboard representation with:
- Magic bitboards for sliding piece attacks
- Hyperbola Quintessence algorithm for initialization
- Complete legal move generation including castling and en passant

## Credits

- Chess move generation based on [surge](https://github.com/nkarve/surge) library
- Neural network inference using [frugally-deep](https://github.com/Dobiasd/frugally-deep)
- Model trained using TensorFlow/Keras on Lichess game data

## License

This project is provided as-is for educational purposes.
