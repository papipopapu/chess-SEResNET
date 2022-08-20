#include <fdeep/fdeep.hpp>
#include <cstring>
#include "position.h"
#include "tables.h"
#include "types.h"
#include <iostream>
#include <string>
#define IX(row, col, type) ((row)*96 + (col)*12 + (type))
#define ENFORCE(x) typename = typename std::enable_if<(x)>::type
fdeep::tensor one_hot_encode(const Position& pos) {
    std::vector<float> one_hot_encoded(768, 0.0);
    int type;
    Piece piece;
    for (int row = 0; row < 8; row++) {
        for (int col = 0; col < 8; col++) {
            piece = pos.at(Square(row*8 + col));
            if (piece != NO_PIECE) {
                switch(piece) {
                    case WHITE_PAWN:
                        type = 0;
                        break;
                    case WHITE_KNIGHT:
                        type = 1;
                        break;
                    case WHITE_BISHOP:
                        type = 2;
                        break;
                    case WHITE_ROOK:
                        type = 3;
                        break;
                    case WHITE_QUEEN:
                        type = 4;
                        break;
                    case WHITE_KING:
                        type = 5;
                        break;
                    case BLACK_PAWN:
                        type = 6;
                        break;
                    case BLACK_KNIGHT:
                        type = 7;
                        break;
                    case BLACK_BISHOP:
                        type = 8;
                        break;
                    case BLACK_ROOK:   
                        type = 9;
                        break;
                    case BLACK_QUEEN:
                        type = 10;
                        break;
                    case BLACK_KING:
                        type = 11;
                        break;
                } 
                one_hot_encoded[IX(7-row, col, type)] = 1; 
            }        
        }
    }
    return fdeep::tensor(fdeep::tensor_shape(8,8,12), one_hot_encoded);
}
fdeep::tensor one_hot_encode_mirror(const Position& pos) {
    std::vector<float> one_hot_encoded(768, 0.0);
    int type;
    Piece piece;
    for (int row = 0; row < 8; row++) {
        for (int col = 0; col < 8; col++) {
            piece = pos.at(Square(row*8 + col));
            if (piece != NO_PIECE) {
                switch(piece) {
                    case WHITE_PAWN:
                        type = 6;
                        break;
                    case WHITE_KNIGHT:
                        type = 7;
                        break;
                    case WHITE_BISHOP:
                        type = 8;
                        break;
                    case WHITE_ROOK:
                        type = 9;
                        break;
                    case WHITE_QUEEN:
                        type = 10;
                        break;
                    case WHITE_KING:
                        type = 11;
                        break;
                    case BLACK_PAWN:
                        type = 0;
                        break;
                    case BLACK_KNIGHT:
                        type = 1;
                        break;
                    case BLACK_BISHOP:
                        type = 2;
                        break;
                    case BLACK_ROOK:   
                        type = 3;
                        break;
                    case BLACK_QUEEN:
                        type = 4;
                        break;
                    case BLACK_KING:
                        type = 5;
                        break;
                }  
                one_hot_encoded[IX(row, col, type)] = 1; 
            }        
        }
    }
    return fdeep::tensor(fdeep::tensor_shape(8,8,12), one_hot_encoded);
}
template<Color Us>
float evaluate(const Position& pos, const fdeep::model& model);
template<>
float evaluate<BLACK>(const Position& pos, const fdeep::model& model) {
    fdeep::tensor input = one_hot_encode(pos);
    return model.predict_single_output(std::vector<fdeep::tensor>{input});
}
template<>
float evaluate<WHITE>(const Position& pos, const fdeep::model& model) {
    fdeep::tensor input =  one_hot_encode_mirror(pos);
    return 1.0 - model.predict_single_output(std::vector<fdeep::tensor>{input});

}
template<uint depthleft>
float alphaBetaMin(const fdeep::model& model, Position &pos, float alpha, float beta );
template<uint depthleft>
float alphaBetaMax(const fdeep::model& model, Position &pos, float alpha, float beta ) {
    float score;
    MoveList<WHITE> legals(pos);
    for (Move move : legals) {
        pos.play<WHITE>(move);
        score = alphaBetaMin<depthleft-1>( model, pos, alpha, beta);
        pos.undo<WHITE>(move);
        if( score >= beta )
            return beta;   // (cant occur at depth 1, since beta=inf)
        if( score > alpha )
            alpha = score; // we want the best score out of the minimums
    }
    return alpha;
}
template<>
float alphaBetaMax<0U>(const fdeep::model& model, Position &pos, float alpha, float beta ) {
    return evaluate<WHITE>(pos, model);
}

template<uint depthleft>
float alphaBetaMin(const fdeep::model& model, Position &pos, float alpha, float beta ) {
    float score;
    MoveList<BLACK> legals(pos);
    for (Move move : legals) {
        pos.play<BLACK>(move);
        score = alphaBetaMax<depthleft-1>( model, pos, alpha, beta );
        pos.undo<BLACK>(move);
        if( score <= alpha )
            return alpha; // fail hard alpha-cutoff
        if( score < beta )
            beta = score; // we want the worst score out of the maximums
    }
    return beta;
}
template<>
float alphaBetaMin<0U>(const fdeep::model& model, Position &pos, float alpha, float beta ) {
    return evaluate<BLACK>(pos, model);
}

template<uint depth>
Move getBestMove(const fdeep::model& model, Position &pos) {
    float bestScore, score;
    Move bestMove;
    if (pos.turn() == WHITE) {
        bestScore = -1;
        MoveList<WHITE> legals(pos);
        for (Move move : legals) {
            pos.play<WHITE>(move);
            score = alphaBetaMin<depth>(model, pos, bestScore, 2);
            pos.undo<WHITE>(move);
            if (score > bestScore) {
                bestScore = score;
                bestMove = move;
            }
        }
    } else {
        bestScore = 2;
        MoveList<BLACK> legals(pos);
        for (Move move : legals) {
            pos.play<BLACK>(move);
            score = alphaBetaMax<depth>(model, pos, -1, bestScore);
            pos.undo<BLACK>(move);
            if (score < bestScore) {
                bestScore = score;
                bestMove = move;
            }
        }
    }
    return bestMove;
}
int main() {
    // program that recieves a fen string, and a depth and returns the eval by the model_ser_mid.json
    const auto model = fdeep::load_model("model_ser_mid.json");
    initialise_all_databases();
	zobrist::initialise_zobrist_keys();
    
    Move bestMove;
    for (;;) {
        std::cin.clear();
        std::cin.sync();
        std::string fen, depth_str;
        int depth;
        Position pos;
        std::cout << "fen: ";
        std::getline(std::cin, fen);
        if (fen == "quit") {
            return 0;
        }
        Position::set(fen, pos);
        std::cout << "depth: ";
        std::getline(std::cin, depth_str);
        depth = std::stoi(depth_str);
        assert(depth >= 0);
        
        if (pos.turn() == WHITE) {
            switch (depth) {
                case 0:
                std::cout << "white: " << alphaBetaMax<0U>(model, pos, -1.f, 2.f) << std::endl;
                bestMove = getBestMove<0U>(model, pos);
                std::cout << "best move: " << bestMove << std::endl;
                break;
                case 1:
                std::cout << "white: " << alphaBetaMax<1U>(model, pos, -1.f, 2.f) << std::endl;
                bestMove = getBestMove<1U>(model, pos);
                std::cout << "best move: " << bestMove << std::endl;
                break;
                case 2:
                std::cout << "white: " << alphaBetaMax<2U>(model, pos, -1.f, 2.f) << std::endl;
                bestMove = getBestMove<2U>(model, pos);
                std::cout << "best move: " << bestMove << std::endl;
                break;
                case 3:
                std::cout << "white: " << alphaBetaMax<3U>(model, pos, -1.f, 2.f) << std::endl;
                bestMove = getBestMove<3U>(model, pos);
                std::cout << "best move: " << bestMove << std::endl;
                break;
                case 4:
                std::cout << "white: " << alphaBetaMax<4U>(model, pos, -1.f, 2.f) << std::endl;
                bestMove = getBestMove<4U>(model, pos);
                std::cout << "best move: " << bestMove << std::endl;
                break;
                case 5:
                std::cout << "white: " << alphaBetaMax<5U>(model, pos, -1.f, 2.f) << std::endl;
                bestMove = getBestMove<5U>(model, pos);
                std::cout << "best move: " << bestMove << std::endl;
                break;
                case 6:
                std::cout << "white: " << alphaBetaMax<6U>(model, pos, -1.f, 2.f) << std::endl;
                bestMove = getBestMove<6U>(model, pos);
                std::cout << "best move: " << bestMove << std::endl;
                break;
                default:
                std::cout << "too deep!"<< std::endl;
            }
        } else {
            switch (depth) {
                case 0:
                std::cout << "white: " << alphaBetaMin<0U>(model, pos, -1.f, 2.f) << std::endl;
                bestMove = getBestMove<0U>(model, pos);
                std::cout << "best move: " << bestMove << std::endl;
                break;
                case 1:
                std::cout << "white: " << alphaBetaMin<1U>(model, pos, -1.f, 2.f) << std::endl;
                bestMove = getBestMove<1U>(model, pos);
                std::cout << "best move: " << bestMove << std::endl;        
                break;
                case 2:
                std::cout << "white: " << alphaBetaMin<2U>(model, pos, -1.f, 2.f) << std::endl;
                bestMove = getBestMove<2U>(model, pos);
                std::cout << "best move: " << bestMove << std::endl;
                break;
                case 3:
                std::cout << "white: " << alphaBetaMin<3U>(model, pos, -1.f, 2.f) << std::endl;
                bestMove = getBestMove<3U>(model, pos);
                std::cout << "best move: " << bestMove << std::endl;
                break;
                case 4:
                std::cout << "white: " << alphaBetaMin<4U>(model, pos, -1.f, 2.f) << std::endl;
                bestMove = getBestMove<4U>(model, pos);
                std::cout << "best move: " << bestMove << std::endl;
                break;
                case 5:
                std::cout << "white: " << alphaBetaMin<5U>(model, pos, -1.f, 2.f) << std::endl;
                bestMove = getBestMove<5U>(model, pos);
                std::cout << "best move: " << bestMove << std::endl;
                break;
                case 6:
                std::cout << "white: " << alphaBetaMin<6U>(model, pos, -1.f, 2.f) << std::endl;
                bestMove = getBestMove<6U>(model, pos);
                std::cout << "best move: " << bestMove << std::endl;
                break;
                default:
                std::cout << "too deep!"<< std::endl;
            }
        }
        
    }


    return 0;
}