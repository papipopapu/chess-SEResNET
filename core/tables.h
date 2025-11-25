/**
 * @file tables.h
 * @brief Attack table declarations for chess move generation
 * 
 * Contains lookup tables and functions for computing piece attacks
 * using magic bitboards for sliding pieces (rooks, bishops, queens).
 */

#pragma once

#include "types.h"

// Precomputed attack tables for non-sliding pieces
extern const Bitboard KING_ATTACKS[NSQUARES];
extern const Bitboard KNIGHT_ATTACKS[NSQUARES];
extern const Bitboard WHITE_PAWN_ATTACKS[NSQUARES];
extern const Bitboard BLACK_PAWN_ATTACKS[NSQUARES];

// Bitboard manipulation utilities
extern Bitboard reverse(Bitboard b);
extern Bitboard sliding_attacks(Square square, Bitboard occ, Bitboard mask);

// Rook attack generation (magic bitboards)
extern Bitboard get_rook_attacks_for_init(Square square, Bitboard occ);
extern const Bitboard ROOK_MAGICS[NSQUARES];
extern Bitboard ROOK_ATTACK_MASKS[NSQUARES];
extern int ROOK_ATTACK_SHIFTS[NSQUARES];
extern Bitboard ROOK_ATTACKS[NSQUARES][4096];
extern void initialise_rook_attacks();

extern Bitboard get_rook_attacks(Square square, Bitboard occ);
extern Bitboard get_xray_rook_attacks(Square square, Bitboard occ, Bitboard blockers);

// Bishop attack generation (magic bitboards)
extern Bitboard get_bishop_attacks_for_init(Square square, Bitboard occ);
extern const Bitboard BISHOP_MAGICS[NSQUARES];
extern Bitboard BISHOP_ATTACK_MASKS[NSQUARES];
extern int BISHOP_ATTACK_SHIFTS[NSQUARES];
extern Bitboard BISHOP_ATTACKS[NSQUARES][512];
extern void initialise_bishop_attacks();

extern Bitboard get_bishop_attacks(Square square, Bitboard occ);
extern Bitboard get_xray_bishop_attacks(Square square, Bitboard occ, Bitboard blockers);

// Lookup tables for squares between two squares and aligned squares
extern Bitboard SQUARES_BETWEEN_BB[NSQUARES][NSQUARES];
extern Bitboard LINE[NSQUARES][NSQUARES];

// Combined attack tables
extern Bitboard PAWN_ATTACKS[NCOLORS][NSQUARES];
extern Bitboard PSEUDO_LEGAL_ATTACKS[NPIECE_TYPES][NSQUARES];

// Initialization functions for attack tables
extern void initialise_squares_between();
extern void initialise_line();
extern void initialise_pseudo_legal();
extern void initialise_all_databases();

/**
 * @brief Returns attacks for a piece at a given square
 * @tparam P Piece type (compile-time constant)
 * @param s Square the piece is on
 * @param occ Occupancy bitboard
 * @return Bitboard of all squares the piece can attack
 */
template<PieceType P>
constexpr Bitboard attacks(Square s, Bitboard occ) {
	static_assert(P != PAWN, "The piece type may not be a pawn; use pawn_attacks instead");
	return P == ROOK ? get_rook_attacks(s, occ) :
		P == BISHOP ? get_bishop_attacks(s, occ) :
		P == QUEEN ? attacks<ROOK>(s, occ) | attacks<BISHOP>(s, occ) :
		PSEUDO_LEGAL_ATTACKS[P][s];
}

/**
 * @brief Returns attacks for a piece at a given square (runtime piece type)
 * @param pt Piece type
 * @param s Square the piece is on
 * @param occ Occupancy bitboard
 * @return Bitboard of all squares the piece can attack
 */
constexpr Bitboard attacks(PieceType pt, Square s, Bitboard occ) {
	switch (pt) {
	case ROOK:
		return attacks<ROOK>(s, occ);
	case BISHOP:
		return attacks<BISHOP>(s, occ);
	case QUEEN:
		return attacks<QUEEN>(s, occ);
	default:
		return PSEUDO_LEGAL_ATTACKS[pt][s];
	}
}

/**
 * @brief Returns pawn attacks from all pawns in a bitboard
 * @tparam C Color of the pawns
 * @param p Bitboard containing pawn positions
 * @return Bitboard of all squares attacked by the pawns
 */
template<Color C>
constexpr Bitboard pawn_attacks(Bitboard p) {
	return C == WHITE ? shift<NORTH_WEST>(p) | shift<NORTH_EAST>(p) :
		shift<SOUTH_WEST>(p) | shift<SOUTH_EAST>(p);
}

/**
 * @brief Returns pawn attacks from a single square
 * @tparam C Color of the pawn
 * @param s Square the pawn is on
 * @return Bitboard of squares attacked by the pawn
 */
template<Color C>
constexpr Bitboard pawn_attacks(Square s) {
	return PAWN_ATTACKS[C][s];
}
