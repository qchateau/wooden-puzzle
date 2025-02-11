import argparse
import logging
import itertools
import numpy as np
from dataclasses import dataclass


log = logging.getLogger()


@dataclass
class Piece:
    id: int
    cubes: np.ndarray

    def permutations(self):
        res = []
        for rot in ROTATIONS:
            for tr in TRANSLATIONS:
                cubes = rot @ self.cubes + tr
                if valid(cubes):
                    res.append(Piece(id=self.id, cubes=cubes))
        return res

    def possible_move_directions(self):
        return (
            2
            - (np.max(self.cubes, axis=1) - np.min(self.cubes, axis=1)).reshape(-1) // 2
        )


PIECES = [
    Piece(
        id=0,
        cubes=np.array(
            [
                [-2, -2, -2],
                [-2, 0, -2],
                #
                [-1, -2, -2],
                [-1, 0, -2],
                [-1, 2, -2],
                [-2, -1, -2],
                [-2, -1, 0],
                [-2, -1, 2],
                [-2, 0, -1],
                [0, 0, -1],
                [2, 0, -1],
            ]
        ).T,
    ),
    Piece(
        id=1,
        cubes=np.array(
            [
                [-2, -2, -2],
                [-2, 0, -2],
                [2, -2, -2],
                [-2, 2, 0],
                #
                [-2, -2, -1],
                [-2, 0, -1],
                [-2, 2, -1],
                [-2, -1, -2],
                [0, -1, -2],
                [2, -1, -2],
                [-1, 0, -2],
            ]
        ).T,
    ),
    Piece(
        id=2,
        cubes=np.array(
            [
                [-2, 0, -2],
                [0, 0, -2],
                [0, -2, -2],
                #
                [-2, -1, -2],
                [0, -1, -2],
                [2, -1, -2],
                [0, -2, -1],
                [0, 0, -1],
                [0, 2, -1],
            ]
        ).T,
    ),
    Piece(
        id=3,
        cubes=np.array(
            [
                [-2, 0, -2],
                [2, 0, -2],
                [-2, -2, 0],
                #
                [-2, -1, -2],
                [0, -1, -2],
                [2, -1, -2],
                [-2, -2, -1],
                [-2, 0, -1],
                [-2, 2, -1],
                [1, 0, -2],
            ]
        ).T,
    ),
    Piece(
        id=4,
        cubes=np.array(
            [
                [-2, -2, -2],
                [0, 2, -2],
                #
                [-2, -2, -1],
                [0, -2, -1],
                [2, -2, -1],
                [-1, -2, -2],
                [-1, 0, -2],
                [-1, 2, -2],
            ]
        ).T,
    ),
    Piece(
        id=5,
        cubes=np.array(
            [
                [-2, -2, -2],
                [0, -2, -2],
                [0, 2, -2],
                [2, 2, 0],
                #
                [-1, -2, -2],
                [-1, 0, -2],
                [-1, 2, -2],
                [-2, 2, -1],
                [0, 2, -1],
                [2, 2, -1],
            ]
        ).T,
    ),
    Piece(
        id=6,
        cubes=np.array(
            [
                [-2, -2, -2],
                [0, -2, -2],
                [-2, 2, -2],
                [0, 2, -2],
                #
                [-2, -2, -1],
                [0, -2, -1],
                [2, -2, -1],
                [-1, -2, -2],
                [-1, 0, -2],
                [-1, 2, -2],
                [0, -1, -2],
            ]
        ).T,
    ),
    Piece(
        id=7,
        cubes=np.array(
            [
                [-2, -2, -2],
                [-2, 0, -2],
                [2, 0, -2],
                [-2, 2, 0],
                #
                [-2, -2, -1],
                [-2, 0, -1],
                [-2, 2, -1],
                [-2, -1, -2],
                [0, -1, -2],
                [2, -1, -2],
            ]
        ).T,
    ),
]


def valid(pos: np.ndarray):
    return np.all(np.logical_and(-2 <= pos, pos <= 2))


def rot_matrix(x: float, y: float, z: float):
    x_mat = np.array(
        [
            [1, 0, 0],
            [0, np.cos(x), np.sin(x)],
            [0, -np.sin(x), np.cos(x)],
        ]
    )
    y_mat = np.array(
        [
            [np.cos(y), 0, -np.sin(y)],
            [0, 1, 0],
            [np.sin(y), 0, np.cos(y)],
        ]
    )
    z_mat = np.array(
        [
            [np.cos(z), np.sin(z), 0],
            [-np.sin(z), np.cos(z), 0],
            [0, 0, 1],
        ]
    )
    return np.round(x_mat @ y_mat @ z_mat).astype(np.int64)


def solution_is_valid(pieces: list[Piece]):
    all_cubes = np.concatenate([p.cubes for p in pieces], axis=1).astype(np.int64)
    encoded = np.asarray(ENCODE_MATRIX @ all_cubes).reshape(-1)
    unique = np.unique(encoded)
    valid = len(unique) == len(encoded)
    return valid


def unique(alist: list[np.ndarray] | list[Piece]):
    unique = []
    for element in alist:
        matrix = element.cubes if isinstance(element, Piece) else element
        if not any(np.all(m == matrix) for m in unique):
            unique.append(element)
    return unique


ENCODE_MATRIX = np.array([[1, 10, 100]])

TRANLATION_RANGE = [-4, -2, 0, 2, 4]
TRANSLATIONS = [
    np.array([[x, y, z]]).T
    for x in TRANLATION_RANGE
    for y in TRANLATION_RANGE
    for z in TRANLATION_RANGE
]

ROTATION_RANGE = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
ROTATIONS = unique(
    [
        rot_matrix(x, y, z)
        for x in ROTATION_RANGE
        for y in ROTATION_RANGE
        for z in ROTATION_RANGE
    ]
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    pieces_perms = []
    for piece in PIECES:
        perms = unique(piece.permutations())
        log.info(
            "Piece %s has %d permutations, move directions: %s",
            piece.id,
            len(perms),
            piece.possible_move_directions(),
        )
        pieces_perms.append(perms)
    pieces_perms = list(sorted(pieces_perms, key=lambda x: len(x)))
    for piece_perms in reversed(pieces_perms):
        if np.all(piece_perms[0].possible_move_directions() == [0, 0, 0]):
            log.info("Piece %d used as reference point", piece_perms[0].id)
            piece_perms[:] = piece_perms[:1]
            break

    perm_factors = np.array(
        list(
            reversed(
                np.cumulative_prod(
                    np.array([1] + list(reversed([len(p) for p in pieces_perms])))
                )
            )
        )
    )
    total = np.cumulative_prod(perm_factors)[0]
    log.info("Total combinations: %d", total)

    solutions = []
    perm_factors = perm_factors[1:]
    indices = np.array([0] * len(PIECES))
    depth = len(PIECES) - 1
    done = False
    count = 0
    checks = 0
    while not done:
        count += 1
        if count % 1000 == 0:
            progress = (np.dot(indices, perm_factors)) / total
            log.info("Indices: %s, %.2f%% (%s checks)", indices, progress * 100, checks)

        skipped_depth = len(PIECES) - 1
        search = [perm[i] for i, perm in zip(indices, pieces_perms)]
        for depth in range(1, len(PIECES)):
            minisearch = search[: (depth + 1)]

            checks += 1
            if not solution_is_valid(minisearch):
                skipped_depth = depth
                break

            if len(minisearch) == len(search):
                solutions.append(search)
                log.info("Solution found ! Indices: %s", indices)
                break

        for depth in reversed(range(0, skipped_depth + 1)):
            indices[depth] += 1
            if indices[depth] < len(pieces_perms[depth]):
                break
            if depth == 0:
                done = True
            indices[depth:] = 0

    log.info("Covered %d permutations in %d checks", total, checks)

    print("Done, solutions:")
    for solution in solutions:
        print("=" * 80)
        for piece in solution:
            print()
            print(f"Piece {piece.id}")
            print(piece.cubes)


if __name__ == "__main__":
    main()
