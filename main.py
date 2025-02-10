import time
import itertools
import numpy as np
from dataclasses import dataclass


def valid(pos: np.matrix):
    return np.all(np.logical_and(-2 <= pos, pos <= 2))


def rot_matrix(x: float, y: float, z: float):
    x_mat = np.matrix(
        [
            [1, 0, 0],
            [0, np.cos(x), np.sin(x)],
            [0, -np.sin(x), np.cos(x)],
        ]
    )
    y_mat = np.matrix(
        [
            [np.cos(y), 0, -np.sin(y)],
            [0, 1, 0],
            [np.sin(y), 0, np.cos(y)],
        ]
    )
    z_mat = np.matrix(
        [
            [np.cos(z), np.sin(z), 0],
            [-np.sin(z), np.cos(z), 0],
            [0, 0, 1],
        ]
    )
    return x_mat * y_mat * z_mat


@dataclass
class Piece:
    id: int
    cubes: np.matrix
    plates: np.matrix

    def permutations(self):
        res = []
        for rot in ROTATIONS:
            for tr in TRANSLATIONS:
                cubes = rot * self.cubes + tr
                if valid(cubes):
                    res.append(Piece(id=self.id, cubes=cubes))
        return res


ENCODE_MAT = np.matrix([[1, 10, 100]])


def solution_is_valid(pieces: list[Piece]):
    all_cubes = np.concatenate([p.cubes for p in pieces], axis=1)
    encoded = np.asarray(ENCODE_MAT * all_cubes).reshape(-1).astype(np.int64)
    unique = np.unique(encoded)
    return len(unique) == len(encoded)


PIECES = [
    Piece(
        id=0,
        cubes=np.matrix(
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
        cubes=np.matrix(
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
        cubes=np.matrix(
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
        cubes=np.matrix(
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
        cubes=np.matrix(
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
        cubes=np.matrix(
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
        cubes=np.matrix(
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
        cubes=np.matrix(
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


TRANLATION_RANGE = [-4, -2, 0, 2, 4]
TRANSLATIONS = [
    np.matrix([[x, y, z]]).T
    for x in TRANLATION_RANGE
    for y in TRANLATION_RANGE
    for z in TRANLATION_RANGE
]

ROTATION_RANGE = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
ROTATIONS = [
    rot_matrix(x, y, z)
    for x in ROTATION_RANGE
    for y in ROTATION_RANGE
    for z in ROTATION_RANGE
]


def encode(cube):
    return np.sum(np.multiply(cube.T, ENCODE_MAT))


def main():
    pieces_perms = []
    for piece in PIECES:
        perms = piece.permutations()
        print(f"Piece {piece.id} has {len(perms)} permutations")
        pieces_perms.append(perms)
    pieces_perms = list(sorted(pieces_perms, key=lambda x: len(x)))

    perm_factors = np.array(
        list(
            reversed(
                np.cumulative_prod(
                    np.array([1] + list(reversed([len(p) for p in pieces_perms])))
                )
            )
        )
    )
    print(perm_factors)
    total = np.cumulative_prod(perm_factors)[0]
    print(f"Total combinations: {total}")

    perm_factors = perm_factors[1:]
    indices = np.array([0] * len(PIECES))
    depth = len(PIECES) - 1

    done = False
    solutions = []
    count = 0
    while not done:
        count += 1
        if count % 1000 == 0:
            progress = (np.dot(indices, perm_factors)) / total
            print(f"indices: {indices}, {progress*100:.2}%")

        skipped_depth = len(PIECES) - 1
        search = [perm[i] for i, perm in zip(indices, pieces_perms)]
        for depth in range(1, len(PIECES)):
            minisearch = search[: (depth + 1)]

            if not solution_is_valid(minisearch):
                skipped_depth = depth
                break

            if len(minisearch) == len(search):
                solutions.append(search)
                print("Solution found !")
                break

        for depth in reversed(range(0, skipped_depth + 1)):
            indices[depth] += 1
            if indices[depth] < len(pieces_perms[depth]):
                break
            if depth == 0:
                done = True
            indices[depth:] = 0


if __name__ == "__main__":
    main()
