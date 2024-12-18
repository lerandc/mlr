# Luis Rangel DaCosta 10 Dec. 2024
# As transcribed from Kelley and Knowles "Crystallography and Crystal Defects", 3rd. ed.
# Generators and Wyckoff positions transcribed from the Bilbao Crystallographic server

import numpy as np

POINT_GROUPS = ["1", "2", "m", "2mm", "4", "4mm", "3", "3m", "6", "6mm"]

PLANE_GROUP_DATA = {
    1: {"full_symbol": "p1", "short_symbol": "p1", "point_group": "1"},
    2: {"full_symbol": "p211", "short_symbol": "p2", "point_group": "2"},
    3: {"full_symbol": "p1m1", "short_symbol": "pm", "point_group": "m"},
    4: {"full_symbol": "p1g1", "short_symbol": "pg", "point_group": "m"},
    5: {"full_symbol": "c1m1", "short_symbol": "cm", "point_group": "m"},
    6: {"full_symbol": "p2mm", "short_symbol": "pmm", "point_group": "2mm"},
    7: {"full_symbol": "p2mg", "short_symbol": "pmg", "point_group": "2mm"},
    8: {"full_symbol": "p2gg", "short_symbol": "pgg", "point_group": "2mm"},
    9: {"full_symbol": "c2mm", "short_symbol": "cmm", "point_group": "2mm"},
    10: {"full_symbol": "p4", "short_symbol": "p4", "point_group": "4"},
    11: {"full_symbol": "p4mm", "short_symbol": "p4m", "point_group": "4mm"},
    12: {"full_symbol": "p4gm", "short_symbol": "p4g", "point_group": "4mm"},
    13: {"full_symbol": "p3", "short_symbol": "p3", "point_group": "3"},
    14: {"full_symbol": "p3m1", "short_symbol": "p3m1", "point_group": "3m"},
    15: {"full_symbol": "p31m", "short_symbol": "p31m", "point_group": "3m"},
    16: {"full_symbol": "p6", "short_symbol": "p6", "point_group": "6"},
    17: {"full_symbol": "p6mm", "short_symbol": "p6m", "point_group": "6mm"},
}

SYMBOL_TO_NUMBER = {}
for number, data in PLANE_GROUP_DATA.items():
    SYMBOL_TO_NUMBER[data["full_symbol"]] = number
    SYMBOL_TO_NUMBER[data["short_symbol"]] = number

# Store generators as affine matrices
GENERATORS = {
    1: [],
    2: [
        [
            [-1, 0, 0],
            [0, -1, 0],
        ],
    ],
    3: [
        [
            [-1, 0, 0],
            [0, 1, 0],
        ],
    ],
    4: [
        [
            [-1, 0, 0],
            [0, 1, 0.5],
        ],
    ],
    5: [
        [
            [-1, 0, 0],
            [0, 1, 0],
        ],
        [
            [1, 0, 0.5],
            [0, 1, 0.5],
        ],
    ],
    6: [
        [
            [-1, 0, 0],
            [0, -1, 0],
        ],
        [
            [-1, 0, 0],
            [0, 1, 0],
        ],
    ],
    7: [
        [
            [-1, 0, 0],
            [0, -1, 0],
        ],
        [
            [-1, 0, 0.5],
            [0, 1, 0],
        ],
    ],
    8: [
        [
            [-1, 0, 0],
            [0, -1, 0],
        ],
        [
            [-1, 0, 0.5],
            [0, 1, 0.5],
        ],
    ],
    9: [
        [
            [-1, 0, 0],
            [0, -1, 0],
        ],
        [
            [-1, 0, 0],
            [0, 1, 0],
        ],
        [
            [1, 0, 0.5],
            [0, 1, 0.5],
        ],
    ],
    10: [
        [
            [-1, 0, 0],
            [0, -1, 0],
        ],
        [
            [0, -1, 0],
            [1, 0, 0],
        ],
    ],
    11: [
        [
            [-1, 0, 0],
            [0, -1, 0],
        ],
        [
            [0, -1, 0],
            [1, 0, 0],
        ],
        [
            [-1, 0, 0],
            [0, 1, 0],
        ],
    ],
    12: [
        [
            [-1, 0, 0],
            [0, -1, 0],
        ],
        [
            [0, -1, 0],
            [1, 0, 0],
        ],
        [
            [-1, 0, 0.5],
            [0, 1, 0.5],
        ],
    ],
    13: [
        [
            [0, -1, 0],
            [1, -1, 0],
        ],
    ],
    14: [
        [
            [0, -1, 0],
            [1, -1, 0],
        ],
        [
            [0, -1, 0],
            [-1, 0, 0],
        ],
    ],
    15: [
        [
            [0, -1, 0],
            [1, -1, 0],
        ],
        [
            [0, 1, 0],
            [1, 0, 0],
        ],
    ],
    16: [
        [
            [0, -1, 0],
            [1, -1, 0],
        ],
        [
            [-1, 0, 0],
            [0, -1, 0],
        ],
    ],
    17: [
        [
            [0, -1, 0],
            [1, -1, 0],
        ],
        [
            [-1, 0, 0],
            [0, -1, 0],
        ],
        [
            [0, -1, 0],
            [-1, 0, 0],
        ],
    ],
}


WYCKOFF_POS = {
    1: {
        "(x, y)": {
            "site_symmetry": "1",
            "equivalent_pos": tuple(),
        },
    },
    2: {
        "(x, y)": {
            "site_symmetry": "1",
            "equivalent_pos": ("(-x, -y)",),
        },
        "(1/2, 1/2)": {
            "site_symmetry": "2",
            "equivalent_pos": tuple(),
        },
        "(0, 1/2)": {
            "site_symmetry": "2",
            "equivalent_pos": tuple(),
        },
        "(1/2, 0)": {
            "site_symmetry": "2",
            "equivalent_pos": tuple(),
        },
        "(0, 0)": {
            "site_symmetry": "2",
            "equivalent_pos": tuple(),
        },
    },
    3: {
        "(x, y)": {
            "site_symmetry": "1",
            "equivalent_pos": ("(-x, y)",),
        },
        "(1/2, y)": {
            "site_symmetry": ".m.",
            "equivalent_pos": tuple(),
        },
        "(0, y)": {
            "site_symmetry": ".m.",
            "equivalent_pos": tuple(),
        },
    },
    4: {
        "(x, y)": {
            "site_symmetry": "1",
            "equivalent_pos": ("(-x, y + 1/2)",),
        },
    },
    5: {
        "(x, y)": {
            "site_symmetry": "1",
            "equivalent_pos": (
                "(-x, y)",
                "(x+1/2, y+1/2)",
                "(-x-1/2, y+1/2)",
            ),
        },
        "(0, y)": {
            "site_symmetry": ".m.",
            "equivalent_pos": ("(1/2, y+1/2)",),
        },
    },
    6: {
        "(x, y)": {
            "site_symmetry": "1",
            "equivalent_pos": (
                "(-x, -y)",
                "(-x, y)",
                "(x, -y)",
            ),
        },
        "(1/2, y)": {
            "site_symmetry": ".m.",
            "equivalent_pos": ("(1/2, -y)",),
        },
        "(0, y)": {
            "site_symmetry": ".m.",
            "equivalent_pos": ("(0, -y)",),
        },
        "(x, 1/2)": {
            "site_symmetry": "..m",
            "equivalent_pos": ("(-x, 1/2)",),
        },
        "(x, 0)": {
            "site_symmetry": "..m",
            "equivalent_pos": ("(-x, 0)",),
        },
        "(1/2, 1/2)": {
            "site_symmetry": "2mm",
            "equivalent_pos": tuple(),
        },
        "(0, 1/2)": {
            "site_symmetry": "2mm",
            "equivalent_pos": tuple(),
        },
        "(1/2, 0)": {
            "site_symmetry": "2mm",
            "equivalent_pos": tuple(),
        },
        "(0, 0)": {
            "site_symmetry": "2mm",
            "equivalent_pos": tuple(),
        },
    },
    7: {
        "(x, y)": {
            "site_symmetry": "1",
            "equivalent_pos": (
                "(-x, -y)",
                "(-x + 1/2, y)",
                "(x + 1/2, -y)",
            ),
        },
        "(1/4, y)": {
            "site_symmetry": ".m.",
            "equivalent_pos": ("(3/4, -y)",),
        },
        "(0, 1/2)": {
            "site_symmetry": "2..",
            "equivalent_pos": ("(1/2, 1/2)",),
        },
        "(0, 0)": {
            "site_symmetry": "2..",
            "equivalent_pos": ("(1/2, 0)",),
        },
    },
    8: {
        "(x, y)": {
            "site_symmetry": "1",
            "equivalent_pos": (
                "(-x, -y)",
                "(-x + 1/2, y + 1/2)",
                "(x + 1/2, -y + 1/2)",
            ),
        },
        "(1/2, 0,)": {
            "site_symmetry": "2..",
            "equivalent_pos": ("(0, 1/2)",),
        },
        "(0, 0)": {
            "site_symmetry": "2..",
            "equivalent_pos": ("(1/2, 1/2)",),
        },
    },
    9: {
        "(x, y)": {
            "site_symmetry": "1",
            "equivalent_pos": (
                "(-x, -y)",
                "(-x, y)",
                "(x, -y)",
                "(x + 1/2, y + 1/2)",
                "(-x + 1/2, -y + 1/2)",
                "(-x + 1/2, y + 1/2)",
                "(x + 1/2, -y + 1/2)",
            ),
        },
        "(0, y)": {
            "site_symmetry": ".m.",
            "equivalent_pos": (
                "(0, -y)",
                "(1/2, y + 1/2)",
                "(1/2, -y - 1/2)",
            ),
        },
        "(x, 0)": {
            "site_symmetry": "..m",
            "equivalent_pos": (
                "(-x, 0)",
                "(x + 1/2, 1/2)",
                "(-x - 1/2, 1/2)",
            ),
        },
        "(1/4, 1/4)": {
            "site_symmetry": "2..",
            "equivalent_pos": (
                "(3/4, 1/4)",
                "(3/4, 3/4)",
                "(1/4, 3/4)",
            ),
        },
        "(0, 1/2)": {
            "site_symmetry": "2mm",
            "equivalent_pos": ("(1/2, 0)",),
        },
        "(0, 0)": {
            "site_symmetry": "2mm",
            "equivalent_pos": ("(1/2, 1/2)",),
        },
    },
    10: {
        "(x, y)": {
            "site_symmetry": "1",
            "equivalent_pos": (
                "(-x, -y)",
                "(-y, x)",
                "(y, -x)",
            ),
        },
        "(1/2, 0,)": {
            "site_symmetry": "2..",
            "equivalent_pos": ("(0, 1/2)",),
        },
        "(1/2, 1/2)": {
            "site_symmetry": "4..",
            "equivalent_pos": tuple(),
        },
        "(0, 0)": {
            "site_symmetry": "4..",
            "equivalent_pos": tuple(),
        },
    },
    11: {
        "(x, y)": {
            "site_symmetry": "1",
            "equivalent_pos": (
                "(-x, -y)",
                "(-y, x)",
                "(y, -x)",
                "(-x, y)",
                "(x, -y)",
                "(y, x)",
                "(-y, -x)",
            ),
        },
        "(x, x)": {
            "site_symmetry": "..m",
            "equivalent_pos": (
                "(-x, -x)",
                "(-x, x)",
                "(x, -x)",
            ),
        },
        "(x, 1/2)": {
            "site_symmetry": ".m.",
            "equivalent_pos": (
                "(-x, 1/2)",
                "(1/2, x)",
                "(1/2, -x)",
            ),
        },
        "(x, 0)": {
            "site_symmetry": ".m.",
            "equivalent_pos": (
                "(-x, 0)",
                "(0, x)",
                "(0, -x)",
            ),
        },
        "(1/2, 0,)": {
            "site_symmetry": "2mm",
            "equivalent_pos": ("(0, 1/2)",),
        },
        "(1/2, 1/2)": {
            "site_symmetry": "4mm",
            "equivalent_pos": tuple(),
        },
        "(0, 0)": {
            "site_symmetry": "4mm",
            "equivalent_pos": tuple(),
        },
    },
    12: {
        "(x, y)": {
            "site_symmetry": "1",
            "equivalent_pos": (
                "(-x, -y)",
                "(-y, x)",
                "(y, -x)",
                "(-x + 1/2, y + 1/2)",
                "(x + 1/2, -y + 1/2)",
                "(y + 1/2, x + 1/2)",
                "(-y + 1/2, -x + 1/2)",
            ),
        },
        "(x, x + 1/2)": {
            "site_symmetry": "..m",
            "equivalent_pos": (
                "(-x, -x + 1/2)",
                "(-x + 1/2, x)",
                "(x + 1/2, -x)",
            ),
        },
        "(1/2, 0,)": {
            "site_symmetry": "2.mm",
            "equivalent_pos": ("(0, 1/2)",),
        },
        "(0, 0)": {
            "site_symmetry": "4..",
            "equivalent_pos": ("(1/2, 1/2)",),
        },
    },
    13: {
        "(x, y)": {
            "site_symmetry": "1",
            "equivalent_pos": (
                "(-y, x-y)",
                "(-x+y, -x)",
            ),
        },
        "(2/3, 1/3)": {
            "site_symmetry": "3..",
            "equivalent_pos": tuple(),
        },
        "(1/3, 2/3)": {
            "site_symmetry": "3..",
            "equivalent_pos": tuple(),
        },
        "(0, 0)": {
            "site_symmetry": "3..",
            "equivalent_pos": tuple(),
        },
    },
    14: {
        "(x, y)": {
            "site_symmetry": "1",
            "equivalent_pos": (
                "(-y, x-y)",
                "(-x+y, -x)",
                "(-y, -x)",
                "(-x+y, y)",
                "(x, x-y)",
            ),
        },
        "(x, -x)": {
            "site_symmetry": ".m.",
            "equivalent_pos": (
                "(x, 2*x)",
                "(-2*x, -x)",
            ),
        },
        "(2/3, 1/3)": {
            "site_symmetry": "3m.",
            "equivalent_pos": tuple(),
        },
        "(1/3, 2/3)": {
            "site_symmetry": "3m.",
            "equivalent_pos": tuple(),
        },
        "(0, 0)": {
            "site_symmetry": "3m.",
            "equivalent_pos": tuple(),
        },
    },
    15: {
        "(x, y)": {
            "site_symmetry": "1",
            "equivalent_pos": (
                "(-y, x-y)",
                "(-x+y, -x)",
                "(y, x)",
                "(x-y, -y)",
                "(-x, -x+y)",
            ),
        },
        "(x, 0)": {
            "site_symmetry": "..m",
            "equivalent_pos": (
                "(0, x)",
                "(-x, -x)",
            ),
        },
        "(2/3, 1/3)": {
            "site_symmetry": "3..",
            "equivalent_pos": ("(1/3, 2/3)",),
        },
        "(0, 0)": {
            "site_symmetry": "3.m",
            "equivalent_pos": tuple(),
        },
    },
    16: {
        "(x, y)": {
            "site_symmetry": "1",
            "equivalent_pos": (
                "(-y, x-y)",
                "(-x+y, -x)",
                "(-x, -y)",
                "(y, -x+y)",
                "(x-y, x)",
            ),
        },
        "(1/2, 0)": {
            "site_symmetry": "2..",
            "equivalent_pos": (
                "(0, 1/2)",
                "(1/2, 1/2)",
            ),
        },
        "(2/3, 1/3)": {
            "site_symmetry": "3..",
            "equivalent_pos": ("(1/3, 2/3)",),
        },
        "(0, 0)": {
            "site_symmetry": "6..",
            "equivalent_pos": tuple(),
        },
    },
    17: {
        "(x, y)": {
            "site_symmetry": "1",
            "equivalent_pos": (
                "(-y,x-y)",
                "(-x+y,-x)",
                "(-x,-y)",
                "(y,-x+y)",
                "(x-y,x)",
                "(-y,-x)",
                "(-x+y,y)",
                "(x,x-y)",
                "(y,x)",
                "(x-y,-y)",
                "(-x,-x+y)",
            ),
        },
        "(x, -x)": {
            "site_symmetry": ".m.",
            "equivalent_pos": (
                "(x, 2*x)",
                "(-2*x, -x)",
                "(-x, x)",
                "(-x, -2*x)",
                "(2*x, x)",
            ),
        },
        "(x, 0)": {
            "site_symmetry": "..m",
            "equivalent_pos": (
                "(0, x)",
                "(-x, -x)",
                "(-x, 0)",
                "(0, -x)",
                "(x, x)",
            ),
        },
        "(1/2, 0)": {
            "site_symmetry": "2mm",
            "equivalent_pos": (
                "(0, 1/2)",
                "(1/2, 1/2)",
            ),
        },
        "(2/3, 1/3)": {
            "site_symmetry": "3m.",
            "equivalent_pos": ("(1/3, 2/3)",),
        },
        "(0, 0)": {
            "site_symmetry": "6mm",
            "equivalent_pos": tuple(),
        },
    },
}

# add identity affine transform to all groups
for plane_group in GENERATORS.values():
    plane_group.append(([[1, 0, 0], [0, 1, 0]]))

for plane_group, symm_ops in GENERATORS.items():
    GENERATORS[plane_group] = [
        np.vstack([np.array(g), [0, 0, 1]]).astype(np.float64) for g in symm_ops
    ]
