from logic import *

AKnight = Symbol("A is a Knight")
AKnave = Symbol("A is a Knave")

BKnight = Symbol("B is a Knight")
BKnave = Symbol("B is a Knave")

CKnight = Symbol("C is a Knight")
CKnave = Symbol("C is a Knave")

knowledgeBase = And(
    Or(AKnight, AKnave),
    Not(And(AKnight, AKnave)),
    Or(BKnight, BKnave),
    Not(And(BKnight, BKnave)),
    Or(CKnight, CKnave),
    Not(And(CKnight, CKnave)),
)

# Puzzle 0
# A says "I am both a knight and a knave."
A0 = And(AKnight, AKnave)
knowledge0 = And(
    knowledgeBase,
    Implication(AKnight, A0),
    Implication(AKnave, Not(A0)),
)

# Puzzle 1
# A says "We are both knaves."
a1 = And(AKnave, BKnave)
# B says nothing.
knowledge1 = And(
    knowledgeBase,
    Implication(AKnight, a1),
    Implication(AKnave, Not(a1)),
)

# Puzzle 2
# A says "We are the same kind."
a2 = Or(And(AKnight, BKnight), And(AKnave, BKnave))
# B says "We are of different kinds."
b2 = Or(And(AKnight, BKnave), And(AKnave, BKnight))
knowledge2 = And(
    knowledgeBase,
    Implication(AKnight, a2),
    Implication(AKnave, Not(a2)),
    Implication(BKnight, b2),
    Implication(BKnave, Not(b2)),
)

# Puzzle 3
# A says either "I am a knight." or "I am a knave.", but you don't know which.
a3 = Or(AKnight, AKnave)
# B says "A said 'I am a knave'."
b3 = AKnave
# B says "C is a knave."
b33 = CKnave
# C says "A is a knight."
c3 = AKnight

knowledge3 = And(
    knowledgeBase,
    Implication(AKnight, a3),
    Implication(AKnave, Not(a3)),
    Implication(BKnight, b3),
    Implication(BKnave, Not(b3)),
    Implication(BKnight, b33),
    Implication(BKnave, Not(b33)),
    Implication(CKnight, c3),
    Implication(CKnave, Not(c3)),
)


def main():
    symbols = [AKnight, AKnave, BKnight, BKnave, CKnight, CKnave]
    puzzles = [
        ("Puzzle 0", knowledge0),
        ("Puzzle 1", knowledge1),
        ("Puzzle 2", knowledge2),
        ("Puzzle 3", knowledge3)
    ]
    for puzzle, knowledge in puzzles:
        print(puzzle)
        if len(knowledge.conjuncts) == 0:
            print("    Not yet implemented.")
        else:
            for symbol in symbols:
                if model_check(knowledge, symbol):
                    print(f"    {symbol}")


if __name__ == "__main__":
    main()
