import random
from dataclasses import dataclass
from enum import Enum
from typing import List
from itertools import combinations

##################################
# Constants
##################################

class Color(Enum):
    RED = 0
    GREEN = 1
    PURPLE = 2

class Shape(Enum):
    OVAL = 0
    SQUIGGLE = 1
    DIAMOND = 2

class Shading(Enum):
    SOLID = 0
    TRIPED = 1
    EMPTY = 2

NUMBERS = [1, 2, 3]

##################################
# Card
##################################

@dataclass(frozen=True)
class Card:
    number: int
    shape: Shape
    color: Color
    shading: Shading

    def code(self) -> str:
        return f"{self.number}{self.shape.name[0]}{self.color.name[0]}{self.shading.name[0]}"

##################################
# Deck
##################################

class Deck:
    def __init__(self):
        self.cards: List[Card] = [
            Card(number, shape, color, shading)
            for number in NUMBERS
            for shape in Shape
            for color in Color
            for shading in Shading
        ]
        self.shuffle()

    def shuffle(self):
        random.shuffle(self.cards)

    def draw(self, n: int) -> List[Card]:
        drawn = self.cards[:n]
        self.cards = self.cards[n:]
        return drawn

    def return_cards(self, cards: List[Card]):
        self.cards.extend(cards)
        self.shuffle()

##################################
# Game
##################################

class SetGame:
    def __init__(self, force_sets: bool = False):
        self.deck = Deck()
        self.board: List[Card] = self.deck.draw(12)
        self.force_sets = force_sets

        if self.force_sets:
            self._ensure_set_on_board(12)

    @staticmethod
    def is_set(cards: List[Card]) -> bool:
        if len(cards) != 3:
            return False

        for attr in ['number', 'shape', 'color', 'shading']:
            values = {getattr(c, attr) for c in cards}
            if len(values) not in (1, 3):
                return False

        return True

    def find_sets_on_board(self) -> List[List[int]]:
        sets_found = []
        for combo in combinations(range(len(self.board)), 3):
            if SetGame.is_set([self.board[i] for i in combo]):
                sets_found.append(list(combo))
        return sets_found

    def _ensure_set_on_board(self, replace_count: int):
        """
        Keep replacing the last `replace_count` cards
        until at least one set exists.
        """
        while not self.find_sets_on_board():
            if len(self.deck.cards) < replace_count:
                break  # no more cards to fix it

            to_replace = self.board[-replace_count:]
            self.deck.return_cards(to_replace)
            self.board[-replace_count:] = self.deck.draw(replace_count)

    def remove_set(self, indices: List[int]):
        indices = sorted(indices, reverse=True)
        removed = [self.board.pop(i) for i in indices]

        replacement = self.deck.draw(len(indices))
        self.board.extend(replacement)

        if self.force_sets:
            self._ensure_set_on_board(len(indices))
    
    def is_terminal(self) -> bool:
        return not self.find_sets_on_board() and not self.deck.cards

    def board_state(self) -> str:
        return '[' + ','.join(c.code() for c in self.board) + ']'


##################################
# Example Play
##################################

if __name__ == "__main__":
    game = SetGame(force_sets=True)
    print("Initial board:", game.board_state())

    while not game.is_terminal():
        sets_found = game.find_sets_on_board()
        print("Sets on board:", sets_found)

        if sets_found:
            game.remove_set(sets_found[0])
            print("Board after removal:", game.board_state())
        
    sets_found = game.find_sets_on_board()
    print("Sets on board:", sets_found)
    print("Game over.")