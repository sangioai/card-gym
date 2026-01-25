import random
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional



##################################
# Constants
##################################

# --- Suit ------------------------------------------------
class Suit(Enum):
    COINS = 0
    CUPS = 1
    SWORDS = 2
    CLUBS = 3

# --- Ranks ------------------------------------------------
BRISCOLA_RANKS = {
    1:  (10, 11),  # Ace
    3:  (9, 10),
    10: (8, 4),
    9:  (7, 3),
    8:  (6, 2),
    7:  (5, 0),
    6:  (4, 0),
    5:  (3, 0),
    4:  (2, 0),
    2:  (1, 0),
}

# --- Stringify constants ------------------------------------------------

DEFAULT_GAME_STRINGIFY = lambda scores, hands, trick, briscola, deck : (
        # f"SCORES:{scores} "
        f"""{f"HAND0:" if len(hands)!=1 else "HAND:"}""" + f"[{','.join(hands[0])}] "
        f"""{f"HAND1:[{','.join(hands[1])}] " if len(hands)!=1 else ""}"""
        f"TRICK:{{{','.join(trick)}}} "
        f"BRISCOLA:{briscola} "
        # f"DECK:{deck}"
    )

# Map suits to short codes
SUIT_CODE = {
    Suit.COINS: "C",
    Suit.CUPS: "U",
    Suit.SWORDS: "S",
    Suit.CLUBS: "B",
}

##################################
# Useful Classes
##################################

# --- Card ------------------------------------------------

@dataclass(frozen=True)
class Card:
    rank: int
    suit: Suit

    @property
    def strength(self) -> int:
        return BRISCOLA_RANKS[self.rank][0]

    @property
    def points(self) -> int:
        return BRISCOLA_RANKS[self.rank][1]
    
# --- Deck ------------------------------------------------

class Deck:
    def __init__(self):
        self.cards = [
            Card(rank, suit)
            for suit in Suit
            for rank in BRISCOLA_RANKS.keys()
        ]
        random.shuffle(self.cards)

    def draw(self) -> Optional[Card]:
        return self.cards.pop() if self.cards else None

# --- Player ------------------------------------------------

class Player:
    def __init__(self):
        self.hand: List[Card] = []
        self.score = 0

    def play_card(self, index: int) -> Card:
        return self.hand.pop(index%len(self.hand)) # avoid bad actions

##################################
# Utility functions
##################################

# --- Winning card algo ------------------------------------------------

def winning_card(card1: Card, card2: Card, briscola: Suit) -> int:
    """
    Returns 0 if card1 wins, 1 if card2 wins
    """
    # Trump beats non-trump
    if card1.suit == briscola and card2.suit != briscola:
        return 0
    if card2.suit == briscola and card1.suit != briscola:
        return 1

    # Same suit → compare strength
    if card1.suit == card2.suit:
        return 0 if card1.strength > card2.strength else 1

    # Different suits, no trump → first card wins
    return 0

# --- Stringify Functions ------------------------------------------------

def card_to_code(card: Card) -> str:
    """Convert a Card object to a short canonical code like '7C'."""
    return f"{card.rank}{SUIT_CODE[card.suit]}"


##################################
# Briscola Class
##################################

class BriscolaGame:
    def __init__(self):
        self.deck = Deck()
        self.players = [Player(), Player()]
        # Deal 3 cards each
        for _ in range(3):
            for p in self.players:
                p.hand.append(self.deck.draw())
        # Last card defines briscola
        self.briscola_card = self.deck.draw()
        self.briscola_suit = self.briscola_card.suit
        # First player index
        self.current_player = 0
        # Sequential trick variable
        self.trick = []

    def sequential_step(self, action: int):
        """
        Sequential trick.
        Actions = index of card in hand. First winning player hand, then opponent.
        """
        # reset trick if needed
        self.trick = [] if len(self.trick) >=2 else self.trick
        # get players
        p0 = self.players[self.current_player]
        p1 = self.players[1 - self.current_player]
        # play card
        self.trick += [p0.play_card(action)] if len(self.trick)==0 else [p1.play_card(action)]
        # partial trick
        if len(self.trick) < 2:
            return {
                "cards": tuple(self.trick),
                "winner": None,
                "points": 0,
                "scores": [p.score for p in self.players],
            }
        return self._step(*self.trick)


    def step(self, action_p0: int, action_p1: int):
        """
        One full trick.
        Actions = index of card in hand.
        """
        # get players
        p0 = self.players[self.current_player]
        p1 = self.players[1 - self.current_player]
        # get cards
        c0 = p0.play_card(action_p0)
        c1 = p1.play_card(action_p1)
        # update trick
        self.trick = [c0, c1]
        # perform step
        return self._step(*self.trick)

    def _step(self, c0: Card, c1: Card):
        # get winner
        winner = winning_card(c0, c1, self.briscola_suit)
        winner_player = self.current_player if winner == 0 else 1 - self.current_player
        # update score
        points = c0.points + c1.points
        self.players[winner_player].score += points
        # Draw cards (winner first)
        for idx in [winner_player, 1 - winner_player]:
            card = self.deck.draw()
            # draw briscola card as last one
            card = self.briscola_card if not card and self.briscola_card else card
            # remove briscola_card if taken
            self.briscola_card = None if card == self.briscola_card  else self.briscola_card
            # append card if present
            self.players[idx].hand += [card] if card else []
        # update current_player
        self.current_player = winner_player
        # reset trick
        self.trick = []
        # return state
        return {
            "cards": (c0, c1),
            "winner": winner_player,
            "points": points,
            "scores": [p.score for p in self.players],
        }

    def is_terminal(self) -> bool:
        # terminates only when all player has their hands free
        return all(len(p.hand) == 0 for p in self.players)
    
    def __str__(self):
        # get current player
        current_player = self.current_player
        # Scores: always player0 first, then player1
        scores = f"{self.players[current_player].score}-{self.players[1-current_player].score}"
        
        # Hands: sorted canonical codes
        hand0 = sorted([card_to_code(c) for c in self.players[current_player].hand])
        hands = [hand0]
        
        # Last trick: sorted canonical codes if given
        trick_codes = sorted([card_to_code(c) for c in self.trick]) if self.trick else []
        
        # Briscola suit code
        briscola = SUIT_CODE[self.briscola_suit]
        
        # Combine into deterministic string
        return DEFAULT_GAME_STRINGIFY(scores, hands, trick_codes, briscola, len(self.deck.cards))
        
    

##################################
# Main
##################################

if __name__ == "__main__":
    ########### normal play
    print("Playing the 2 cards at the same time.")
    game = BriscolaGame()
    while not game.is_terminal():   
        # get cards from players hands
        a0 = random.randrange(len(game.players[game.current_player].hand))
        a1 = random.randrange(len(game.players[1 - game.current_player].hand))
        # play trick
        info = game.step(a0, a1)
        # print game state
        print(str(game))
    scores = [p.score for p in game.players]
    print("Final scores:", [p.score for p in game.players])

    ########### sequential play
    print("Playing the 1 card at the same time.")
    game = BriscolaGame()
    turn = 0
    while not game.is_terminal():   
        # get player by turn
        p = game.current_player if turn%2==0 else (1-game.current_player)
        # get random actions from player hand
        a0 = random.randrange(len(game.players[p].hand))
        # play partial trick
        info = game.sequential_step(a0)
        # print game state  
        print(str(game))
        turn +=1
    scores = [p.score for p in game.players]
    print("Final scores:", [p.score for p in game.players])