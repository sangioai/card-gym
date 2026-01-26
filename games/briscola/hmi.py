
## Rendering
import pygame
import numpy as np
from typing import Tuple, Optional
from .logic import *


pygame.init()

##################################
# Constants
##################################

# --- Screen constants -------------------------------------

SCREEN_WIDTH = 400
SCREEN_HEIGHT = 400
CARD_WIDTH = 60
CARD_HEIGHT = 90
CARD_MARGIN = 12

GREEN = (0, 120, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (180, 0, 0)
GOLD = (220, 180, 0)

# --- Symbols ----------------------------------------------

SUIT_SYMBOLS = {
    Suit.COINS: "♦",
    Suit.CUPS: "♥",
    Suit.SWORDS: "♠",
    Suit.CLUBS: "♣",
}

RANK_SYMBOLS = {
    1: "1",
    3: "3",
    10: "10",
    9: "9",
    8: "8",
    7: "7",
    6: "6",
    5: "5",
    4: "4",
    2: "2",
}

##################################
# Drawing Primitives
##################################

def rank_to_str(rank: int) -> str:
    return RANK_SYMBOLS[rank]

# --- Card drawers -----------------------------------

def draw_card(
    screen: pygame.Surface,
    card: Card,
    pos: Tuple[int, int],
    font: pygame.font.Font,
    highlight: bool = False,
):
    if card is None: return
    x, y = pos

    bg_color = GOLD if highlight else WHITE

    pygame.draw.rect(
        screen,
        bg_color,
        (x, y, CARD_WIDTH, CARD_HEIGHT),
        border_radius=6,
    )
    pygame.draw.rect(
        screen,
        BLACK,
        (x, y, CARD_WIDTH, CARD_HEIGHT),
        2,
        border_radius=6,
    )

    color = RED if card.suit in (Suit.CUPS, Suit.COINS) else BLACK

    rank_text = font.render(rank_to_str(card.rank), True, color)
    suit_text = font.render(SUIT_SYMBOLS[card.suit], True, color)

    screen.blit(rank_text, (x + 6, y + 6))
    screen.blit(suit_text, (x + 6, y + 32))

# --- Hand Drawer -------------------------------------

def draw_hand(
    screen: pygame.Surface,
    hand: list[Card],
    y: int,
    font: pygame.font.Font,
    hide: bool = False,
):
    total_width = len(hand) * (CARD_WIDTH + CARD_MARGIN)
    x = (SCREEN_WIDTH - total_width) // 2

    for i, card in enumerate(hand):
        if hide:
            pygame.draw.rect(
                screen,
                BLACK,
                (x + i * (CARD_WIDTH + CARD_MARGIN), y, CARD_WIDTH, CARD_HEIGHT),
                border_radius=6,
            )
        else:
            draw_card(
                screen,
                card,
                (x + i * (CARD_WIDTH + CARD_MARGIN), y),
                font,
            )

# --- Trick Drawer -------------------------------------

def draw_trick(
    screen: pygame.Surface,
    trick: tuple[Optional[Card], Optional[Card]],
    font: pygame.font.Font,
):
    center_x = SCREEN_WIDTH // 2 - CARD_WIDTH //2
    center_y = SCREEN_HEIGHT // 2 - CARD_HEIGHT //2

    offsets = [(-0, 0), (0, 0)]

    for card, (dx, dy) in zip(trick, offsets):
        if card:
            draw_card(
                screen,
                card,
                (center_x + dx, center_y + dy),
                font,
            )

# --- Briscola Drawer -------------------------------------

def draw_briscola(
    screen: pygame.Surface,
    card: Card,
    font: pygame.font.Font,
):
    if card is None: return
    x = SCREEN_WIDTH - CARD_WIDTH - 40
    y = SCREEN_HEIGHT // 2 - CARD_HEIGHT // 2

    # label = font.render("Briscola", True, WHITE)
    # screen.blit(label, (x - 10, y - 30))

    draw_card(screen, card, (x, y), font, highlight=True)

# --- Score Drawer -------------------------------------

def draw_scores(
    screen: pygame.Surface,
    scores: list[int],
    font: pygame.font.Font,
):
    s0 = font.render(f"Player 0: {scores[0]}", True, WHITE)
    s1 = font.render(f"Player 1: {scores[1]}", True, WHITE)

    screen.blit(s0, (30, SCREEN_HEIGHT - 35))
    screen.blit(s1, (30, 10))

# --- Deck Drawer -------------------------------------

def draw_deck(
    screen: pygame.Surface,
    deck_size: int,
    font: pygame.font.Font,
):
    x = 40
    y = SCREEN_HEIGHT // 2 - CARD_HEIGHT // 2

    # Draw card back
    # dimmer card graphics if deck is empty
    if deck_size == 0:
        pygame.draw.rect(
            screen,
            (60, 60, 60),
            (x, y, CARD_WIDTH, CARD_HEIGHT),
            border_radius=6,
        )
    else:
        pygame.draw.rect(
            screen,
            BLACK,
            (x, y, CARD_WIDTH, CARD_HEIGHT),
            border_radius=6,
        )
    pygame.draw.rect(
        screen,
        WHITE,
        (x, y, CARD_WIDTH, CARD_HEIGHT),
        2,
        border_radius=6,
    )
    
    # Card count
    count_text = font.render(str(deck_size), True, WHITE)
    text_rect = count_text.get_rect(center=(x + CARD_WIDTH // 2, y + CARD_HEIGHT // 2))
    screen.blit(count_text, text_rect)

##################################
# Main render fuction
##################################

# --- Main render loop -------------------------------------

def render_game(
    game: BriscolaGame,
    trick: tuple[Optional[Card], Optional[Card]] = (None, None),
    render: bool = True,
    return_rgb: bool = False,
):
    """
    If render=True:
        Opens a pygame window and runs the render loop.
    If render=False:
        Renders a single frame offscreen and returns an RGB image if requested.

    Returns:
        np.ndarray (H, W, 3) if return_rgb=True, else None
    """
    if game.is_terminal():
        pygame.quit()
        return
    if render:
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Briscola")
    else:
        # Offscreen surface (no window)
        screen = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))

    font = pygame.font.SysFont("arial", 24)
    clock = pygame.time.Clock()

    running = True
    rgb_image = None

    if render:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
    else:
        # Render only one frame in headless mode
        running = False

    screen.fill(GREEN)

    # Hands
    draw_hand(
        screen,
        game.players[0].hand,
        y=SCREEN_HEIGHT - CARD_HEIGHT - 40,
        font=font,
        hide=False,
    )
    draw_hand(
        screen,
        game.players[1].hand,
        y=40,
        font=font,
        hide=True,
    )

    # Deck
    draw_deck(screen, len(game.deck.cards), font=font)

    # Center trick
    draw_trick(screen, trick, font)

    # Briscola card
    draw_briscola(screen, game.briscola_card, font)

    # Scores
    draw_scores(screen, [p.score for p in game.players], font)

    if render:
        pygame.display.flip()
    elif return_rgb:
        # Convert surface to RGB array (W, H, 3) → (H, W, 3)
        rgb_image = pygame.surfarray.array3d(screen)
        rgb_image = np.transpose(rgb_image, (1, 0, 2))
    # if render:
    #     pygame.quit()
    return rgb_image


##################################
# Main
##################################

if __name__ == "__main__":
    import PIL.Image
    import random
    # init game
    game = BriscolaGame()
    # do one step
    a0 = random.randrange(len(game.players[game.current_player].hand))
    a1 = random.randrange(len(game.players[1 - game.current_player].hand))
    # get trick cards
    info = game.step(a0, a1)
    trick = info["cards"]
    # render game
    rgb = render_game(game, trick=trick, render=False, return_rgb=True) 
    # draw image
    PIL.Image.fromarray(rgb)