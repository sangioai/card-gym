# hmi_set_dynamic.py

import pygame
import numpy as np
from typing import List
from .logic import Card

pygame.init()

##################################
# CONFIG (dynamic layout)
##################################

CARD_W = 100
CARD_H = 140
MARGIN_X = 20
MARGIN_Y = 20
BACKGROUND = (0, 0, 0)          # black for contrastive clarity
CARD_BG = (245, 245, 245)
CARD_BORDER = (200, 200, 200)
COLORS = {
    "RED": (220, 40, 40),
    "GREEN": (40, 160, 70),
    "PURPLE": (120, 60, 170),
}

##################################
# SHAPES
##################################

def draw_diamond(surface, rect, color, width=0):
    cx, cy = rect.center
    w, h = rect.width // 2, rect.height // 2
    pts = [(cx, cy-h), (cx+w, cy), (cx, cy+h), (cx-w, cy)]
    pygame.draw.polygon(surface, color, pts, width)

def draw_oval(surface, rect, color, width=0):
    pygame.draw.ellipse(surface, color, rect, width)

def draw_squiggle(surface, rect, color, width=0):
    x, y, w, h = rect
    pts = [
        (x, y+h*0.4),
        (x+w*0.25, y),
        (x+w*0.75, y+h*0.2),
        (x+w, y+h*0.6),
        (x+w*0.75, y+h),
        (x+w*0.25, y+h*0.8),
    ]
    pygame.draw.polygon(surface, color, pts, width)

##################################
# CARD RENDER
##################################

def draw_card(surface, card: Card, x, y, card_w=CARD_W, card_h=CARD_H):

    pygame.draw.rect(surface, CARD_BG, (x, y, card_w, card_h), border_radius=8)
    pygame.draw.rect(surface, CARD_BORDER, (x, y, card_w, card_h), 1, border_radius=8)

    color = COLORS[card.color.name]
    count = card.number
    spacing = card_h // 3
    center_y = y + card_h // 2
    start_y = center_y - (count-1)*spacing//2

    for i in range(count):
        rect = pygame.Rect(
            x + card_w * 0.2,
            start_y + i*spacing - 12,
            card_w * 0.6,
            24
        )
        outline = 2 if card.shading.name == "EMPTY" else 0

        if card.shape.name == "OVAL":
            draw_oval(surface, rect, color, outline)
        elif card.shape.name == "DIAMOND":
            draw_diamond(surface, rect, color, outline)
        elif card.shape.name == "SQUIGGLE":
            draw_squiggle(surface, rect, color, outline)

        if card.shading.name == "STRIPED":
            stripe = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
            stripe.fill((0,0,0,0))
            shape_rect = pygame.Rect(0,0,rect.width,rect.height)
            if card.shape.name == "OVAL":
                draw_oval(stripe, shape_rect, color)
            elif card.shape.name == "DIAMOND":
                draw_diamond(stripe, shape_rect, color)
            else:
                draw_squiggle(stripe, shape_rect, color)
            stripe.set_colorkey((0,0,0))
            for sx in range(0, rect.width, 5):
                pygame.draw.line(stripe, color, (sx,0), (sx,rect.height), 1)
            surface.blit(stripe, rect.topleft)

##################################
# DYNAMIC BOARD RENDER
##################################

def render_board(board: List[Card]) -> np.ndarray:
    """
    Dynamic renderer based on number of cards.
    """

    n_cards = len(board)

    # Dynamic cols/rows (try square-ish layout)
    cols = max(int(np.ceil(np.sqrt(n_cards))),3)
    rows = int(np.ceil(n_cards / cols))

    # Adjust card spacing to fit
    width = MARGIN_X*2 + cols*CARD_W + (cols-1)*10
    height = MARGIN_Y*2 + rows*CARD_H + (rows-1)*10
    spacing_x = (width - 2*MARGIN_X - cols*CARD_W) // max(1, cols-1)
    spacing_y = (height - 2*MARGIN_Y - rows*CARD_H) // max(1, rows-1)

    surface = pygame.Surface((width, height))
    surface.fill(BACKGROUND)

    for idx, card in enumerate(board):
        col = idx % cols
        row = idx // cols

        x = MARGIN_X + col * (CARD_W + spacing_x)
        y = MARGIN_Y + row * (CARD_H + spacing_y)

        draw_card(surface, card, x, y, CARD_W, CARD_H)

    rgb = pygame.surfarray.array3d(surface)
    return np.transpose(rgb, (1, 0, 2))


if __name__ == "__main__":
    from logic import SetGame
    import PIL.Image

    game = SetGame()

    # Test with fewer cards
    for test_n in [3, 6, 12, 24, 48]:
        rgb = render_board(game.deck.cards[:test_n])
        PIL.Image.fromarray(rgb).show()