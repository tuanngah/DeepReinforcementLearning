import gym
from gym import spaces
import numpy as np
import pygame
from collections import deque
import cv2
import random

# ————— Reward parameters —————
GAME_OVER_PENALTY  = 10    
SURVIVAL_BONUS     = 1     
ALPHA              = 0.2   
BETA               = 1.0   

# Scoring cho line clears 
LINE_SCORES = {
    1: 100,
    2: 300,
    3: 500,
    4: 800
}

# Tetromino definitions 
TETROMINOS = {
    'I': [
        [(0,1),(1,1),(2,1),(3,1)],
        [(2,0),(2,1),(2,2),(2,3)]
    ],
    'O': [
        [(1,0),(2,0),(1,1),(2,1)]
    ],
    'T': [
        [(1,0),(0,1),(1,1),(2,1)],
        [(1,0),(1,1),(2,1),(1,2)],
        [(0,1),(1,1),(2,1),(1,2)],
        [(1,0),(0,1),(1,1),(1,2)]
    ],
    'S': [
        [(1,0),(2,0),(0,1),(1,1)],
        [(1,0),(1,1),(2,1),(2,2)]
    ],
    'Z': [
        [(0,0),(1,0),(1,1),(2,1)],
        [(2,0),(1,1),(2,1),(1,2)]
    ],
    'J': [
        [(0,0),(0,1),(1,1),(2,1)],
        [(1,0),(2,0),(1,1),(1,2)],
        [(0,1),(1,1),(2,1),(2,2)],
        [(1,0),(1,1),(0,2),(1,2)]
    ],
    'L': [
        [(2,0),(0,1),(1,1),(2,1)],
        [(1,0),(1,1),(1,2),(2,2)],
        [(0,1),(1,1),(2,1),(0,2)],
        [(0,0),(1,0),(1,1),(1,2)]
    ]
}

class TetrisEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self):
        super(TetrisEnv, self).__init__()
        self.board_width  = 10
        self.board_height = 20
        self.block_size   = 30

        self.action_space      = spaces.Discrete(6)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(4,84,84), dtype=np.uint8
        )

        pygame.init()
        self.screen = pygame.Surface((
            self.board_width * self.block_size,
            self.board_height * self.block_size
        ))

        self.frames = deque(maxlen=4)
        self.reset()

    def reset(self):
        self.board      = np.zeros((self.board_height, self.board_width), dtype=int)
        self.score      = 0
        self.game_over  = False
        self.hold_piece = None
        self.hold_used  = False
        self.next_piece = self._new_piece()
        self._spawn_piece()

        initial = self._get_frame()
        self.frames.clear()
        for _ in range(4):
            self.frames.append(initial)
        return np.stack(self.frames, axis=0)

    def step(self, action):
        reward = 0
        done   = False

        if not self.game_over:
            x, y = self.current_pos

            # 1) apply action (move, rotate, hold)
            if   action == 0 and self._valid(self.current_shape, (x-1,y)):
                self.current_pos[0] -= 1
            elif action == 1 and self._valid(self.current_shape, (x+1,y)):
                self.current_pos[0] += 1
            elif action == 2:
                nr = (self.current_rotation+1) % len(self.current_rotations)
                ns = self.current_rotations[nr]
                if self._valid(ns, (x,y)):
                    self.current_rotation, self.current_shape = nr, ns
            elif action == 5 and not self.hold_used:
                held = self.hold_piece
                self.hold_piece = self.current_piece
                self._spawn_piece(new_piece=held)
                self.hold_used = True

            # 2) gravity / soft-drop / hard-drop logic
            if action == 3:  # soft drop
                if self._valid(self.current_shape, (x,y+1)):
                    self.current_pos[1] += 1
                else:
                    reward = self._lock()
            elif action == 4:  # hard drop
                while self._valid(self.current_shape, (x,y+1)):
                    y += 1
                self.current_pos[1] = y
                reward = self._lock()
            else:  # gravity
                if self._valid(self.current_shape, (x,y+1)):
                    self.current_pos[1] += 1
                else:
                    reward = self._lock()

        # 3) penalty game over
        if self.game_over:
            done   = True
            reward -= GAME_OVER_PENALTY

        # 4) build next observation
        obs = self._get_frame()
        self.frames.append(obs)
        return np.stack(self.frames, axis=0), reward, done, {}

    def _lock(self):
        # place piece
        for dx, dy in self.current_shape:
            self.board[self.current_pos[1]+dy,
                       self.current_pos[0]+dx] = 1

        # clear lines
        cleared   = 0
        new_board = []
        for r in range(self.board_height):
            if all(self.board[r,:]):
                cleared += 1
            else:
                new_board.append(self.board[r,:])
        while len(new_board) < self.board_height:
            new_board.insert(0, np.zeros(self.board_width, dtype=int))
        self.board = np.array(new_board)

        # spawn next
        self._spawn_piece()

        # compute reward: line-score + survival bonus
        delta_score = LINE_SCORES.get(cleared, 0)
        self.score += delta_score
        reward = delta_score + SURVIVAL_BONUS

        # count holes above/below mid
        mid = self.board_height // 2
        light_holes = dead_holes = 0
        for c in range(self.board_width):
            seen = False
            for r in range(self.board_height):
                if self.board[r,c] == 1:
                    seen = True
                elif seen:
                    if r < mid:
                        light_holes += 1
                    else:
                        dead_holes += 1

        # subtract hole penalties
        reward -= ALPHA * light_holes
        reward -= BETA  * dead_holes

        return reward

    def _would_lock(self, action):
        x, y   = self.current_pos
        shape  = self.current_shape

        # simulate action
        if action == 0:
            pos = [x-1, y]
        elif action == 1:
            pos = [x+1, y]
        elif action == 2:
            nr    = (self.current_rotation+1) % len(self.current_rotations)
            shape = self.current_rotations[nr]
            pos   = [x, y]
        else:
            pos   = [x, y+1]

        # gravity for non-drop
        if action not in (3,4):
            pos[1] += 1

        return not self._valid(shape, pos)

    # ————— Helpers —————

    def _new_piece(self):
        key = random.choice(list(TETROMINOS.keys()))
        return {'key':key, 'rotations':TETROMINOS[key]}

    def _spawn_piece(self, new_piece=None):
        if new_piece:
            self.current_piece = new_piece
        else:
            self.current_piece = self.next_piece
            self.next_piece  = self._new_piece()
        self.current_rotations = self.current_piece['rotations']
        self.current_rotation  = 0
        self.current_shape     = self.current_rotations[0]
        self.current_pos       = [self.board_width//2-2, 0]
        self.hold_used         = False
        if not self._valid(self.current_shape, self.current_pos):
            self.game_over = True

    def _valid(self, shape, pos):
        xo, yo = pos
        for dx, dy in shape:
            x, y = xo+dx, yo+dy
            if x<0 or x>=self.board_width or y<0 or y>=self.board_height or self.board[y,x]:
                return False
        return True

    def render(self, mode='human', **kwargs):
        self.screen.fill((0,0,0))
        # draw locked blocks
        for r in range(self.board_height):
            for c in range(self.board_width):
                if self.board[r,c]:
                    pygame.draw.rect(self.screen, (200,200,200),
                                     (c*self.block_size, r*self.block_size,
                                      self.block_size, self.block_size))
        # draw current piece
        for dx, dy in self.current_shape:
            px = (self.current_pos[0]+dx)*self.block_size
            py = (self.current_pos[1]+dy)*self.block_size
            pygame.draw.rect(self.screen, (100,200,100),
                             (px,py,self.block_size,self.block_size))

        if mode=='rgb_array':
            arr = pygame.surfarray.array3d(self.screen)
            return np.transpose(arr,(1,0,2))
        else:
            if not hasattr(self,'window'):
                pygame.display.init()
                self.window = pygame.display.set_mode((
                    self.board_width*self.block_size,
                    self.board_height*self.block_size))
            self.window.blit(self.screen,(0,0))
            pygame.display.flip()

    def _get_frame(self):
        arr  = self.render(mode='rgb_array')
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        return cv2.resize(gray, (84,84), interpolation=cv2.INTER_AREA)

    def close(self):
        pygame.quit()
