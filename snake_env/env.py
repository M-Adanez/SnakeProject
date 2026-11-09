# Import necessary libraries
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import random
from math import fmod
import time
import scipy.special as sc

# BOdy positions to store
BODY_POSITIONS = 15

# Initialize max score
max_score = 0

# Function to handle collision with apple and update score
def collision_with_apple(apple_position, score, halfTable, snake_positions, action_size=1, apple_curriculum: bool = False, apple_dist = None):
    # New apple position must stay on the grid reachable by the snake's step size
    apple_position = random_apple_position(apple_curriculum, apple_dist, halfTable, snake_positions, action_size)
    score += 1  # Increment the score
    return apple_position, score

def valid_apple_pos(pos, snake_positions):
    for snake_pos in snake_positions:
        if snake_pos == pos:
            return False
    return True

# Generate a random apple position aligned with the snake's movement grid
def random_apple_position(apple_curriculum, apple_dist, halfTable, snake_positions, action_size=1):
    step = 10 * action_size
    max_k = (halfTable - step) // step
    head_x, head_y = snake_positions[0]

    if apple_curriculum:
        vals, probs = apple_dist

        wrap_limit = ((halfTable + step - 1) // step) * step

        choice_x = np.random.choice(vals, p = probs)
        choice_y = np.random.choice(vals, p = probs)

        new_pos = [
            head_x + int(fmod(step * choice_x, wrap_limit)),
            head_y + int(fmod(step * choice_y, wrap_limit)),
        ]

        for _ in range (64):
            choice_x = np.random.choice(vals, p = probs)
            choice_y = np.random.choice(vals, p = probs)

            new_pos = [
                        head_x + int(fmod(step * choice_x, wrap_limit)),
                        head_y + int(fmod(step * choice_y, wrap_limit)),
                    ]
            if not collision_with_boundaries(new_pos, 2 * halfTable) and valid_apple_pos(new_pos, snake_positions):
                return new_pos

        # caso de que no se encuentre ninguna posicion libre
        occupied = {tuple(pos) for pos in snake_positions}
        free_cells = [
            (x,y)
            for x in range(head_x % step, halfTable * 2, step)
            for y in range(head_y % step, halfTable * 2, step)
            if (x,y) not in occupied
        ]

        if not free_cells:
            raise RuntimeError("No queda ninguna celda libre")

        return list(random.choice(free_cells))

    else:

        new_pos = [
            halfTable + (step * random.randint(-max_k, max_k)),
            halfTable + (step * random.randint(-max_k, max_k)),
        ]

        while not valid_apple_pos(new_pos, snake_positions):
            new_pos = [
                        halfTable + (step * random.randint(-max_k, max_k)),
                        halfTable + (step * random.randint(-max_k, max_k)),
                    ]

        return new_pos



# Function to check if the snake head collides with game boundaries
def collision_with_boundaries(snake_head, tableSize):
    if snake_head[0] >= tableSize or snake_head[0] < 0 or snake_head[1] >= tableSize or snake_head[1] < 0:
        return True
    else:
        return False

# Function to check if the snake collides with itself
def collision_with_self(snake_position):
    snake_head = snake_position[0]
    if snake_head in snake_position[1:]:
        return True
    else:
        return False


# MADE WITH AI
def get_apple_distr(k: int, sd: float) -> np.ndarray:
    enteros = np.arange(-k, k + 1)
    
    def cdf_normal_centrada(x):
        return 0.5 * sc.erf((x / np.sqrt(2)) / sd)

    cdf_upper = cdf_normal_centrada(enteros + 0.5)
    cdf_lower = cdf_normal_centrada(enteros - 0.5)

    # Normaliza solo la masa dentro del intervalo de enteros.
    probabilidades = cdf_upper - cdf_lower
    probabilidades /= probabilidades.sum()
    
    return (enteros, probabilidades)


# Define custom snake game environment class inheriting from gym's Env class
class SnakeEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 150
    }
    

    

    def __init__(self, render_mode=None, action_size=1, table_size: int = 500, random_spawn = False, random_spawn_prob:float = 0, reward_decay = False, apple_curriculum = False, decay_rule = 5, render_fps=10, initial_sd = 1, apples_rule = 250):
        '''
        Utiliar reward_decay: bool para especificar si se quiere utilizar decay en el reward de cada step.
        decay_rule define cada cuantas manazas comidas se divide entre dos el reward

        Utilizar apple_curriculum: bool para especificar si se quiere utilizar entrenamiento con apple curriculum.
        initial_sd define la desviación de la distribución de aparición de las manzanas.
        steps_rule indica cada cuantas manzanas se actualiza la probabilidad de aparición cerca (el sd)
        '''
        
        super(SnakeEnv, self).__init__()
        # Define the action space: 4 discrete actions (left, right, up, down)
        self.action_space = spaces.Discrete(4)

        self.tableSizeObs = table_size # control of the grid size
        self.tableSize = table_size
        self.halfTable = int(self.tableSize / 2)
        
        # Define the observation space: a Box with size based on snake length goal and other game parameters
        self.observation_space = spaces.Box(
            low=-self.tableSizeObs,
            high=self.tableSizeObs,
            shape=(9 + 2 * BODY_POSITIONS,),
            dtype=np.float64
        )

        # Spawn the snake in a random position
        self.random_spawn_prob = random_spawn_prob
        
        # Initialize game display and reward variables
        self.img = np.zeros((self.tableSize, self.tableSize, 3), dtype='uint8')
        self.prev_reward = 0
        self.total_reward = 0
        self.score = 0
        self.max_score = 0

        # Modify apple 
        self.apple_curriculum = apple_curriculum
        self.sd = initial_sd
        self.apple_distr = get_apple_distr(self.halfTable, self.sd)
        self.apples_rule = apples_rule
        self.apples_eaten = 0
        self.can_update_sd = True

        # Reward engineering
        self.decay_rule = decay_rule
        self.decay = lambda n_apples : np.log2(2 + n_apples / self.decay_rule) if reward_decay else 1

        # rendering
        self.render_mode = render_mode
        self.render_fps = render_fps 

        # number of cells the snake moves in each step
        self.action_size = action_size

    def compute_apple_sd(self):
        self.sd += 0.5
        

    # Step function that updates the environment after an action is taken
    def step(self, action):
        if self.can_update_sd and (self.apples_eaten + 1) % self.apples_rule == 0:
            self.can_update_sd = False
            self.compute_apple_sd()
            self.apple_distr = get_apple_distr(self.halfTable, self.sd)

        # Calculate previous distance to the apple
        prev_distance = np.linalg.norm(np.array(self.snake_head) - np.array(self.apple_position))

        # Perform the action
        self._take_action(action)

        # Initialize reward
        reward = 0

        apple_eaten = False
        # Check if snake eats the apple
        if self.snake_head == self.apple_position:
            self.snake_position.insert(0, list(self.snake_head))
            self.apple_position, self.score = collision_with_apple(self.apple_position, self.score, self.halfTable, self.snake_position, self.action_size, self.apple_curriculum, self.apple_distr)
            reward += 12  # Reward for eating an apple
            self.apples_eaten += 1
            apple_eaten = True
            self.can_update_sd = True
        else:
            self.snake_position.insert(0, list(self.snake_head))
            self.snake_position.pop()

        # Check for collisions (with boundaries or self)
        terminated = False
        if collision_with_boundaries(self.snake_head, self.tableSize):
            reward -= 17
            terminated = True
        
        if collision_with_self(self.snake_position):
            reward -= 15
            terminated = True

        truncated = False

        # Handle termination (game over) scenario
        if not apple_eaten:
            # Calculate the current distance to the apple
            current_distance = np.linalg.norm(np.array(self.snake_head) - np.array(self.apple_position))

            if current_distance < prev_distance:
                reward += 1 * (1/self.decay(self.score))
            else:
                reward -= 1 * (1/self.decay(self.score))

        # TRY
        reward -= 0.3

        # Information dictionary
        info = {"apples": self.score}

        return self._observation(), reward, terminated, truncated, info

    def _observation(self):
        head_x = self.snake_head[0]
        head_y = self.snake_head[1]
        snake_length = len(self.snake_position)
        apple_delta_x = self.apple_position[0] - head_x
        apple_delta_y = self.apple_position[1] - head_y

        common_features = [
            head_x, head_y, *self._danger_flags(),
            apple_delta_x, apple_delta_y, snake_length,
        ]
        step_size = 10 * self.action_size
        body_features = [0.0] * (2 * BODY_POSITIONS)
        for i, (x, y) in enumerate(self.snake_position[1:BODY_POSITIONS + 1]):
            body_features[2 * i] = (x - head_x) / step_size
            body_features[2 * i + 1] = (y - head_y) / step_size

        return np.array(common_features + body_features, dtype=np.float64)

    # Reset the environment to the initial state
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # Reset game board
        self.img = np.zeros((self.tableSize, self.tableSize, 5), dtype='uint8')
        
        # Reset snake position, spaced according to the movement step size
        step = 10 * self.action_size
        head_x = head_y = self.halfTable
        if self.random_spawn_prob > 0 and self.np_random.random() < self.random_spawn_prob:
            grid_start = self.halfTable % step
            grid = range(grid_start, self.tableSize, step)
            # The body extends two cells to the left of the head.
            valid_columns = [
                x for x in grid
                if not collision_with_boundaries((x - 2 * step, grid_start), self.tableSize)
            ]
            if not valid_columns:
                raise ValueError("El tablero no tiene espacio para la serpiente inicial")
            head_x = valid_columns[self.np_random.integers(len(valid_columns))]
            head_y = grid[self.np_random.integers(len(grid))]

        self.snake_position = [[head_x - segment * step, head_y] for segment in range(3)]
        
        # Generate random apple position aligned with the movement grid
        self.apple_position = random_apple_position(self.apple_curriculum, self.apple_distr, self.halfTable, self.snake_position, self.action_size)

        # Update max score if needed
        if self.score > self.max_score:
            self.max_score = self.score
            print(f"New maximum score registered: {self.max_score}")
        
        # Reset score and snake head position
        self.score = 0
        self.snake_head = self.snake_position[0].copy()

        # Initialize direction of the snake
        self.direction = 1

        # Reset reward values
        self.prev_reward = 0
        self.total_reward = 0

        return self._observation(), {}

    # Render the game visually using OpenCV
    def render(self, mode=None):
        if mode is None:
            mode = self.render_mode

        self._update_ui() # Update the game UI

        if mode == 'human':
            cv2.imshow('Snake Game', self.img)
            cv2.waitKey(int(1000/self.render_fps)) # in milliseconds
            
            #time.sleep(0.1)  # Add delay between frames to slow down execution
        elif mode == 'rgb_array':
            return self.img.copy()

    # Close the OpenCV windows
    def close(self):
        cv2.destroyAllWindows()

    # Update the UI with the current positions of the snake and the apple
    def _update_ui(self):
        # Clear the game board
        self.img = np.zeros((self.tableSize, self.tableSize, 3), dtype='uint8')
        
        # Draw apple on the board
        cv2.rectangle(
            self.img,
            (self.apple_position[0], self.apple_position[1]),
            (self.apple_position[0] + 10, self.apple_position[1] + 10),
            (0, 0, 255),
            -1
        )
        
        # Draw snake on the board
        for _i,position in enumerate(self.snake_position):

            cv2.rectangle(
                self.img,
                (position[0], position[1]),
                (position[0] + 10, position[1] + 10),
                (0, 255, 255) if _i == 0 else (0, 255, 0),
                -1
            )

    def _danger_flags(self):
        offsets = ((-1, 0), (1, 0), (0, 1), (0, -1))
        opposite = (1, 0, 3, 2)
        step_size = 10 * self.action_size
        head_x, head_y = self.snake_head
        danger = []

        for action in range(4):
            effective_action = self.direction if action == opposite[self.direction] else action
            dx, dy = offsets[effective_action]
            next_head = [head_x + dx * step_size, head_y + dy * step_size]

            body = self.snake_position
            if next_head != self.apple_position:
                body = body[:-1] # if doesnt grow

            danger.append(int(
                collision_with_boundaries(next_head, self.tableSize)
                or next_head in body
            ))

        return danger

    # Handle actions and update the snake's position
    def _take_action(self, action):
        # Avoid direct opposite movements
        if action == 0 and self.direction == 1:
            action = 1
        elif action == 1 and self.direction == 0:
            action = 0
        elif action == 2 and self.direction == 3:
            action = 3
        elif action == 3 and self.direction == 2:
            action = 2

        # Update direction based on the action
        self.direction = action

        # Move the snake head according to the action
        if action == 0:  # Move left
            self.snake_head[0] -= 10*self.action_size
        elif action == 1:  # Move right
            self.snake_head[0] += 10*self.action_size
        elif action == 2:  # Move down
            self.snake_head[1] += 10*self.action_size
        elif action == 3:  # Move up
            self.snake_head[1] -= 10*self.action_size
