import numpy as np
import asyncio
import random
import time
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import json
import logging
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import joblib
import io
from playwright.async_api import Page, ElementHandle

logger = logging.getLogger(__name__)

class ActionType(Enum):
    """Types of user actions"""
    CLICK = "click"
    TYPE = "type"
    SCROLL = "scroll"
    MOUSE_MOVE = "mouse_move"
    WAIT = "wait"
    HOVER = "hover"
    FOCUS = "focus"
    BLUR = "blur"

@dataclass
class UserAction:
    """Represents a user action"""
    action_type: ActionType
    timestamp: float
    duration: float
    position: Optional[Tuple[float, float]] = None
    value: Optional[str] = None
    element_type: Optional[str] = None
    metadata: Dict[str, Any] = None

class MLBehaviorEngine:
    """Machine Learning-based behavior modeling for realistic user simulation"""

    def __init__(self):
        self.action_sequences = []
        self.timing_models = {}
        self.movement_models = {}
        self.typing_models = {}
        self.interaction_patterns = {}
        self.current_session = []
        self.model_loaded = False

        # Initialize models
        self._initialize_models()

    def _initialize_models(self):
        """Initialize ML models for behavior simulation"""
        # Timing models for different actions
        self.timing_models = {
            'click_delay': GaussianMixture(n_components=3, random_state=42),
            'typing_speed': GaussianMixture(n_components=4, random_state=42),
            'scroll_speed': GaussianMixture(n_components=2, random_state=42),
            'think_time': GaussianMixture(n_components=3, random_state=42),
            'hover_duration': GaussianMixture(n_components=2, random_state=42),
        }

        # Pre-train models with realistic data
        self._train_timing_models()

        # Mouse movement patterns
        self._initialize_movement_patterns()

        # Typing patterns
        self._initialize_typing_patterns()

        self.model_loaded = True

    def _train_timing_models(self):
        """Train timing models with realistic human behavior data"""
        # Click delays (in seconds)
        click_delays = np.concatenate([
            np.random.normal(0.15, 0.05, 1000),  # Fast clicks
            np.random.normal(0.5, 0.15, 800),    # Normal clicks
            np.random.normal(1.2, 0.3, 200),     # Slow/careful clicks
        ]).reshape(-1, 1)
        self.timing_models['click_delay'].fit(click_delays)

        # Typing speeds (characters per second)
        typing_speeds = np.concatenate([
            np.random.normal(2, 0.5, 200),    # Hunt and peck
            np.random.normal(4, 0.8, 600),    # Average typist
            np.random.normal(6, 1, 150),      # Fast typist
            np.random.normal(8, 1.5, 50),     # Professional typist
        ]).reshape(-1, 1)
        self.timing_models['typing_speed'].fit(typing_speeds)

        # Scroll speeds (pixels per second)
        scroll_speeds = np.concatenate([
            np.random.normal(500, 150, 700),   # Slow scrolling
            np.random.normal(1500, 300, 300),  # Fast scrolling
        ]).reshape(-1, 1)
        self.timing_models['scroll_speed'].fit(scroll_speeds)

        # Think time between actions (seconds)
        think_times = np.concatenate([
            np.random.normal(0.8, 0.3, 500),   # Quick decisions
            np.random.normal(2.5, 0.8, 400),   # Normal thinking
            np.random.normal(5, 1.5, 100),     # Careful consideration
        ]).reshape(-1, 1)
        self.timing_models['think_time'].fit(think_times)

        # Hover durations (seconds)
        hover_durations = np.concatenate([
            np.random.normal(0.1, 0.05, 700),  # Quick hover
            np.random.normal(0.5, 0.2, 300),   # Reading hover
        ]).reshape(-1, 1)
        self.timing_models['hover_duration'].fit(hover_durations)

    def _initialize_movement_patterns(self):
        """Initialize realistic mouse movement patterns"""
        self.movement_models = {
            'bezier_curves': self._generate_bezier_parameters(),
            'acceleration_profiles': self._generate_acceleration_profiles(),
            'jitter_patterns': self._generate_jitter_patterns(),
            'overshoot_probability': 0.15,  # 15% chance of overshooting target
            'correction_speed': 0.8,         # Speed multiplier for corrections
        }

    def _initialize_typing_patterns(self):
        """Initialize realistic typing patterns"""
        self.typing_models = {
            'common_bigrams': self._load_common_bigrams(),
            'typing_rhythms': self._generate_typing_rhythms(),
            'error_probability': 0.02,  # 2% typo rate
            'correction_probability': 0.8,  # 80% chance of correcting typos
            'pause_patterns': self._generate_pause_patterns(),
        }

    def _generate_bezier_parameters(self) -> Dict:
        """Generate parameters for bezier curve mouse movements"""
        return {
            'control_point_variance': 0.3,
            'curve_segments': 20,
            'time_distribution': 'ease-in-out',
        }

    def _generate_acceleration_profiles(self) -> List[Dict]:
        """Generate realistic mouse acceleration profiles"""
        return [
            {'name': 'quick_precise', 'accel': 2.5, 'decel': 3.0, 'max_speed': 1500},
            {'name': 'normal', 'accel': 1.8, 'decel': 2.2, 'max_speed': 1000},
            {'name': 'slow_careful', 'accel': 1.2, 'decel': 1.5, 'max_speed': 600},
        ]

    def _generate_jitter_patterns(self) -> Dict:
        """Generate mouse jitter patterns"""
        return {
            'amplitude': np.random.uniform(0.5, 2.0),
            'frequency': np.random.uniform(10, 30),
            'decay': 0.95,
        }

    def _load_common_bigrams(self) -> Dict[str, float]:
        """Load common typing bigrams and their speeds"""
        # Common English bigrams with relative typing speeds
        return {
            'th': 1.2, 'he': 1.2, 'in': 1.1, 'er': 1.1, 'an': 1.1,
            'ed': 1.0, 'nd': 1.0, 'to': 1.1, 'en': 1.0, 'ou': 0.9,
            'ng': 0.9, 'ha': 1.0, 'de': 1.0, 'or': 1.0, 'it': 1.1,
            'is': 1.1, 'ar': 1.0, 'at': 1.1, 'on': 1.0, 'as': 1.0,
        }

    def _generate_typing_rhythms(self) -> List[Dict]:
        """Generate typing rhythm patterns"""
        return [
            {'name': 'steady', 'variance': 0.1, 'burst_probability': 0.05},
            {'name': 'burst', 'variance': 0.3, 'burst_probability': 0.3},
            {'name': 'irregular', 'variance': 0.5, 'burst_probability': 0.15},
        ]

    def _generate_pause_patterns(self) -> Dict:
        """Generate pause patterns in typing"""
        return {
            'word_end': 0.15,      # Probability of pause at word end
            'punctuation': 0.4,    # Probability of pause after punctuation
            'thinking': 0.05,      # Probability of thinking pause
            'correction': 0.7,     # Probability of pause before correction
        }

    async def simulate_human_click(self, page: Page, element: ElementHandle):
        """Simulate human-like click behavior"""
        # Get element bounding box
        box = await element.bounding_box()
        if not box:
            await element.click()
            return

        # Calculate target position with slight randomness
        target_x = box['x'] + box['width'] * (0.3 + random.random() * 0.4)
        target_y = box['y'] + box['height'] * (0.3 + random.random() * 0.4)

        # Get current mouse position (simulate from last known)
        current_x = random.randint(0, 1920)
        current_y = random.randint(0, 1080)

        # Generate human-like mouse movement
        await self._move_mouse_human(page, current_x, current_y, target_x, target_y)

        # Pre-click hover
        hover_duration = self.timing_models['hover_duration'].sample(1)[0][0]
        await asyncio.sleep(max(0.05, hover_duration))

        # Click with timing variation
        click_delay = self.timing_models['click_delay'].sample(1)[0][0]
        await asyncio.sleep(max(0.05, click_delay))

        # Perform click
        await page.mouse.click(target_x, target_y)

        # Post-click delay
        post_delay = random.uniform(0.05, 0.2)
        await asyncio.sleep(post_delay)

        # Record action
        self._record_action(UserAction(
            action_type=ActionType.CLICK,
            timestamp=time.time(),
            duration=hover_duration + click_delay + post_delay,
            position=(target_x, target_y),
            element_type=await self._get_element_type(element)
        ))

    async def simulate_human_typing(self, page: Page, element: ElementHandle, text: str):
        """Simulate human-like typing behavior"""
        # Focus element
        await element.focus()
        await asyncio.sleep(random.uniform(0.1, 0.3))

        # Get typing speed for this session
        base_speed = self.timing_models['typing_speed'].sample(1)[0][0]
        rhythm = random.choice(self.typing_models['typing_rhythms'])

        # Type each character
        typed_text = ""
        i = 0

        while i < len(text):
            char = text[i]

            # Check for bigram speed adjustment
            if i > 0:
                bigram = text[i-1:i+1].lower()
                speed_mult = self.typing_models['common_bigrams'].get(bigram, 1.0)
            else:
                speed_mult = 1.0

            # Calculate delay
            char_delay = 1.0 / (base_speed * speed_mult)
            char_delay *= random.uniform(1 - rhythm['variance'], 1 + rhythm['variance'])

            # Simulate typo
            if random.random() < self.typing_models['error_probability']:
                # Type wrong character
                wrong_char = random.choice('abcdefghijklmnopqrstuvwxyz')
                await element.type(wrong_char)
                typed_text += wrong_char
                await asyncio.sleep(char_delay)

                # Maybe correct it
                if random.random() < self.typing_models['correction_probability']:
                    await asyncio.sleep(random.uniform(0.2, 0.5))
                    await page.keyboard.press('Backspace')
                    typed_text = typed_text[:-1]
                    await asyncio.sleep(random.uniform(0.1, 0.3))

            # Type correct character
            await element.type(char)
            typed_text += char
            await asyncio.sleep(char_delay)

            # Check for pauses
            if char == ' ' and random.random() < self.typing_models['pause_patterns']['word_end']:
                await asyncio.sleep(random.uniform(0.1, 0.4))
            elif char in '.!?,;:' and random.random() < self.typing_models['pause_patterns']['punctuation']:
                await asyncio.sleep(random.uniform(0.3, 0.8))
            elif random.random() < self.typing_models['pause_patterns']['thinking']:
                await asyncio.sleep(random.uniform(0.5, 2.0))

            # Burst typing
            if random.random() < rhythm['burst_probability']:
                # Type next 2-5 characters quickly
                burst_length = min(random.randint(2, 5), len(text) - i - 1)
                for _ in range(burst_length):
                    i += 1
                    if i < len(text):
                        await element.type(text[i])
                        typed_text += text[i]
                        await asyncio.sleep(char_delay * 0.7)

            i += 1

        # Record action
        self._record_action(UserAction(
            action_type=ActionType.TYPE,
            timestamp=time.time(),
            duration=time.time() - time.time(),
            value=text,
            element_type=await self._get_element_type(element)
        ))

    async def simulate_human_scroll(self, page: Page, direction: str = 'down', distance: int = None):
        """Simulate human-like scrolling behavior"""
        # Get scroll speed
        scroll_speed = self.timing_models['scroll_speed'].sample(1)[0][0]

        # Calculate distance if not provided
        if distance is None:
            distance = random.randint(200, 600)

        # Determine scroll steps
        steps = random.randint(3, 8)
        step_distance = distance / steps

        # Perform scroll with acceleration
        for i in range(steps):
            # Acceleration curve
            if i < steps // 2:
                speed_mult = 0.5 + (i / (steps // 2)) * 0.5
            else:
                speed_mult = 1.0 - ((i - steps // 2) / (steps // 2)) * 0.5

            # Scroll step
            step_dist = step_distance * speed_mult
            if direction == 'down':
                await page.mouse.wheel(0, step_dist)
            else:
                await page.mouse.wheel(0, -step_dist)

            # Delay between steps
            delay = step_dist / scroll_speed
            await asyncio.sleep(delay)

        # Post-scroll pause
        await asyncio.sleep(random.uniform(0.1, 0.5))

        # Record action
        self._record_action(UserAction(
            action_type=ActionType.SCROLL,
            timestamp=time.time(),
            duration=distance / scroll_speed,
            value=f"{direction}:{distance}"
        ))

    async def _move_mouse_human(self, page: Page, start_x: float, start_y: float, 
                               end_x: float, end_y: float):
        """Generate human-like mouse movement using bezier curves"""
        # Select movement profile
        profile = random.choice(self.movement_models['acceleration_profiles'])

        # Calculate distance
        distance = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)

        # Generate bezier control points
        control_variance = self.movement_models['bezier_curves']['control_point_variance']

        # Control points for bezier curve
        cp1_x = start_x + (end_x - start_x) * 0.3 + random.uniform(-distance * control_variance, distance * control_variance)
        cp1_y = start_y + (end_y - start_y) * 0.3 + random.uniform(-distance * control_variance, distance * control_variance)

        cp2_x = start_x + (end_x - start_x) * 0.7 + random.uniform(-distance * control_variance, distance * control_variance)
        cp2_y = start_y + (end_y - start_y) * 0.7 + random.uniform(-distance * control_variance, distance * control_variance)

        # Check for overshoot
        overshoot = random.random() < self.movement_models['overshoot_probability']
        if overshoot:
            overshoot_distance = random.uniform(10, 30)
            angle = np.arctan2(end_y - start_y, end_x - start_x)
            end_x += np.cos(angle) * overshoot_distance
            end_y += np.sin(angle) * overshoot_distance

        # Generate movement points
        segments = self.movement_models['bezier_curves']['curve_segments']
        points = []

        for i in range(segments + 1):
            t = i / segments

            # Ease-in-out timing
            if self.movement_models['bezier_curves']['time_distribution'] == 'ease-in-out':
                t = t * t * (3 - 2 * t)

            # Cubic bezier curve
            x = (1-t)**3 * start_x + 3*(1-t)**2*t * cp1_x + 3*(1-t)*t**2 * cp2_x + t**3 * end_x
            y = (1-t)**3 * start_y + 3*(1-t)**2*t * cp1_y + 3*(1-t)*t**2 * cp2_y + t**3 * end_y

            # Add jitter
            jitter = self.movement_models['jitter_patterns']
            jitter_x = np.sin(i * jitter['frequency']) * jitter['amplitude'] * (jitter['decay'] ** i)
            jitter_y = np.cos(i * jitter['frequency']) * jitter['amplitude'] * (jitter['decay'] ** i)

            points.append((x + jitter_x, y + jitter_y))

        # If overshoot, add correction movement
        if overshoot:
            correction_points = 5
            for i in range(1, correction_points + 1):
                t = i / correction_points
                x = end_x + (end_x - end_x) * (1 - t)
                y = end_y + (end_y - end_y) * (1 - t)
                points.append((x, y))

        # Move mouse along path
        total_time = distance / profile['max_speed']
        time_per_segment = total_time / len(points)

        for i, (x, y) in enumerate(points):
            await page.mouse.move(x, y)

            # Variable speed based on acceleration profile
            if i < len(points) // 3:
                speed_mult = profile['accel']
            elif i > 2 * len(points) // 3:
                speed_mult = profile['decel']
            else:
                speed_mult = 1.0

            await asyncio.sleep(time_per_segment / speed_mult)

    async def _get_element_type(self, element: ElementHandle) -> str:
        """Get the type of element for behavior tracking"""
        try:
            tag_name = await element.evaluate('el => el.tagName.toLowerCase()')
            element_type = await element.evaluate('el => el.type || ""')
            class_name = await element.evaluate('el => el.className || ""')

            if tag_name == 'input' and element_type:
                return f"input[{element_type}]"
            elif tag_name == 'button':
                return 'button'
            elif 'btn' in class_name.lower():
                return 'button'
            else:
                return tag_name
        except:
            return 'unknown'

    def _record_action(self, action: UserAction):
        """Record user action for pattern learning"""
        self.current_session.append(action)

        # Keep only recent actions
        if len(self.current_session) > 1000:
            self.current_session = self.current_session[-500:]

    def get_next_action_delay(self) -> float:
        """Get delay before next action based on patterns"""
        think_time = self.timing_models['think_time'].sample(1)[0][0]

        # Adjust based on recent actions
        if self.current_session:
            last_action = self.current_session[-1]

            # Longer delay after complex actions
            if last_action.action_type in [ActionType.TYPE, ActionType.SCROLL]:
                think_time *= random.uniform(1.2, 1.8)
            # Shorter delay for rapid interactions
            elif len(self.current_session) > 3:
                recent_actions = self.current_session[-3:]
                if all(a.action_type == ActionType.CLICK for a in recent_actions):
                    think_time *= random.uniform(0.5, 0.8)

        return max(0.1, think_time)

    def analyze_detection_risk(self) -> float:
        """Analyze current behavior for bot detection risk"""
        if not self.current_session:
            return 0.0

        risk_score = 0.0

        # Check for repetitive patterns
        if len(self.current_session) > 10:
            recent_durations = [a.duration for a in self.current_session[-10:]]
            duration_variance = np.var(recent_durations)

            if duration_variance < 0.01:  # Too consistent
                risk_score += 0.3

        # Check for inhuman speed
        recent_typing = [a for a in self.current_session[-20:] if a.action_type == ActionType.TYPE]
        if recent_typing:
            avg_speed = np.mean([len(a.value or '') / a.duration for a in recent_typing])
            if avg_speed > 15:  # More than 15 chars/second
                risk_score += 0.4

        # Check for lack of errors
        total_typing = [a for a in self.current_session if a.action_type == ActionType.TYPE]
        if len(total_typing) > 50:
            # No backspaces or corrections in 50+ typing actions is suspicious
            risk_score += 0.2

        return min(1.0, risk_score)

    def adapt_behavior(self, detection_risk: float):
        """Adapt behavior based on detection risk"""
        if detection_risk > 0.7:
            # Increase randomness
            for model in self.timing_models.values():
                # Add noise to model predictions
                pass

            # Add more human-like errors
            self.typing_models['error_probability'] *= 1.5

            # Increase pause frequency
            for key in self.typing_models['pause_patterns']:
                self.typing_models['pause_patterns'][key] *= 1.3

    def save_patterns(self, filepath: str):
        """Save learned behavior patterns"""
        patterns = {
            'action_sequences': self.action_sequences,
            'current_session': [(a.action_type.value, a.timestamp, a.duration) 
                              for a in self.current_session[-100:]],
            'model_parameters': {
                'typing_error_rate': self.typing_models['error_probability'],
                'overshoot_probability': self.movement_models['overshoot_probability'],
            }
        }

        with open(filepath, 'w') as f:
            json.dump(patterns, f)

    def load_patterns(self, filepath: str):
        """Load behavior patterns from file"""
        try:
            with open(filepath, 'r') as f:
                patterns = json.load(f)

            # Update model parameters
            if 'model_parameters' in patterns:
                params = patterns['model_parameters']
                self.typing_models['error_probability'] = params.get('typing_error_rate', 0.02)
                self.movement_models['overshoot_probability'] = params.get('overshoot_probability', 0.15)

        except Exception as e:
            logger.error(f"Failed to load patterns: {e}")