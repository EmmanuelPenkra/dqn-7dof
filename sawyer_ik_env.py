import gymnasium as gym
import numpy as np
from gymnasium import spaces
import mujoco

class SawyerIKEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        xml_path: str = "mujoco_menagerie/rethink_robotics_sawyer/scene.xml",
        delta: float = np.deg2rad(0.5),  # 0.0025° in radians
    ):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data  = mujoco.MjData(self.model)
        self.delta = delta

        # 3^7 = 2187 discrete actions (each joint ∈ {-1,0,+1})
        self.N = 7
        self.action_space = spaces.Discrete(3**self.N)

        # observations = 7 joint angles + 3D error vector
        high = np.array([np.pi]*7 + [np.inf]*3, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.prev_dist = 0.0
        self._viewer = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.data.qpos[:] = 0.0

        # double the original goal range
        orig_low  = np.array([0.2, 0.0, 0.3], dtype=np.float32)
        orig_high = np.array([0.4, 0.4, 0.6], dtype=np.float32)
        center     = (orig_low + orig_high) / 2
        half_range = (orig_high - orig_low) / 2
        new_half   = half_range * 2
        new_low    = center - new_half
        new_high   = center + new_half

        self.goal = self.np_random.uniform(new_low, new_high)

        mujoco.mj_forward(self.model, self.data)
        ee = self.data.xpos[-1]
        self.prev_dist = np.linalg.norm(self.goal - ee)

        obs = self._get_obs().astype(np.float32)
        return obs, {}

    def _get_obs(self):
        ee_pos = self.data.xpos[-1].copy()      # last site/worldbody pos
        return np.concatenate([self.data.qpos[:7], self.goal - ee_pos])

    def step(self, action):
        # decode action into 7 trits: each ∈ {0,1,2} → {-1,0,+1}
        acts = np.base_repr(action, base=3).zfill(7)[::-1]
        for j, ch in enumerate(acts):
            sign = int(ch) - 1
            self.data.qpos[j] += sign * self.delta

        mujoco.mj_forward(self.model, self.data)

        ee = self.data.xpos[-1]
        dist = np.linalg.norm(self.goal - ee)

        # shaped reward as in the paper
        fg, m = 0.001, 100
        reward = np.arctan(((self.prev_dist - dist)/(np.pi/2))*(1/fg)) * m
        self.prev_dist = dist

        terminated = bool(dist < fg)
        truncated  = False

        obs = self._get_obs().astype(np.float32)
        return obs, reward, terminated, truncated, {}  

    def render(self, mode="human"):
        # If we haven’t created a viewer yet (or it failed), launch/passive returns an object
        if getattr(self, "_viewer", None) is None:
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
        # advance one physics step
        mujoco.mj_step(self.model, self.data)
        # only call sync if launch_passive actually returned something
        if self._viewer is not None:
            self._viewer.sync()