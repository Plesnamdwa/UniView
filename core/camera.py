"""
core/camera.py
Mathematical Arcball / Orbit Camera using PyGLM.
Supports:
- Left Mouse Button (LMB) Drag: Orbit around pivot target (yaw / pitch)
- Right Mouse Button (RMB) Drag: Pan camera target along screen-aligned right/up axes
- Mouse Scroll Wheel: Smooth zoom in/out with minimum distance clamping
- Smooth Eased Interpolation: Animated transition when framing the full scene or isolating a sub-mesh AABB
"""

from __future__ import annotations
import math
import glm
from core.math_utils import AABB


class ArcballCamera:
    """Hardware-ready mathematical Orbit / Arcball camera built on PyGLM."""

    def __init__(
        self,
        target: glm.vec3 = glm.vec3(0.0, 1.0, 0.0),
        distance: float = 4.0,
        yaw_deg: float = 35.0,
        pitch_deg: float = 20.0,
        fov_deg: float = 45.0,
        aspect_ratio: float = 16.0 / 9.0,
        near_clip: float = 0.01,
        far_clip: float = 1000.0,
    ):
        self.target: glm.vec3 = glm.vec3(target)
        self.distance: float = max(float(distance), 0.05)
        self.yaw_deg: float = float(yaw_deg)
        self.pitch_deg: float = float(glm.clamp(pitch_deg, -89.0, 89.0))

        self.fov_deg: float = float(fov_deg)
        self.aspect_ratio: float = max(float(aspect_ratio), 0.01)
        self.near_clip: float = float(near_clip)
        self.far_clip: float = float(far_clip)

        self.min_distance: float = 0.02
        self.max_distance: float = 5000.0

        # Sensitivity settings
        self.orbit_sensitivity: float = 0.35
        self.pan_sensitivity: float = 0.0018
        self.zoom_sensitivity: float = 0.12

        # Default home state for reset
        self._home_target: glm.vec3 = glm.vec3(self.target)
        self._home_distance: float = self.distance
        self._home_yaw: float = self.yaw_deg
        self._home_pitch: float = self.pitch_deg

        # Smooth interpolation state (for sub-mesh Focus / Isolate & scene framing)
        self._animating: bool = False
        self._anim_elapsed: float = 0.0
        self._anim_duration: float = 0.32  # seconds
        self._start_target: glm.vec3 = glm.vec3(self.target)
        self._end_target: glm.vec3 = glm.vec3(self.target)
        self._start_distance: float = self.distance
        self._end_distance: float = self.distance

    @property
    def is_animating(self) -> bool:
        return self._animating

    @property
    def position(self) -> glm.vec3:
        """Compute world-space eye position from spherical coordinates around target."""
        yaw_rad = math.radians(self.yaw_deg)
        pitch_rad = math.radians(self.pitch_deg)

        cos_pitch = math.cos(pitch_rad)
        x = self.target.x + self.distance * cos_pitch * math.sin(yaw_rad)
        y = self.target.y + self.distance * math.sin(pitch_rad)
        z = self.target.z + self.distance * cos_pitch * math.cos(yaw_rad)
        return glm.vec3(x, y, z)

    @property
    def forward(self) -> glm.vec3:
        """Normalized direction vector from camera eye toward target."""
        diff = self.target - self.position
        length = glm.length(diff)
        if length < 1e-6:
            return glm.vec3(0.0, 0.0, -1.0)
        return diff / length

    @property
    def right(self) -> glm.vec3:
        """Normalized screen-right vector in world space."""
        world_up = glm.vec3(0.0, 1.0, 0.0)
        r = glm.cross(self.forward, world_up)
        if glm.length(r) < 1e-6:
            return glm.vec3(1.0, 0.0, 0.0)
        return glm.normalize(r)

    @property
    def up(self) -> glm.vec3:
        """Normalized camera-up vector in world space."""
        return glm.normalize(glm.cross(self.right, self.forward))

    def get_view_matrix(self) -> glm.mat4:
        """Return the 4x4 View Matrix."""
        return glm.lookAt(self.position, self.target, self.up)

    def get_projection_matrix(self) -> glm.mat4:
        """Return the 4x4 Perspective Projection Matrix."""
        # Dynamically adapt near/far planes to distance to avoid Z-fighting
        dynamic_near = max(0.005, min(self.near_clip, self.distance * 0.01))
        dynamic_far = max(self.far_clip, self.distance * 20.0)
        return glm.perspective(
            math.radians(self.fov_deg),
            self.aspect_ratio,
            dynamic_near,
            dynamic_far,
        )

    def get_orientation_matrix(self) -> glm.mat4:
        """Return pure 3D rotation matrix of the camera view (for corner XYZ Gizmo)."""
        view = self.get_view_matrix()
        rot = glm.mat4(glm.mat3(view))
        return rot

    def set_aspect_ratio(self, width: int, height: int) -> None:
        if height > 0 and width > 0:
            self.aspect_ratio = float(width) / float(height)

    def orbit(self, delta_x: float, delta_y: float) -> None:
        """Rotate camera around target pivot (LMB drag)."""
        self.yaw_deg -= delta_x * self.orbit_sensitivity
        self.pitch_deg += delta_y * self.orbit_sensitivity
        self.pitch_deg = float(glm.clamp(self.pitch_deg, -89.0, 89.0))
        # Wrap yaw within [-360, 360]
        self.yaw_deg = math.fmod(self.yaw_deg, 360.0)

    def pan(self, delta_x: float, delta_y: float) -> None:
        """Translate camera target horizontally and vertically (RMB drag)."""
        self._animating = False
        scale = self.distance * self.pan_sensitivity
        offset = (-self.right * delta_x + self.up * delta_y) * scale
        self.target += offset

    def zoom(self, wheel_steps: float) -> None:
        """Zoom in or out smoothly using mouse wheel delta (e.g., +1.0 or -1.0 steps)."""
        self._animating = False
        factor = math.pow(1.0 - self.zoom_sensitivity, wheel_steps)
        self.distance = float(glm.clamp(self.distance * factor, self.min_distance, self.max_distance))

    def frame_aabb(self, aabb: AABB, animate: bool = True, save_as_home: bool = False) -> None:
        """
        Compute optimal camera target and distance to frame the given AABB in the viewport.
        If `animate=True`, smoothly interpolates from current state to target state.
        """
        if not aabb.is_valid:
            return

        new_target = aabb.center
        half_fov_rad = math.radians(self.fov_deg * 0.5)
        sin_half_fov = max(math.sin(half_fov_rad), 0.1)

        # Ensure full bounding sphere fits with comfortable padding (1.35x)
        radius = max(aabb.radius, 0.05)
        optimal_dist = (radius / sin_half_fov) * 1.25
        optimal_dist = float(glm.clamp(optimal_dist, self.min_distance, self.max_distance))

        if save_as_home:
            self._home_target = glm.vec3(new_target)
            self._home_distance = optimal_dist
            self._home_yaw = 35.0
            self._home_pitch = 20.0

        if animate:
            self._start_target = glm.vec3(self.target)
            self._end_target = glm.vec3(new_target)
            self._start_distance = self.distance
            self._end_distance = optimal_dist
            self._anim_elapsed = 0.0
            self._animating = True
        else:
            self._animating = False
            self.target = glm.vec3(new_target)
            self.distance = optimal_dist

    def reset_to_home(self, animate: bool = True) -> None:
        """Restore camera to the initial framed home position."""
        if animate:
            self._start_target = glm.vec3(self.target)
            self._end_target = glm.vec3(self._home_target)
            self._start_distance = self.distance
            self._end_distance = self._home_distance
            self._anim_elapsed = 0.0
            self._animating = True
        else:
            self._animating = False
            self.target = glm.vec3(self._home_target)
            self.distance = self._home_distance
            self.yaw_deg = self._home_yaw
            self.pitch_deg = self._home_pitch

    def update_animation(self, dt: float) -> bool:
        """
        Step smooth camera interpolation by `dt` seconds.
        Returns True if the camera moved and viewport needs a redraw.
        """
        if not self._animating:
            return False

        self._anim_elapsed += max(dt, 0.001)
        t = min(self._anim_elapsed / max(self._anim_duration, 0.01), 1.0)
        # Cubic ease-out: 1 - (1 - t)^3
        ease_t = 1.0 - math.pow(1.0 - t, 3.0)

        self.target = glm.mix(self._start_target, self._end_target, ease_t)
        self.distance = float(self._start_distance + (self._end_distance - self._start_distance) * ease_t)

        if t >= 1.0:
            self.target = glm.vec3(self._end_target)
            self.distance = self._end_distance
            self._animating = False

        return True
