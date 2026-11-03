#version 330 core

in vec3 v_world_pos;
out vec4 fragColor;

uniform vec3 u_cam_pos;
uniform float u_grid_size;     // Major grid line spacing (e.g. 1.0)
uniform float u_fade_dist;     // Outer distance at which grid fades out
uniform vec3 u_grid_color;     // Theme-adaptive neutral grid line color

void main() {
    float g_size = max(u_grid_size, 0.01);
    vec2 coord = v_world_pos.xz / g_size;

    // Major grid lines (1.0 spacing)
    vec2 grid = abs(fract(coord - 0.5) - 0.5) / max(fwidth(coord), vec2(0.0001));
    float line_major = min(grid.x, grid.y);
    float c_major = 1.0 - min(line_major, 1.0);

    // Minor grid lines (0.1 spacing)
    vec2 coord_sub = v_world_pos.xz / (g_size * 0.1);
    vec2 grid_sub = abs(fract(coord_sub - 0.5) - 0.5) / max(fwidth(coord_sub), vec2(0.0001));
    float line_sub = min(grid_sub.x, grid_sub.y);
    float c_sub = (1.0 - min(line_sub, 1.0)) * 0.28;

    float grid_alpha = max(c_major * 0.45, c_sub * 0.22);

    // Theme-adaptive neutral grid line color
    vec3 line_color = u_grid_color;

    // X-Axis (Z = 0) highlighted in clean red
    if (abs(v_world_pos.z) < 0.025 * g_size) {
        line_color = vec3(0.85, 0.22, 0.22);
        grid_alpha = max(grid_alpha, 0.85);
    }
    // Z-Axis (X = 0) highlighted in clean blue
    else if (abs(v_world_pos.x) < 0.025 * g_size) {
        line_color = vec3(0.22, 0.48, 0.90);
        grid_alpha = max(grid_alpha, 0.85);
    }

    // Distance fade towards horizon
    float dist = length(v_world_pos.xz - u_cam_pos.xz);
    float fade_start = u_fade_dist * 0.35;
    float fade = 1.0 - smoothstep(fade_start, u_fade_dist, dist);

    float final_alpha = grid_alpha * fade;
    if (final_alpha < 0.004) {
        discard;
    }

    fragColor = vec4(line_color, final_alpha);
}
