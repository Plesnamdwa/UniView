#version 330 core

in vec3 v_world_pos;
in vec3 v_world_normal;
out vec4 fragColor;

uniform vec3 u_cam_pos;
uniform vec4 u_bone_color;       // Base bone color with alpha (e.g. orange for selected, cyan for regular)
uniform int u_is_selected;

void main() {
    vec3 N = normalize(v_world_normal);
    if (!gl_FrontFacing) {
        N = -N;
    }
    vec3 V = normalize(u_cam_pos - v_world_pos);

    // Directional light for crisp octahedral facet definition
    vec3 light_dir = normalize(vec3(0.5, 0.9, 0.6));
    float diff = max(dot(N, light_dir), 0.0) * 0.6 + 0.4;

    // View-aligned rim glow
    float rim = pow(1.0 - max(dot(N, V), 0.0), 2.5) * 0.35;

    vec3 base_rgb = u_bone_color.rgb;
    if (u_is_selected != 0) {
        // Brighten selected bone highlight
        base_rgb = mix(base_rgb, vec3(1.0, 0.95, 0.5), 0.25);
    }

    vec3 final_rgb = base_rgb * diff + vec3(rim);
    fragColor = vec4(clamp(final_rgb, 0.0, 1.0), u_bone_color.a);
}
