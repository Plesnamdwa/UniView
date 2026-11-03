#version 330 core

in vec3 v_world_pos;
in vec3 v_world_normal;
in vec2 v_uv;
in float v_active_weight;

out vec4 fragColor;

// Viewport Camera Position
uniform vec3 u_cam_pos;

// Viewport Mode:
// 0: Textured Mode
// 1: No Material / Base Color Mode
// 2: Vertex Weight Heatmap Mode
// 3: Unlit Flat
uniform int u_mode;

// Material Slot Uniforms
uniform vec3 u_base_color;
uniform vec3 u_ambient;
uniform vec3 u_specular;
uniform float u_shininess;
uniform float u_opacity;
uniform int u_shading_type;      // 0: Phong, 1: Unlit

// Texture Uniforms
uniform int u_has_texture;
uniform sampler2D u_diffuse_texture;

// Weight Heatmap Color Function: 0.0 (Blue) -> 0.25 (Cyan) -> 0.5 (Green) -> 0.75 (Yellow) -> 1.0 (Red)
vec3 get_heatmap_color(float w) {
    if (w <= 0.0001) {
        // Deep indigo / dark blue for unweighted vertices
        return vec3(0.04, 0.08, 0.35);
    }
    float val = clamp(w, 0.0, 1.0);
    if (val < 0.25) {
        float t = val / 0.25;
        return mix(vec3(0.0, 0.2, 1.0), vec3(0.0, 0.9, 1.0), t);
    } else if (val < 0.5) {
        float t = (val - 0.25) / 0.25;
        return mix(vec3(0.0, 0.9, 1.0), vec3(0.1, 0.9, 0.1), t);
    } else if (val < 0.75) {
        float t = (val - 0.5) / 0.25;
        return mix(vec3(0.1, 0.9, 0.1), vec3(1.0, 0.9, 0.0), t);
    } else {
        float t = (val - 0.75) / 0.25;
        return mix(vec3(1.0, 0.9, 0.0), vec3(1.0, 0.05, 0.0), t);
    }
}

void main() {
    vec3 N = normalize(v_world_normal);
    // Double-sided lighting: flip normal if viewing backface
    if (!gl_FrontFacing) {
        N = -N;
    }

    vec3 V = normalize(u_cam_pos - v_world_pos);

    // 1. Determine Base Diffuse Color
    vec3 albedo = u_base_color;
    float alpha = u_opacity;

    if (u_mode == 2) {
        // Heatmap Mode
        albedo = get_heatmap_color(v_active_weight);
    } else if (u_mode == 0 && u_has_texture != 0) {
        // Textured Mode with valid texture
        vec4 tex_sample = texture(u_diffuse_texture, v_uv);
        albedo = tex_sample.rgb;
        alpha *= tex_sample.a;
    }

    // 2. Unlit or Heatmap Shading
    if (u_shading_type == 1 || u_mode == 3) {
        fragColor = vec4(albedo, alpha);
        return;
    }

    if (u_mode == 2) {
        // Subtle directional lighting on heatmap to show 3D form
        vec3 light_dir = normalize(vec3(0.4, 0.8, 0.6));
        float diff = max(dot(N, light_dir), 0.0) * 0.4 + 0.6;
        fragColor = vec4(albedo * diff, alpha);
        return;
    }

    // 3. Balanced Studio 3-Point + Hemisphere Shading Model
    // Build a camera-relative basis so the model is well-lit from every orbit angle
    vec3 world_up = vec3(0.0, 1.0, 0.0);
    vec3 cam_right = normalize(cross(world_up, V));
    if (length(cam_right) < 0.001) {
        cam_right = vec3(1.0, 0.0, 0.0);
    }
    vec3 cam_up = normalize(cross(V, cam_right));

    // Studio Key Light (top-right of camera) & Soft Fill Light (left of camera)
    vec3 key_light = normalize(V * 0.75 + cam_right * 0.42 + cam_up * 0.52);
    vec3 fill_light = normalize(V * 0.55 - cam_right * 0.65 - cam_up * 0.15);
    vec3 back_light = normalize(-V * 0.60 + world_up * 0.80);

    // Soft Hemisphere Ambient (never pitch black even when material u_ambient is #000000)
    float hemi = N.y * 0.5 + 0.5;
    vec3 hemi_ambient = mix(vec3(0.28, 0.28, 0.32), vec3(0.46, 0.46, 0.48), hemi);
    vec3 effective_ambient = max(u_ambient, hemi_ambient);
    vec3 ambient = effective_ambient * albedo;

    // Diffuse with soft wrap (Half-Lambert blend) for natural character & cloth shading
    float ndotl_key = dot(N, key_light);
    float diff_key = mix(max(ndotl_key, 0.0), pow(ndotl_key * 0.5 + 0.5, 1.4), 0.35) * 0.72;
    float diff_fill = max(dot(N, fill_light), 0.0) * 0.28;
    float diff_back = max(dot(N, back_light), 0.0) * 0.12;
    vec3 diffuse = (diff_key + diff_fill + diff_back) * albedo;

    // Specular (Blinn-Phong)
    vec3 H = normalize(key_light + V);
    float spec_angle = max(dot(N, H), 0.0);
    float spec_factor = pow(spec_angle, max(u_shininess, 8.0));
    vec3 specular = spec_factor * u_specular * 0.65;

    // Subtle Fresnel / Rim highlight for 3D silhouette clarity
    float rim_factor = pow(1.0 - max(dot(N, V), 0.0), 3.2) * 0.10;
    vec3 rim = vec3(rim_factor);

    vec3 final_color = ambient + diffuse + specular + rim;
    fragColor = vec4(clamp(final_color, 0.0, 1.0), alpha);
}
