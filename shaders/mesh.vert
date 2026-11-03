#version 330 core

// Vertex inputs matching SubMeshData layout
layout (location = 0) in vec3 in_position;
layout (location = 1) in vec3 in_normal;
layout (location = 2) in vec2 in_uv;
layout (location = 3) in ivec4 in_bone_ids;
layout (location = 4) in vec4 in_bone_weights;

// Uniforms
uniform mat4 u_view;
uniform mat4 u_proj;
uniform mat4 u_model;

// Skeletal Skinning Uniforms
uniform int u_is_skinned;
uniform mat4 u_bones[128];

// Heatmap Uniform
uniform int u_mode;               // 0: Textured, 1: Base Color / No Material, 2: Weight Heatmap, 3: Unlit
uniform int u_active_bone_id;

// Outputs to Fragment Shader
out vec3 v_world_pos;
out vec3 v_world_normal;
out vec2 v_uv;
out float v_active_weight;

void main() {
    vec4 total_pos = vec4(0.0);
    vec3 total_norm = vec3(0.0);

    float sum_weights = in_bone_weights.x + in_bone_weights.y + in_bone_weights.z + in_bone_weights.w;

    if (u_is_skinned != 0 && sum_weights > 0.001) {
        for (int i = 0; i < 4; i++) {
            int b_id = in_bone_ids[i];
            float w = in_bone_weights[i];
            if (b_id >= 0 && b_id < 128 && w > 0.0) {
                mat4 bone_mat = u_bones[b_id];
                total_pos += (bone_mat * vec4(in_position, 1.0)) * w;
                total_norm += (mat3(bone_mat) * in_normal) * w;
            }
        }
        if (length(total_norm) > 1e-4) {
            total_norm = normalize(total_norm);
        } else {
            total_norm = in_normal;
        }
    } else {
        total_pos = vec4(in_position, 1.0);
        total_norm = in_normal;
    }

    // World-space transformations
    vec4 world_p = u_model * total_pos;
    v_world_pos = world_p.xyz;
    v_world_normal = normalize(mat3(u_model) * total_norm);
    v_uv = in_uv;

    // Calculate bone weight for Heatmap Mode
    float active_w = 0.0;
    if (u_mode == 2 && u_active_bone_id >= 0) {
        if (in_bone_ids.x == u_active_bone_id) active_w += in_bone_weights.x;
        if (in_bone_ids.y == u_active_bone_id) active_w += in_bone_weights.y;
        if (in_bone_ids.z == u_active_bone_id) active_w += in_bone_weights.z;
        if (in_bone_ids.w == u_active_bone_id) active_w += in_bone_weights.w;
    }
    v_active_weight = clamp(active_w, 0.0, 1.0);

    gl_Position = u_proj * u_view * world_p;
}
