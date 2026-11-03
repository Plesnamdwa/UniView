#version 330 core

layout (location = 0) in vec3 in_position;

uniform mat4 u_view;
uniform mat4 u_proj;
uniform mat4 u_model;

out vec3 v_world_pos;

void main() {
    vec4 world_p = u_model * vec4(in_position, 1.0);
    v_world_pos = world_p.xyz;
    gl_Position = u_proj * u_view * world_p;
}
