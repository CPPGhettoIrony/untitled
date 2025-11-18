#version 330 core

in vec3 vertexPos;
in vec2 fragTexCoord;
in vec4 fragColor;

out vec4 finalColor;

uniform float u_time;
uniform vec2 u_resolution;

void main() {

    vec2 uv = vec2(1 - fragTexCoord.x, fragTexCoord.y);
    vec2 normCoord = ((2.0 * (vec2(1) - uv) * u_resolution - u_resolution) / u_resolution.y);

    finalColor = vec4(fragTexCoord, 0., 1.);
}