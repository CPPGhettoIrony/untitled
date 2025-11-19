#version 330 core

in vec3 vertexPos;
in vec2 fragTexCoord;
in vec4 fragColor;

out vec4 finalColor;

uniform float u_time;
uniform vec2 u_resolution;

const float maxDistance = 100.0;
const float epsilon     = 0.001;

const uint  imax        = 5u;

vec3 camera_pos = vec3(0,1,-6);
vec3 camera_rot = vec3(0,0,0);
const float fov = 0.25;

const uint nLights = 1u;

struct Hit {

    float   d;          // SDF Distance
    float   len;        // Length of the ray from the origin

    bool    hit;        // If the ray hit something
    
    vec3    pos;        // Absolute ray hit position
    vec3    dir;        // Absolute ray direction

    vec3    rfp;        // object reference pose        for mapping
    mat3    rfr;        // object reference rotation    for mapping

    vec3    un;         // Unshaded normal
    vec3    normal;     // Normal of the hit surface

    uint    matID;      // Used for material operators;

    vec3    col;        // Unshaded color   
    float   ref;        // Reflectivity     
    float   shn;        // Shininess
    float   spc;        // Specular

    vec3    lco;        // Line color
    float   lth;        // Line thickness

};

/* - - - - - -  MATERIAL HELPERS - - - - */

uint hash(uint x, uint seed) {
    const uint m = 0x5bd1e995U;
    uint hash = seed;
    // process input
    uint k = x;
    k *= m;
    k ^= k >> 24;
    k *= m;
    hash *= m;
    hash ^= k;
    // some final mixing
    hash ^= hash >> 13;
    hash *= m;
    hash ^= hash >> 15;
    return hash;
}

// implementation of MurmurHash (https://sites.google.com/site/murmurhash/) for a  
// 2-dimensional unsigned integer input vector.

uint hash(uvec2 x, uint seed){
    const uint m = 0x5bd1e995U;
    uint hash = seed;
    // process first vector element
    uint k = x.x; 
    k *= m;
    k ^= k >> 24;
    k *= m;
    hash *= m;
    hash ^= k;
    // process second vector element
    k = x.y; 
    k *= m;
    k ^= k >> 24;
    k *= m;
    hash *= m;
    hash ^= k;
	// some final mixing
    hash ^= hash >> 13;
    hash *= m;
    hash ^= hash >> 15;
    return hash;
}


vec2 gradientDirection(uint hash) {
    switch (int(hash) & 3) { // look at the last two bits to pick a gradient direction
    case 0:
        return vec2(1.0, 1.0);
    case 1:
        return vec2(-1.0, 1.0);
    case 2:
        return vec2(1.0, -1.0);
    case 3:
        return vec2(-1.0, -1.0);
    }
}

float interpolate(float value1, float value2, float value3, float value4, vec2 t) {
    return mix(mix(value1, value2, t.x), mix(value3, value4, t.x), t.y);
}

vec2 fade(vec2 t) {
    // 6t^5 - 15t^4 + 10t^3
	return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

float perlin(vec2 position, uint seed) {
    vec2 floorPosition = floor(position);
    vec2 fractPosition = position - floorPosition;
    uvec2 cellCoordinates = uvec2(floorPosition);
    float value1 = dot(gradientDirection(hash(cellCoordinates, seed)), fractPosition);
    float value2 = dot(gradientDirection(hash((cellCoordinates + uvec2(1, 0)), seed)), fractPosition - vec2(1.0, 0.0));
    float value3 = dot(gradientDirection(hash((cellCoordinates + uvec2(0, 1)), seed)), fractPosition - vec2(0.0, 1.0));
    float value4 = dot(gradientDirection(hash((cellCoordinates + uvec2(1, 1)), seed)), fractPosition - vec2(1.0, 1.0));
    return interpolate(value1, value2, value3, value4, fade(fractPosition));
}

float perlin(vec2 position, int frequency, int octaveCount, float persistence, float lacunarity, uint seed) {
    float value = 0.0;
    float amplitude = 1.0;
    float currentFrequency = float(frequency);
    uint currentSeed = seed;
    for (int i = 0; i < octaveCount; i++) {
        currentSeed = hash(currentSeed, 0x0U); // create a new seed for each octave
        value += perlin(position * currentFrequency, currentSeed) * amplitude;
        amplitude *= persistence;
        currentFrequency *= lacunarity;
    }
    return value;
}

float voronoi(vec2 uv, float randomness, uint seed) {
    vec2 cell = floor(uv);
    vec2 fract = uv - cell;

    float minDist = 1e10;

    // Check neighboring cells (3x3 grid)
    for (int j = -1; j <= 1; ++j) {
        for (int i = -1; i <= 1; ++i) {
            uvec2 neighbor = uvec2(cell) + uvec2(i, j);

            // Hash determines the feature point inside the cell
            uint h = hash(neighbor, seed);
            vec2 random_offset = vec2(h & 0xFFu, (h >> 8) & 0xFFu) / 255.0; // [0,1)

            // Interpolate between center of cell and random offset
            vec2 offset = mix(vec2(0.5), random_offset, clamp(randomness, 0.0, 1.0));

            vec2 feature = vec2(i, j) + offset;
            float dist = length(fract - feature);

            minDist = min(minDist, dist);
        }
    }

    return minDist;
}

vec3 bumpNormal(vec2 uv, vec3 normal, vec3 h, float bumpStrength) {

    // Compute gradient (partial derivatives)
    float dx = (h.y - h.x) / epsilon;
    float dy = (h.z - h.x) / epsilon;

    // Robust tangent space basis from normal
    vec3 up = abs(normal.y) < 0.999 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 T = normalize(cross(up, normal));
    vec3 B = normalize(cross(normal, T));

    // Perturb the normal using the gradient
    vec3 bumped = normal 
                - T * dx * bumpStrength
                - B * dy * bumpStrength;

    return normalize(bumped);
}

float map_A(float i, float min0, float max0) {
    return (clamp(i, min0, max0) - min0) / (max0 - min0);  
}

// ONLY USE WITH A FUNCTION THAT HAS A VEC2 AS INPUT AND RETURNS A FLOAT
#define BUMP(func, uv, normal, strength) bumpNormal(uv, normal, vec3(func(uv), func(vec2(epsilon, .0) + uv), func(vec2(.0, epsilon) + uv)), strength)

// Direction & Fresnel

vec3 direction(Hit h) {return normalize(camera_pos - h.pos);}

float fresnel(Hit h) {return dot(normalize(camera_pos - h.pos), h.normal);}

/* - - - - - -  MATERIALS - - - - - - */


Hit world(Hit h) {
    h.col = vec3(0.4,0.7,1);
    return h;
}

// Default material
Hit def(Hit h, vec2 uv) {
    
    h.col = vec3(1.);
    h.ref = 0;
    h.shn = 64;
    h.spc = 1;

    return h;
}


float bump(vec2 uv) {
    uv *= 30;
    return 1-map_A(voronoi(uv, 1., 0u), 0.2, 0.5);
}

Hit cartoon(vec3 col, vec3 shn, float thickness, Hit h, vec2 uv) {    

    h.col = mix(col, shn, fresnel(h));
    h.lco = h.col *.3;
    h.lth = 0.15;
    h.ref = 0.3;
    h.shn = 64;
    h.spc = 1;

    //h.normal = BUMP(bump, uv, h.normal, 0.002);

    return h;

}

Hit A(Hit h, vec2 uv) {return cartoon(vec3(1.0, 0.3, 0.), vec3(1.0, 1.0, 0.05), 0.3, h, uv);}
Hit B(Hit h, vec2 uv) {return cartoon(vec3(0., 1.0, 0.3), vec3(0.05, 1.0, 1.0), 0.1, h, uv);}

Hit floor(Hit h, vec2 uv) {

    h.col = (distance(uv, vec2(0.5))>0.5)? vec3(0.,0.,1.) : vec3(1., 1., 0);
    h.lco = h.col *.3;
    h.ref = 0.3;
    h.shn = 64;
    h.spc = 1;
    h.lth = 0;

    return h;
}


// Mix diferent hits based on the three UV planes for mapping

vec2 averagev2(vec3 n, vec2 fx, vec2 fy, vec2 fz) {
    float f = max(max(n.x, n.y), n.z);
    return (f == n.x)? fx : ((f == n.y)? fy : fz);
}

vec3 modv(vec3 v, float mx, float offset) {
    return vec3(mod(v.x + offset, mx), mod(v.y + offset, mx), mod(v.z + offset, mx));
}

Hit getMaterial(Hit h, uint matID) {

    // reset normals if previously modified by another material
    h.normal = h.un;

    h.matID = matID;

    vec3 surfacePosition = modv(h.pos - h.rfp, 1., 0.);

    vec3 n = pow(abs(h.normal), vec3(8.0));
    n /= max(dot(n, vec3(1.0)), epsilon);

    vec2 uvX = surfacePosition.yz;
    vec2 uvY = surfacePosition.xz;
    vec2 uvZ = surfacePosition.xy;

    vec2 uv = averagev2(n, uvX, uvY, uvZ);

    switch(matID) {
     case 0u:
            return def(h, uv);
     case 1u:
            return A(h, uv);
     case 2u:
            return B(h, uv);
     case 3u:
            return floor(h, uv);

        default:
            return def(h, uv);
    }
    
}


/* - - - - - - ROTATION + TRANSLATION - - - - */


mat3 rotationFromEuler(vec3 euler) {
    float cx = cos(euler.x), sx = sin(euler.x);
    float cy = cos(euler.y), sy = sin(euler.y);
    float cz = cos(euler.z), sz = sin(euler.z);

    mat3 rx = mat3(
        1.0, 0.0, 0.0,
        0.0, cx, -sx,
        0.0, sx, cx
    );

    mat3 ry = mat3(
        cy, 0.0, sy,
        0.0, 1.0, 0.0,
        -sy, 0.0, cy
    );

    mat3 rz = mat3(
        cz, -sz, 0.0,
        sz, cz, 0.0,
        0.0, 0.0, 1.0
    );

    return ry * rz * rx; 
}

// The camera uses a different.. thing? idk
mat3 rotationFromEuler2(vec3 euler) {
    float cx = cos(euler.x), sx = sin(euler.x);
    float cy = cos(euler.y), sy = sin(euler.y);
    float cz = cos(euler.z), sz = sin(euler.z);

    mat3 rx = mat3(
        1.0, 0.0, 0.0,
        0.0, cx, -sx,
        0.0, sx, cx
    );

    mat3 ry = mat3(
        cy, 0.0, sy,
        0.0, 1.0, 0.0,
        -sy, 0.0, cy
    );

    mat3 rz = mat3(
        cz, -sz, 0.0,
        sz, cz, 0.0,
        0.0, 0.0, 1.0
    );

    return rz * ry * rx; 
}

// We are applying the transform to the space, not the primitive, so we must apply the inverse transform
vec3 applyTransform(vec3 p, vec3 pos, mat3 rot) {
    return transpose(rot) * (p - pos);
}


/* - - - - - -  PRIMITIVES - - - - - - */

// Sphere

float sphereFunc(vec3 p, vec3 pos, float r) {
    return length(p - pos) - r;
}

vec3 sphereNormal(vec3 p, vec3 pos, float r) {
    const vec2 e = vec2(epsilon, 0.0);
    return normalize(vec3(
        sphereFunc(p + e.xyy, pos, r) - sphereFunc(p - e.xyy, pos, r),
        sphereFunc(p + e.yxy, pos, r) - sphereFunc(p - e.yxy, pos, r),
        sphereFunc(p + e.yyx, pos, r) - sphereFunc(p - e.yyx, pos, r)
    ));
}

Hit sphere(vec3 p, vec3 pos, float r, uint matID) {
    Hit ret;
    ret.d       = sphereFunc(p , pos, r);
    ret.len     = 0.0;
    ret.pos     = p;
    ret.normal  = sphereNormal(p, pos, r);
    ret.un      = ret.normal;
    ret.rfp     = pos; 
    ret.rfr     = mat3(1.);
    ret = getMaterial(ret, matID); 
    return ret;
}

// Ground

Hit ground(vec3 p, float h, uint matID) {
    Hit ret;
    ret.d       = p.y - h;
    ret.len     = 0.0;
    ret.pos     = p;
    ret.normal  = vec3(0.0, 1.0, 0.0);
    ret.un      = ret.normal;
    ret.rfp     = vec3(.0); 
    ret.rfr     = mat3(1.);
    ret = getMaterial(ret, matID);
    return ret;
}

// Box

float sdBox( vec3 p, vec3 b ) {
    vec3 q = abs(p) - b;
    return length(max(q,0.0)) + min(max(q.x,max(q.y,q.z)),0.0);
}

float boxFunc(vec3 p, vec3 pos, mat3 rot, vec3 b ) {
    return sdBox(applyTransform(p, pos, rot), b);
}

vec3 boxNormal(vec3 p, vec3 pos, mat3 rot, vec3 b) {
    const vec2 e = vec2(epsilon, 0.0);
    return normalize(vec3(
        boxFunc(p + e.xyy, pos, rot, b) - boxFunc(p - e.xyy, pos, rot, b),
        boxFunc(p + e.yxy, pos, rot, b) - boxFunc(p - e.yxy, pos, rot, b),
        boxFunc(p + e.yyx, pos, rot, b) - boxFunc(p - e.yyx, pos, rot, b)
    ));
}

Hit box(vec3 p, vec3 pos, mat3 rot, vec3 b, uint matID) {
    Hit ret;
    ret.d       = boxFunc(p, pos, rot, b);
    ret.len     = 0.0;
    ret.pos     = p;
    ret.normal  = boxNormal(p, pos, rot, b);
    ret.un      = ret.normal;
    ret.rfp     = pos;
    ret.rfr     = rot;
    ret         = getMaterial(ret, matID); 
    return ret;
}

// Torus

float sdTorus( vec3 p, float r1, float r2 ) {
  vec2 q = vec2(length(p.xz)-r1,p.y);
  return length(q)-r2;
}

float torusFunc(vec3 p, vec3 pos, mat3 rot, float r1, float r2) {
    return sdTorus(applyTransform(p, pos, rot), r1, r2);
}

vec3 torusNormal(vec3 p, vec3 pos, mat3 rot, float r1, float r2) {
    const vec2 e = vec2(epsilon, 0.0);
    return normalize(vec3(
        torusFunc(p + e.xyy, pos, rot, r1, r2) - torusFunc(p - e.xyy, pos, rot, r1, r2),
        torusFunc(p + e.yxy, pos, rot, r1, r2) - torusFunc(p - e.yxy, pos, rot, r1, r2),
        torusFunc(p + e.yyx, pos, rot, r1, r2) - torusFunc(p - e.yyx, pos, rot, r1, r2)
    ));
}

Hit torus(vec3 p, vec3 pos, mat3 rot, float r1, float r2, uint matID) {
    Hit ret;
    ret.d       = torusFunc(p, pos, rot, r1, r2);
    ret.len     = 0.0;
    ret.pos     = p;
    ret.normal  = torusNormal(p, pos, rot, r1, r2);
    ret.un      = ret.normal;
    ret.rfp     = pos;
    ret.rfr     = rot;
    ret         = getMaterial(ret, matID); 
    return ret;
}

// Link

float sdLink( vec3 p, float le, float r1, float r2 ) {
    vec3 q = vec3( p.x, max(abs(p.y)-le,0.0), p.z );
    return length(vec2(length(q.xy)-r1,q.z)) - r2;
}

float linkFunc(vec3 p, vec3 pos, mat3 rot, float le, float r1, float r2) {
    return sdLink(applyTransform(p, pos, rot), le, r1, r2);
}

vec3 linkNormal(vec3 p, vec3 pos, mat3 rot, float le, float r1, float r2) {
    const vec2 e = vec2(epsilon, 0.0);
    return normalize(vec3(
        linkFunc(p + e.xyy, pos, rot, le, r1, r2) - linkFunc(p - e.xyy, pos, rot, le, r1, r2),
        linkFunc(p + e.yxy, pos, rot, le, r1, r2) - linkFunc(p - e.yxy, pos, rot, le, r1, r2),
        linkFunc(p + e.yyx, pos, rot, le, r1, r2) - linkFunc(p - e.yyx, pos, rot, le, r1, r2)
    ));
}

Hit link(vec3 p, vec3 pos, mat3 rot, float le, float r1, float r2, uint matID) {
    Hit ret;
    ret.d       = linkFunc(p, pos, rot, le, r1, r2);
    ret.len     = 0.0;
    ret.pos     = p;
    ret.normal  = linkNormal(p, pos, rot, le, r1, r2);
    ret.un      = ret.normal;
    ret.rfp     = pos;
    ret.rfr     = rot;
    ret         = getMaterial(ret, matID); 
    return ret;
}

// Cone

float sdCone(vec3 p, vec2 q) {
    // c is the sin/cos of the angle, h is height
    // Alternatively pass q instead of (c,h),
    // which is the point at the base in 2D

    vec2 w = vec2( length(p.xz), p.y );
    vec2 a = w - q*clamp( dot(w,q)/dot(q,q), 0.0, 1.0 );
    vec2 b = w - q*vec2( clamp( w.x/q.x, 0.0, 1.0 ), 1.0 );
    float k = sign( q.y );
    float d = min(dot( a, a ),dot(b, b));
    float s = max( k*(w.x*q.y-w.y*q.x),k*(w.y-q.y)  );
    return sqrt(d)*sign(s);
}

float coneFunc(vec3 p, vec3 pos, mat3 rot, float r, float h) {
    return sdCone(applyTransform(p, pos, rot), vec2(r, -h));
}

vec3 coneNormal(vec3 p, vec3 pos, mat3 rot, float r, float h) {
    const vec2 e = vec2(epsilon, 0.0);
    return normalize(vec3(
        coneFunc(p + e.xyy, pos, rot, r, h) - coneFunc(p - e.xyy, pos, rot, r, h),
        coneFunc(p + e.yxy, pos, rot, r, h) - coneFunc(p - e.yxy, pos, rot, r, h),
        coneFunc(p + e.yyx, pos, rot, r, h) - coneFunc(p - e.yyx, pos, rot, r, h)
    ));
}

Hit cone(vec3 p, vec3 pos, mat3 rot, float r, float h, uint matID) {
    Hit ret;
    ret.d       = coneFunc(p, pos, rot, r, h);
    ret.len     = 0.0;
    ret.pos     = p;
    ret.normal  = coneNormal(p, pos, rot, r, h);
    ret.un      = ret.normal;
    ret.rfp     = pos;
    ret.rfr     = rot;
    ret         = getMaterial(ret, matID); 
    return ret;
}

// Capsule 

float sdCapsule( vec3 p, float r, float h ) {
    p.y -= clamp( p.y, 0.0, h );
    return length( p ) - r;
}

float capsuleFunc(vec3 p, vec3 pos, mat3 rot, float r, float h) {
    return sdCapsule(applyTransform(p, pos, rot), r, h);
}

vec3 capsuleNormal(vec3 p, vec3 pos, mat3 rot, float r, float h) {
    const vec2 e = vec2(epsilon, 0.0);
    return normalize(vec3(
        capsuleFunc(p + e.xyy, pos, rot, r, h) - capsuleFunc(p - e.xyy, pos, rot, r, h),
        capsuleFunc(p + e.yxy, pos, rot, r, h) - capsuleFunc(p - e.yxy, pos, rot, r, h),
        capsuleFunc(p + e.yyx, pos, rot, r, h) - capsuleFunc(p - e.yyx, pos, rot, r, h)
    ));
}

Hit capsule(vec3 p, vec3 pos, mat3 rot, float r, float h, uint matID) {
    Hit ret;
    ret.d       = capsuleFunc(p, pos, rot, r, h);
    ret.len     = 0.0;
    ret.pos     = p;
    ret.normal  = capsuleNormal(p, pos, rot, r, h);
    ret.un      = ret.normal;
    ret.rfp     = pos;
    ret.rfr     = rot;
    ret         = getMaterial(ret, matID); 
    return ret;
}

// Cylinder

float sdCylinder( vec3 p, float r, float h ) {
    vec2 d = abs(vec2(length(p.xz),p.y)) - vec2(r,h);
    return min(max(d.x,d.y),0.0) + length(max(d,0.0));
}

float cylinderFunc(vec3 p, vec3 pos, mat3 rot, float r, float h) {
    return sdCylinder(applyTransform(p, pos, rot), r, h);
}

vec3 cylinderNormal(vec3 p, vec3 pos, mat3 rot, float r, float h) {
    const vec2 e = vec2(epsilon, 0.0);
    return normalize(vec3(
        cylinderFunc(p + e.xyy, pos, rot, r, h) - cylinderFunc(p - e.xyy, pos, rot, r, h),
        cylinderFunc(p + e.yxy, pos, rot, r, h) - cylinderFunc(p - e.yxy, pos, rot, r, h),
        cylinderFunc(p + e.yyx, pos, rot, r, h) - cylinderFunc(p - e.yyx, pos, rot, r, h)
    ));
}

Hit cylinder(vec3 p, vec3 pos, mat3 rot, float r, float h, uint matID) {
    Hit ret;
    ret.d       = cylinderFunc(p, pos, rot, r, h);
    ret.len     = 0.0;
    ret.pos     = p;
    ret.normal  = cylinderNormal(p, pos, rot, r, h);
    ret.un      = ret.normal;
    ret.rfp     = pos;
    ret.rfr     = rot;
    ret         = getMaterial(ret, matID); 
    return ret;
}

// Octahedron

float sdOctahedron( vec3 p, float s ) {

  p = abs(p);
  float m = p.x+p.y+p.z-s;
  vec3 q;
       if( 3.0*p.x < m ) q = p.xyz;
  else if( 3.0*p.y < m ) q = p.yzx;
  else if( 3.0*p.z < m ) q = p.zxy;
  else return m*0.57735027;
    
  float k = clamp(0.5*(q.z-q.y+s),0.0,s); 
  return length(vec3(q.x,q.y-s+k,q.z-k)); 

}

float octahedronFunc(vec3 p, vec3 pos, mat3 rot, float s) {
    return sdOctahedron(applyTransform(p, pos, rot), s);
}

vec3 octahedronNormal(vec3 p, vec3 pos, mat3 rot, float s) {
    const vec2 e = vec2(epsilon, 0.0);
    return normalize(vec3(
        octahedronFunc(p + e.xyy, pos, rot, s) - octahedronFunc(p - e.xyy, pos, rot, s),
        octahedronFunc(p + e.yxy, pos, rot, s) - octahedronFunc(p - e.yxy, pos, rot, s),
        octahedronFunc(p + e.yyx, pos, rot, s) - octahedronFunc(p - e.yyx, pos, rot, s)
    ));
}

Hit octahedron(vec3 p, vec3 pos, mat3 rot, float s, uint matID) {
    Hit ret;
    ret.d       = octahedronFunc(p, pos, rot, s);
    ret.len     = 0.0;
    ret.pos     = p;
    ret.normal  = octahedronNormal(p, pos, rot, s);
    ret.un      = ret.normal;
    ret.rfp     = pos;
    ret.rfr     = rot;
    ret         = getMaterial(ret, matID); 
    return ret;
}

// Ellipsoid

float sdEllipsoid( vec3 p, vec3 r ) {
  float k0 = length(p/r);
  float k1 = length(p/(r*r));
  return k0*(k0-1.0)/k1;
}

float ellipsoidFunc(vec3 p, vec3 pos, mat3 rot, vec3 b ) {
    return sdEllipsoid(applyTransform(p, pos, rot), b);
}

vec3 ellipsoidNormal(vec3 p, vec3 pos, mat3 rot, vec3 b) {
    const vec2 e = vec2(epsilon, 0.0);
    return normalize(vec3(
        ellipsoidFunc(p + e.xyy, pos, rot, b) - ellipsoidFunc(p - e.xyy, pos, rot, b),
        ellipsoidFunc(p + e.yxy, pos, rot, b) - ellipsoidFunc(p - e.yxy, pos, rot, b),
        ellipsoidFunc(p + e.yyx, pos, rot, b) - ellipsoidFunc(p - e.yyx, pos, rot, b)
    ));
}

Hit ellipsoid(vec3 p, vec3 pos, mat3 rot, vec3 b, uint matID) {
    Hit ret;
    ret.d       = ellipsoidFunc(p, pos, rot, b);
    ret.len     = 0.0;
    ret.pos     = p;
    ret.normal  = ellipsoidNormal(p, pos, rot, b);
    ret.un      = ret.normal;
    ret.rfp     = pos;
    ret.rfr     = rot;
    ret         = getMaterial(ret, matID); 
    return ret;
}

// Unary operators

Hit displace(Hit h, float d) {
    h.d   -= d;
    return h;
}

// CSG Operations

Hit union_(Hit a, Hit b) {
    return (a.d < b.d) ? a : b;
}

Hit intersect(Hit a, Hit b) {
    return (a.d > b.d) ? a : b;
}

Hit subtract(Hit a, Hit b) {
    Hit r;
    r.d = max(a.d, -b.d);
    
    // Which object defines the surface? Usually, the one with the greater distance
    if (a.d > -b.d) {
        r = a;
    } else {
        r = b;
        r.normal = -b.normal; // Invert B's normal
        r.un     = -b.un;
    }

    r.d = max(a.d, -b.d); // Always reassign correct distance
    return r;
}

Hit blendMaterials(Hit r, Hit a, Hit b, float hBlend) {

    r.un        = normalize(mix(b.un, a.un, hBlend));
    r.normal    = normalize(mix(b.normal, a.normal, hBlend));

    r.col       = mix(b.col, a.col, hBlend);
    r.ref       = mix(b.ref, a.ref, hBlend);
    
    r.shn       = mix(b.shn, a.shn, hBlend);        
    r.spc       = mix(b.spc, a.spc, hBlend);

    r.lth       = mix(b.lth, a.lth, hBlend);
    r.lco       = mix(b.lco, a.lco, hBlend);

    r.matID     = b.matID;

    return r;     

}

Hit morph(Hit a, Hit b, float k) {
    Hit r;

    r.d     = mix(b.d, a.d, k);
    r       = blendMaterials(r, a, b, k);

    return r;
}

Hit changeMaterial(Hit a, uint matID) {
    a = getMaterial(a, matID);
    return a;
}

Hit color(Hit a, Hit b, float k) {

    Hit ab      = changeMaterial(a, b.matID);
    Hit area    = intersect(a,  b);

    float d;
    if(k == 0) d = area.d < epsilon? 1 : 0;
    else
        d = (area.d >= epsilon)? clamp(k - area.d, 0, k)/k : 1;

    return morph(ab, a, d);
}

// polynomial smooth‑min helper
float smin(float d1, float d2, float k, out float h)
{
    h = clamp(0.5 + 0.5 * (d2 - d1) / k, 0.0, 1.0);
    return mix(d2, d1, h) - k * h * (1.0 - h);
}

Hit union_(Hit a, Hit b, float k)
{
    // 1) choose the winner by raw distance
    Hit r = (a.d < b.d) ? a : b;
    
    // 2) compute the blended distance (also get blend factor h)
    float hBlend;
    r.d     = smin(a.d, b.d, k, hBlend);

    r = blendMaterials(r, a, b, hBlend);

    // r.hit, r.pos, r.len remain intact, so the marcher keeps working
    return r;
}

float smax(float d1, float d2, float k, out float h) {
    h = clamp(0.5 - 0.5 * (d2 - d1) / k, 0.0, 1.0);
    return mix(d2, d1, h) + k * h * (1.0 - h);
}

Hit subtract(Hit a, Hit b, float k) {
    
    Hit r = (a.d > -b.d) ? a : b; // Choose the winner by raw distance for initial guess

    b.normal    = -b.normal;
    b.un        = -b.un;

    float hBlend;
    // The smooth maximum for a and -b.d
    r.d = smax(a.d, -b.d, k, hBlend);

    r = blendMaterials(r, a, b, hBlend);

    return r;
}

Hit intersect(Hit a, Hit b, float k) {
    
    Hit r = (a.d > -b.d) ? a : b; // Choose the winner by raw distance for initial guess

    float hBlend;
    // The smooth maximum for a and -b.d
    r.d = smax(a.d, b.d, k, hBlend);

    r = blendMaterials(r, a, b, hBlend);

    return r;
}

Hit joint(Hit a, Hit b, Hit c, float k) {
    k = c.d >= epsilon? clamp(k - c.d, k, 0) : k;
    return union_(union_(a, c), b, k);
}

// -------------- ACTUAL PROGRAM --------------

Hit Operator(vec3 p);
Hit scene(vec3 p);

Hit Operator(vec3 p){
	
    float t = u_time*0.001;
    Hit sph, oct;

	sph = sphere(p, vec3(sin(t), 1, 0.0), 0.5,  1u);
	oct = octahedron(p, vec3(-sin(t), 1, 0.0), rotationFromEuler(vec3(cos(t), sin(t), -cos(t))), 0.5, 2u);
	return union_(sph, oct, 0.5);
}

Hit scene(vec3 p){
	Hit hit;
	hit = Operator(p);
	hit = union_(hit, ground(p, 0.0,  3u));
	return hit;
}
// This will be replaced in opal.py

// Raymarching loop
Hit raymarch(vec3 ro, vec3 rd) {

    Hit h;
    h.dir = normalize(-rd);

    float t = 0.0;

    for (int i = 0; i < 512 && t < maxDistance; ++i) {

        vec3 p = ro + rd * t;
        h = scene(p);

        if (h.d < epsilon) {
            h.len = t;
            h.hit = true;
            return h;
        }
            
        t += h.d;
    }

    h.hit = false;

    return h; // background
}

struct Light {
    vec3    col;    //  color
    bool    point;  //  false = directional, true = point
    vec3    vec;    //          Direction           Position
    float   str;    //  Strength
    float   amb;    //  Ambient 
};

vec3 clampv01(vec3 v) {
    return vec3(clamp(v.x, 0, 1), clamp(v.y, 0, 1), clamp(v.z, 0, 1));
}

// Basic lighting
vec3 phong(vec3 col, Hit h, Light l, vec3 viewDir) {

    // If the light is a point, the light direction is the difference between the light position and hit position
    vec3 vec = l.point? normalize(h.pos - l.vec) : l.vec;

    vec3 ambient = col * (l.amb);

    float diff = max(dot(h.normal, -vec), 0.0);
    vec3 diffuse = col * diff;

    vec3 reflectDir = reflect(vec, h.normal);
    float spec = pow(max(dot(viewDir, reflectDir), 0), h.shn);
    vec3 specular = h.spc * vec3(spec); // white highlight

    return (ambient + diffuse + specular) * l.col * l.str;
}

//Process all lights
vec3 lighting(Hit h, Light ls[nLights], vec3 viewDir) {

    vec3 col = vec3(0.0);

    // All colors must be added
    for (uint i = 0u; i < nLights; ++i) {
        float s = ls[i].str;
        if (s > 0.0) {
            vec3 contrib = phong(h.col, h, ls[i], viewDir) * s;
            col += contrib;
        }
    }

    return col;
}

vec3 shadow(vec3 col, Hit h, Light l) {

    vec3 vec = l.point? normalize(h.pos - l.vec) : l.vec;

    Hit shd = raymarch(h.pos + h.normal * epsilon * 2, -vec);

    float d = dot(h.normal, vec);

    if(d < 0 && shd.hit && (shd.len < length(h.pos - l.vec) || !l.point)) {
        vec3 sc = col*(1+d);
        col = mix(sc, col, l.amb);
    }

    return col;

}

vec3 shadows(vec3 col, Hit h, Light ls[nLights]) {

    for(uint i = 0u; i < nLights; ++i)
        col = shadow(col, h, ls[i]);

    return col;
}

vec4 render(vec3 ro, vec3 rd, Light ls[nLights]) {

    vec3 viewDir = normalize(-rd);

    Hit hit = raymarch(ro, rd);

    // The skybox (when theres no hit, it renders the skybox, there's nothing to shadow or reflect there)
    if(hit.hit) {

        //return vec4(dot(viewDir, hit.normal), 0., 0., 1.);

        //Line thickness
        if(abs(dot(viewDir, hit.un)) < hit.lth) return vec4(hit.lco, 1);

        vec3 col = hit.col;

        // Add lighting
        col = lighting(hit, ls, viewDir);

        if(hit.ref > 0) {

            // Calculate reflection on the first iteration
            vec3 refDir = reflect(rd, hit.normal);
            Hit ref     = raymarch(hit.pos + hit.normal * epsilon * 2., refDir);

            // Apply first iteration reflection + shading + line thickness

            if(ref.hit) {
                viewDir = normalize(ref.pos - ro);
                bool line = abs(dot(refDir, ref.un)) < ref.lth;
                ref.col = line? ref.lco :lighting(ref, ls, viewDir);
                ref.col = shadows(ref.col, ref, ls);
            } else ref = world(ref);

            col         = mix(col, ref.col, hit.ref);

            // Final reflection value
            float fref  = hit.ref;

            // Apply every iteration
            for(uint i = 1u; i <= imax; ++i) {

                if(!ref.hit) break;

                refDir  = reflect(refDir, ref.normal);
                ref     = raymarch(ref.pos + ref.normal * epsilon * 2., refDir);

                if(ref.hit) {
                    viewDir = normalize(ref.pos - ro);
                    bool line = abs(dot(refDir, ref.un)) < ref.lth;
                    ref.col = line? ref.lco :lighting(ref, ls, viewDir);
                    ref.col = shadows(ref.col, ref, ls);
                } else ref = world(ref);

                fref   *= ref.ref;

                col     = mix(col, ref.col, fref);

            }

        } 

        // Get projected shadow from other objects
        col = shadows(col, hit, ls);

        return vec4(col, 1.0);

    }

    return vec4(world(hit).col, 1.);

}

void main() {

    vec2 uv = vec2(1 - fragTexCoord.x, fragTexCoord.y);
    vec2 normCoord = ((2.0 * (vec2(1) - uv) * u_resolution - u_resolution) / u_resolution.y);

    vec3 ro = camera_pos;
    vec3 rd = rotationFromEuler(camera_rot) * normalize(vec3(normCoord * fov, 1.0));


    Light ls[nLights] = Light[](
        Light(vec3(1.0, 1.0, 1.0), 
        true, 
        vec3(0, 3, -1), 
        1,
        0.5)
    );

    finalColor = render(ro, rd, ls);
}