#version 300 es
precision mediump float;

in vec2 uv;
out vec4 fragColor;

uniform int timems;

uniform float msaaLevel;
uniform int marchingDepth;
uniform int tracingDepth;

uniform float fogDensity;
uniform bool enableShadows;
uniform bool enableToonEffect;
uniform bool enableTasm;

uniform mat4 cameraTrafo;

uniform float screenWidth;
uniform float screenHeight;

uniform int numLights;
uniform sampler2D sekaiData;
uniform sampler2D lightsData;
uniform sampler2D tasmData;

uniform sampler2D skyTexture;

const int PlaneType = 0;
const int SphereType = 1;
const int BoxType = 2;
const int LensType = 3;

int sdfCount = 0;
int objectsOnRayCount = 0;
int[16] objectsOnRay;
int numObjectsOnRay = 0;

float randomSeed = 0.0;


float[64] tasmRegisters;
const int REG_VOID = 0;
const int REG_PTR_VOID = 1;
const int REG_PC = 2;
const int REG_PTR_PC = 3;
const int REG_SP = 4;
const int REG_PTR_SP = 5;
const int REG_PARAM1 = 6;
const int REG_PARAM2 = 7;
const int REG_PTR_PARAM1 = 8;
const int REG_PTR_PARAM2 = 9;
const int REG_COLOR_R = 10;
const int REG_COLOR_G = 11;
const int REG_COLOR_B = 12;
const int REG_NORMAL_X = 13;
const int REG_NORMAL_Y = 14;
const int REG_NORMAL_Z = 15;
const int REG_ATTRIB_1 = 16;
const int REG_ATTRIB_2 = 17;
const int REG_ATTRIB_3 = 18;
const int REG_PTR_COLOR = 19;
const int REG_PTR_NORMAL = 20;
const int REG_PTR_ATTRIBUTES = 21;
const int REG_PTR_ATTRIB_2 = 22;
const int REG_PTR_ATTRIB_3 = 23;
const int REG_END_VALUE = 24;
const int REG_PTR_END_VALUE = 25;
const int REG_ENV_TIMEMS = 26;
const int REG_ENV_ST_X = 27;
const int REG_ENV_ST_Y = 28;
const int REG_ENV_RAY_DISTANCE = 29;
const int REG_STACK = 32;
const int REG_USER = 56;


int intr(float v)
{
    return int(round(v));
}


int makeCubeLocator(vec3 v)
{
    const float cubeSize = 4.0;

    vec3 t = v / cubeSize;

    int cubeNr = (int(t.x) << 12) + (int(t.y) << 6) + int(t.z);
    return cubeNr;
}

vec3 resolveCubeLocator(int cubeNr)
{
    const int cubeSize = 4;
    return vec3(
        float((cubeNr >> 12) * cubeSize),
        float(((cubeNr >> 6) % 64) * cubeSize),
        float((cubeNr % 64) * cubeSize)
    );
}

int makeSuperCubeLocator(vec3 v, int level)
{
    float cubeSize = float(4 << level);

    vec3 t = v / cubeSize;

    int cubeNr = (int(t.x) << 12) + (int(t.y) << 6) + int(t.z);
    return cubeNr;
}

bool isSuperCubeEmpty(int superCubeNr, int level)
{
    vec4 data = texelFetch(sekaiData, ivec2(superCubeNr / 4, 8000 + level), 0);
    return data[superCubeNr % 4] < 0.5;
}

int makeObjectLocator(vec3 locInCube)
{
    int ox = int(floor(locInCube.x));
    int oy = int(floor(locInCube.y));
    int oz = int(floor(locInCube.z));
    return (ox << 4) + (oy << 2) + oz;
}

vec3 resolveObjectLocator(int objLoc)
{
    return vec3(
        float(objLoc >> 4) + 0.5,
        float((objLoc >> 2) & 3) + 0.5,
        float(objLoc & 3) + 0.5
    );
}

int makeWorldLocator(int cubeNr, int objLoc)
{
    return (cubeNr << 6) + objLoc;
}

ivec2 resolveWorldLocator(int wl)
{
    return ivec2(
        wl >> 6,
        wl % 64
    );
}

mat4 cubeTrafo(int cubeNr)
{
    vec3 p = resolveCubeLocator(cubeNr);
    return mat4(
        vec4(1.0, 0.0, 0.0, 0.0),
        vec4(0.0, 1.0, 0.0, 0.0),
        vec4(0.0, 0.0, 1.0, 0.0),
        vec4(p, 1.0)
    );
}

mat4 cubeTrafoInverse(int cubeNr)
{
    vec3 p = resolveCubeLocator(cubeNr);
    return mat4(
        vec4(1.0, 0.0, 0.0, 0.0),
        vec4(0.0, 1.0, 0.0, 0.0),
        vec4(0.0, 0.0, 1.0, 0.0),
        vec4(-p, 1.0)
    );
}

ivec2 cubeDataOffset(int cubeNr)
{
    return ivec2(
        (cubeNr % 128) * 64,
        cubeNr / 128
    );
}

ivec2 objectDataOffset(int obj)
{
    int cubeNr = resolveWorldLocator(obj).x;
    int objLoc = resolveWorldLocator(obj).y;
    return cubeDataOffset(cubeNr) + ivec2(2 + objLoc * 1, 0);
}

int objectDataEntry(int obj, int idx)
{
    ivec2 r = resolveWorldLocator(obj);
    int cubeNr = r.x;
    int objLoc = r.y;
    ivec2 offset = cubeDataOffset(cubeNr) + ivec2(2 + objLoc / 2, 0);
    if (objLoc % 2 == 0)
    {
        if (idx == 0)
        {
            return intr(texelFetch(sekaiData, offset, 0).r);
        }
        else
        {
            return intr(texelFetch(sekaiData, offset, 0).g);
        }
    }
    else
    {
        if (idx == 0)
        {
            return intr(texelFetch(sekaiData, offset, 0).b);
        }
        else
        {
            return intr(texelFetch(sekaiData, offset, 0).a);
        }

    }
}

vec2 aabbMinMax(float origin, float dir, float boxMin, float boxMax)
{
    if (abs(dir) < 0.000001)
    {
        dir += 0.000001;
    }
    float tMin = (boxMin - origin) / dir;
    float tMax = (boxMax - origin) / dir;

    if (tMax < tMin)
    {
        return vec2(tMax, tMin);
    }
    else
    {
        return vec2(tMin, tMax);
    }
}

vec3 hitAabb(vec3 origin, vec3 rayDirection)
{
    vec2 tx = aabbMinMax(origin.x, rayDirection.x, -0.5, 0.5);
    vec2 ty = aabbMinMax(origin.y, rayDirection.y, -0.5, 0.5);
    vec2 tz = aabbMinMax(origin.z, rayDirection.z, -0.5, 0.5);

    // greatest min and smallest max
    float rayMin = (tx.s > ty.s) ? tx.s : ty.s;
    float rayMax = (tx.t < ty.t) ? tx.t : ty.t;

    if (tx.s > ty.t || ty.s > tx.t)
    {
        return vec3(0.0);
    }
    if (rayMin > tz.t || tz.s > rayMax)
    {
        return vec3(0.0);
    }
    if (tz.s > rayMin)
    {
        rayMin = tz.s;
    }
    if (tz.t < rayMax)
    {
        rayMax = tz.t;
    }
    return vec3(rayMin, rayMax, 1.0);
}

float lerp(float a, float b, float c)
{
    return a + (b - a) * c;
}

float seededRandom(vec2 st)
{
    st += vec2(randomSeed);
    randomSeed += 1.0;
    return fract(sin(dot(st.xy, vec2(12.9898,78.233))) * 43758.5453123);
}

float random(vec2 st)
{
    return fract(sin(dot(st.xy, vec2(12.9898,78.233))) * 43758.5453123);
}

mat2 rotate2d(float angle)
{
    return mat2(cos(angle), -sin(angle),
                sin(angle), cos(angle));
}

/*
vec2 wrapSt(vec2 st)
{
    float s = st.s;
    float t = st.t;
    if (s < 0.0)
    {
        s = -s + 0.5;
    }
    if (t < 0.0)
    {
        t = -t + 0.5;
    }
    if (s > 1.0)
    {
        s -= floor(s);
    }
    if (t > 1.0)
    {
        t -= floor(t);
    }
    return vec2(s, t);
}
*/

/* Creates a transformation matrix for transforming to surface space.
 */
mat4 createSurfaceTrafo(vec3 normal)
{
    vec3 p = vec3(1.0, 0.0, 0.0);

    vec3 tangent = p;

    // rotate normal around y counter-clockwise for second point
    if (abs(dot(normal, vec3(0.0, 1.0, 0.0))) < 1.0)
    {
        p = vec3(
            normal.x * cos(0.1) - normal.z * sin(0.1),
            0.0,
            normal.x * sin(0.1) - normal.z * cos(0.1)
        );

        float dp = dot(normal, p);
        tangent = normalize(p - normal * dp);
    }

    vec3 bitangent = normalize(cross(normal, tangent));
   
    return mat4(
        vec4(tangent, 0.0),
        vec4(bitangent, 0.0),
        vec4(normal, 0.0),
        vec4(0.0, 0.0, 0.0, 1.0)
    );
}

/*
vec2 mosaic(vec2 st, float size)
{
    return vec2(ceil(st.x * size) / size, ceil(st.y * size) / size);
}
*/

/* Generates a normal map from a height map.
 */
vec3 generateBumpNormal(float a, float b, float c, float d)
{
    return normalize(vec3(
        a - b,
        c - d,
        1.0
    ));
}

vec2 generateMipMap(vec2 st, int level)
{
    return floor(st * float(level)) / float(level);
}

float generateLine(vec2 st, float start, float end)
{
    return step(st.y, end) * (1.0 - step(st.y, start));
}

/*
float generateLines(vec2 pos, float b)
{
    float scale = 10.0;
    pos *= scale;
    return smoothstep(0.0, 0.5 + b * 0.5, abs((sin(pos.x * 3.1415) + b * 2.0)) * 0.5);
}
*/

float generateWhiteNoise(vec2 st)
{
    return fract(sin(dot(st.xy, vec2(12.9898,78.233))) * 43758.5453123);
}

/*
float generateNoise(vec2 st)
{
    vec2 i = floor(st);
    vec2 f = fract(st);
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(random(i  + vec2(0.0, 0.0)),
                   random(i + vec2(1.0, 0.0)), u.x),
               mix(random(i + vec2(0.0, 1.0)),
                   random(i + vec2(1.0, 1.0)), u.x), u.y);
}
*/

float generateCellularNoise2D(vec2 p, int size, float variant)
{
    float fsize = float(size);
    float cubeSize = 1.0 / fsize;

    /*
    if (length(p - vec2(0.0)) < 0.1) return 1.0;
    if (length(p - vec2(1.0)) < 0.1) return 1.0;
    if (length(p - vec2(1.0, 0.0)) < 0.1) return 1.0;
    */

    // in which section am I?
    ivec2 q = ivec2(
        int(floor(p.x / cubeSize)),
        int(floor(p.y / cubeSize))
    );

    // check the surroundings
    float minDist = 9999.0;
    for (int x = -1; x < 2; ++x)
    {
        for (int y = -1; y < 2; ++y)
        {
            ivec2 sampleCube = q + ivec2(x, y);

            vec2 moduloPoint = vec2(
                float((sampleCube.x + size) % size),
                float((sampleCube.y + size) % size)
            );
            vec2 randomPoint = vec2(
                random(moduloPoint.xy) + sin(moduloPoint.x + variant) * 0.1,
                random(moduloPoint.yx * 0.1) + cos(moduloPoint.y + variant) * 0.2
            );
            vec2 samplePoint = (vec2(sampleCube) + randomPoint) * cubeSize;

            minDist = min(distance(samplePoint, p), minDist);
        }
    }
    return minDist / cubeSize;
}

float generateCellularNoise3D(vec3 p, int size)
{
    float fsize = float(size);
    float cubeSize = 1.0 / fsize;

    // in which section am I?
    ivec3 q = ivec3(
        int(floor(p.x / cubeSize)),
        int(floor(p.y / cubeSize)),
        int(floor(p.z / cubeSize))
    );

    // check the surroundings
    float minDist = 9999.0;
    for (int x = -1; x < 2; ++x)
    {
        for (int y = -1; y < 2; ++y)
        {
            for (int z = -1; z < 2; ++z)
            {
                vec3 sampleCube = vec3(q + ivec3(x, y, z));

                vec3 randomPoint = vec3(
                    random(sampleCube.xy),
                    random(sampleCube.xz),
                    random(sampleCube.yz)
                );
                vec3 samplePoint = (sampleCube + randomPoint) * cubeSize;

                minDist = min(distance(samplePoint, p), minDist);

            }
        }
    }
    return minDist / cubeSize;
}

float generateCheckerboard(vec2 st)
{
    float value1 = step(fract(st.x), 0.5);
    float value2 = step(fract(st.y), 0.5);
    return min(value1, value2) + (1.0 - max(value1, value2));
}

/*
float generateSteppedSin(vec2 st, float steps)
{
    return floor(steps * sin(st.x * 3.14195)) / steps;
}

float generateSteppedPyramid(vec2 st, float steps)
{
    float value1 = generateSteppedSin(st, steps);
    float value2 = generateSteppedSin(st.yx, steps);
    return min(value1, value2);
}

float generateTriangle(vec2 st)
{
    return step(clamp(st.x - st.y, 0.0, 1.0), 0.0);
}

float generateRipple(vec2 st, float p)
{
    return 0.5 + sin(
        pow(
            pow(abs(st.s), p) + pow(abs(st.t), p),
            (1.0 / p)
        )
    ) / 2.0;
}

float generateWaves(vec2 st)
{
    float e = 2.7183;
    return (pow(e, sin(st.s) * cos(st.t)) / (e * e));
}
*/



/* Procedural bricks texture.
 */
/*
mat3 pmatBricks(vec2 st)
{
    st = wrapSt(st);
    float value1 = generateLine(st, 0.0, 0.025);
    float value2 = generateLine(st, 0.975, 1.0);
    float value3 = generateLine(st, 0.47, 0.52);

    float value4 = generateLine(st.yx, 0.2, 0.25) * generateLine(st, 0.0, 0.47);
    float value5 = generateLine(st.yx, 0.65, 0.7) * generateLine(st, 0.52, 1.0);

    float linesMask = min(1.0, value1 + value2 + value3 + value4 + value5);

    float noise1 = 0.8 + generateWhiteNoise(st) * 0.2;
    float noise2 = 0.9 + generateWhiteNoise(st) * 0.1;

    vec3 color1 = vec3(0.95, 0.95, 0.77) * noise1;
    vec3 color2 = vec3(0.67, 0.44, 0.44); // * noise2;
    vec3 color = linesMask > 0.0 ? color1 : color2;

    float height = 3.0 * (1.0 - linesMask) * noise2;

    return mat3(color, vec3(height), vec3(1.0, 0.0, 0.0));
}
*/

/*
mat3 pmatWood(vec2 st, vec3 colorA, vec3 colorB)
{
    // from https://thebookofshaders.com/edit.php#11/wood.frag
    vec2 pos = st.yx * vec2(10.0, 3.0);

    float pattern = pos.x;

    // Add noise
    pos = rotate2d(generateNoise(pos)) * pos;

    // Draw lines
    pattern = generateLines(pos, 0.5);

    return mat3(colorA * pattern, vec3(0.0, 0.0, 1.0), vec3(1.0, 0.0, 0.0));
}
*/

vec3 gammaCorrection(vec3 color)
{
    float exp = 1.0 / 2.2;
    return vec3(
        pow(color.r, exp),
        pow(color.g, exp),
        pow(color.b, exp)
    );
}

vec3 gammaCorrectionInverse(vec3 color)
{
    float exp = 2.2;
    return vec3(
        pow(color.r, exp),
        pow(color.g, exp),
        pow(color.b, exp)
    );
}

vec3 flattenColor(vec3 color, int colors)
{
    float divider = float(colors);
    return round(color * divider) / divider;
    /*
    return vec3(
        round(color.r * divider) / divider,
        round(color.g * divider) / divider,
        round(color.b * divider) / divider
    );
    */
}

/*
mat4 translationM(vec3 t)
{
    mat4 m = mat4(
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0
    );

    for (int c = 0; c < 3; ++c)
    {
        m[3][c] = t[c];
    }
    return m;
}

mat4 rotationY(float angle)
{
    float rad = angle / 180.0 * 3.14;
    float c = cos(rad);
    float s = sin(rad);

    return mat4(
        c, 0.0, -s, 0.0,
        0.0, 1.0, 0.0, 0.0,
        s, 0.0, c, 0.0,
        0.0, 0.0, 0.0, 1.0
    );
}

mat4 rotationZ(float angle)
{
    float rad = angle / 180.0 * 3.14;
    float c = cos(rad);
    float s = sin(rad);

    return mat4(
        c, s, 0.0, 0.0,
        -s, c, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0
    );
}
*/

vec3 getLightLocation(int n)
{
    int pos = n * 3;
    return texelFetch(lightsData, ivec2(pos, 0), 0).xyz;
}

vec3 getLightColor(int n)
{
    int pos = n * 3;
    return texelFetch(lightsData, ivec2(pos + 1, 0), 0).rgb;
}

float getLightRange(int n)
{
    int pos = n * 3;
    return texelFetch(lightsData, ivec2(pos + 2, 0), 0).r;
}

mat4 getObjectTrafo(int n)
{
    ivec2 r = resolveWorldLocator(n);
    int cubeNr = r.x;
    mat4 cm = cubeTrafo(cubeNr);

    //ivec2 objData = objectDataOffset(n);
    //int objLoc = int(texelFetch(sekaiData, objData, 0).r);
    int objLoc = objectDataEntry(n, 0);

    vec3 p = resolveObjectLocator(objLoc);
    mat4 om = mat4(
        vec4(1.0, 0.0, 0.0, 0.0),
        vec4(0.0, 1.0, 0.0, 0.0),
        vec4(0.0, 0.0, 1.0, 0.0),
        vec4(p, 1.0)
    );

    return cm * om;
}

mat4 getObjectInverseTrafo(int n)
{
    ivec2 r = resolveWorldLocator(n);
    int cubeNr = r.x;
    mat4 cm = cubeTrafoInverse(cubeNr);

    //ivec2 objData = objectDataOffset(n);
    //int objLoc = int(texelFetch(sekaiData, objData, 0).r);
    int objLoc = objectDataEntry(n, 0);

    vec3 p = resolveObjectLocator(objLoc);
    mat4 om = mat4(
        vec4(1.0, 0.0, 0.0, 0.0),
        vec4(0.0, 1.0, 0.0, 0.0),
        vec4(0.0, 0.0, 1.0, 0.0),
        vec4(-p, 1.0)
    );

    return cm * om;
}

vec3 refr(vec3 ray, vec3 surfaceNormal, float ior)
{
    float eta = 1.0 / ior;
    float cosi = clamp(dot(surfaceNormal, ray), -1.0, 1.0);

    if (cosi > 0.0)
    {
        // exiting material, flipping around
        surfaceNormal *= -1.0;
        ior = 1.0 / ior;
        cosi = -cosi;
    }
    else
    {
        // entering material
    }
    float k = 1.0 - eta * eta * (1.0 - (cosi * cosi));
    if (k < 0.0)
    {
        // no refraction possible (total internal reflection)
        return vec3(0.0);
    }
    else
    {
        vec3 t1 = ray * eta;
        float t2 = cosi * eta + sqrt(k);
        return t1 - surfaceNormal * t2;
    }
}

/* Transforms a world-space point into object space.
 */
vec3 transformPoint(vec3 p, int obj)
{
    mat4 m = getObjectInverseTrafo(obj);
    return (m * vec4(p, 1.0)).xyz;    
}

/* Transforms a surface normal in object space into world space.
 */
vec3 transformNormalOW(vec3 normal, int obj)
{
    mat4 trafo = getObjectTrafo(obj);
    vec3 objLocW = (trafo * vec4(0.0, 0.0, 0.0, 1.0)).xyz;
    vec3 surfaceLocW = (trafo * vec4(normal, 1.0)).xyz;
    return normalize(surfaceLocW - objLocW);
}

/*
float dot2(vec2 v)
{
    return dot(v,v);
}

float dot2(vec3 v)
{
    return dot(v,v);
}

float ndot(vec2 a, vec2 b)
{
    return a.x * b.x - a.y * b.y;
}
*/

float sdf(int obj, vec3 p)
{
    /* SDF are courtesy of I~nigo Quilez:
     * https://iquilezles.org/articles/distfunctions/
     */

    float dist = 0.0;
    int type = BoxType; //getObjectType(obj);
    float radius = 1.0; //getObjectRadius(obj);

    if (type == BoxType)
    {
        vec3 halfSides = vec3(radius) * 0.5;
        vec3 pt = p - vec3(0.0);
        vec3 q = abs(pt) - halfSides;
        dist = length(max(q, 0.0)) - min(0.0, max(max(q.x, q.y), q.z));
        // create a hollow shell
        //return abs(dist) - 0.05;
    }
    else if (type == PlaneType)
    {
        dist = p.y;
    }
    else if (type == SphereType)
    {
        dist = length(p - vec3(0.0, 0.0, 0.0)) - radius;
    }
    else if (type == LensType)
    {
        float s1 = length(p - vec3(-radius / 2.0, 0.0, 0.0)) - radius;
        float s2 = length(p - vec3(+radius / 2.0, 0.0, 0.0)) - radius;
        dist = max(s1, s2);
    }

    /*
    case TorusType:
        vec2 q = vec2(length(p.xz) - t.x, p.y);
        return length(q) - t.y;

    case ConeType:
        vec2 q = h * vec2(c.x / c.y, -1.0);
            
        vec2 w = vec2(length(p.xz), p.y);
        vec2 a = w - q * clamp(dot(w, q) / dot(q, q), 0.0, 1.0);
        vec2 b = w - q * vec2(clamp(w.x / q.x, 0.0, 1.0), 1.0);
        float k = sign(q.y);
        float d = min(dot(a, a),dot(b, b));
        float s = max(k * (w.x * q.y - w.y * q.x), k * (w.y - q.y));
        return sqrt(d) * sign(s);

    case Triangle:
        vec3 ba = b - a;
        vec3 pa = p - a;
        vec3 cb = c - b;
        vec3 pb = p - b;
        vec3 ac = a - c;
        vec3 pc = p - c;
        vec3 nor = cross(ba, ac);

        return sqrt(
            (sign(dot(cross(ba, nor), pa)) +
             sign(dot(cross(cb, nor), pb)) +
             sign(dot(cross(ac, nor), pc)) < 2.0)
            ?
            min(min(
                dot2(ba * clamp(dot(ba, pa)/dot2(ba), 0.0, 1.0) - pa),
            dot2(cb * clamp(dot(cb, pb) / dot2(cb), 0.0, 1.0) - pb)),
            dot2(ac * clamp(dot(ac, pc) / dot2(ac), 0.0, 1.0) - pc))
            :
            dot(nor, pa) * dot(nor, pa) / dot2(nor));
    */

    else
    {
        dist = 9999.0;
    }

    return dist;
}

/* Processes a set of TASM instructions to generate a texture.
 */
mat3 processTasm(int program, vec2 st, vec3 p, float travelDist)
{
    // Since the GPU is quite limited on what it can do, implementing the
    // TASM instruction set might be too heavy for it. Therefore, all TASM
    // instructions are broken down into microcode defined by the TASM firmware.
    // The GPU processes the microcode only.

    bool programTooLong = false;
    bool stackOutOfBounds = false;

    int batchSize = 0;
    int srcPointer = 0;
    int destPointer = 0;
    int offsets = 0;
    int srcOffset = 0;
    int destOffset = 0;

    float workParam1 = 0.0;
    float workParam2 = 0.0;
    float workParam3 = 0.0;
    float workParam4 = 0.0;
    
    float v = 0.0;
    vec3 resultVec;

    float instructionSize = 0.0;
    int op = 0;
    int opCode = 0;
    int ri;

    vec4 instruction;
    vec4 microCodeCopyReg1;
    vec4 microCodeTest;
    vec4 microCodeBinOp;
    vec4 microCodeGenOp;
    vec4 microCodeAddReg;
    vec4 microCodeCopyReg2;

    // initialize
    tasmRegisters[REG_PC] = 0.0;
    tasmRegisters[REG_SP] = float(REG_STACK);

    tasmRegisters[REG_COLOR_R] = 1.0;
    tasmRegisters[REG_COLOR_G] = 0.0;
    tasmRegisters[REG_COLOR_B] = 0.0;

    tasmRegisters[REG_NORMAL_X] = 0.0;
    tasmRegisters[REG_NORMAL_Y] = 0.0;
    tasmRegisters[REG_NORMAL_Z] = 1.0;

    tasmRegisters[REG_ATTRIB_1] = 1.0;
    tasmRegisters[REG_ATTRIB_2] = 0.0;
    tasmRegisters[REG_ATTRIB_3] = 0.0;

    tasmRegisters[REG_ENV_TIMEMS] = float(timems);
    tasmRegisters[REG_ENV_ST_X] = st.x;
    tasmRegisters[REG_ENV_ST_Y] = st.y;
    tasmRegisters[REG_ENV_RAY_DISTANCE] = travelDist;

    for (int i = 0; i < 96; ++i)
    {
        programTooLong = i == 95;
        stackOutOfBounds = tasmRegisters[REG_SP] < float(REG_STACK) ||
                           tasmRegisters[REG_SP] >= float(REG_USER);

        if (tasmRegisters[REG_PC] < 0.0 || programTooLong || stackOutOfBounds)
        {
            // exit
            break;
        }

        instruction = texelFetch(tasmData, ivec2(int(tasmRegisters[REG_PC]), program), 0);

        opCode = int(instruction.r);
        instructionSize = instruction.g;
        tasmRegisters[REG_PARAM1] = instruction.b;
        tasmRegisters[REG_PARAM2] = instruction.a;

        // caching these appears to add too much overhead and memory spilling, and we're generally
        // better off without caching
        microCodeCopyReg1 = texelFetch(tasmData, ivec2(0, 8000 + opCode), 0);
        microCodeTest = texelFetch(tasmData, ivec2(1, 8000 + opCode), 0);
        microCodeBinOp = texelFetch(tasmData, ivec2(2, 8000 + opCode), 0);
        microCodeGenOp = texelFetch(tasmData, ivec2(3, 8000 + opCode), 0);
        microCodeAddReg = texelFetch(tasmData, ivec2(4, 8000 + opCode), 0);
        microCodeCopyReg2 = texelFetch(tasmData, ivec2(5, 8000 + opCode), 0);

        // advance program counter
        tasmRegisters[REG_PC] += instructionSize;

        // copy n registers from *source to *dest (avoid with batchSize = 0)
        batchSize = int(microCodeCopyReg1.r);
        srcPointer = int(tasmRegisters[int(microCodeCopyReg1.g)]);
        destPointer = int(tasmRegisters[int(microCodeCopyReg1.b)]);
        offsets = int(microCodeCopyReg1.a);
        srcOffset = (offsets >> 4) - 8;
        destOffset = (offsets & 15) - 8;
        for (ri = 0; ri < 3; ++ri)
        {
            tasmRegisters[destPointer + ri + destOffset] = ri < batchSize ? tasmRegisters[srcPointer + ri + srcOffset]
                                                                          : tasmRegisters[destPointer + ri + destOffset];
        }

        // test
        op = int(microCodeTest.r);
        if (op > 0)
        {
            workParam1 = tasmRegisters[int(tasmRegisters[REG_SP]) - 2];
            workParam2 = tasmRegisters[int(tasmRegisters[REG_SP]) - 1];
            tasmRegisters[REG_PC] = (op == 1 && workParam1 < workParam2) ||
                                (op == 2 && workParam1 <= workParam2) ||
                                (op == 3 && abs(workParam1 - workParam2) < 0.0001) ||
                                (op == 4 && workParam1 > workParam2) ||
                                (op == 5 && workParam1 >= workParam2)
                              ? tasmRegisters[REG_PC]
                              : tasmRegisters[REG_PARAM1];
        }

        // binop
        op = intr(microCodeBinOp.r);
        if (op > 0)
        {
            batchSize = intr(microCodeBinOp.g);
            for (ri = 0; ri < 3; ++ri)
            {
                workParam1 = tasmRegisters[intr(tasmRegisters[REG_SP]) - 2 * batchSize + ri];
                workParam2 = tasmRegisters[intr(tasmRegisters[REG_SP]) - batchSize + ri];
                v = (op == 1) ? workParam1 + workParam2 : v;
                v = (op == 2) ? workParam1 - workParam2 : v;
                v = (op == 3) ? workParam1 * workParam2 : v;
                v = (op == 4) ? workParam1 / workParam2 : v;
                v = (op == 5) ? min(workParam1, workParam2) : v;
                v = (op == 6) ? max(workParam1, workParam2) : v;
                v = (op == 7) ? workParam1 + exp(workParam2) : v;
                tasmRegisters[intr(tasmRegisters[REG_SP]) - 2 * batchSize + ri] = ri < batchSize ? v
                                                                                                 : workParam1;
            }
        }

        // gen
        op = intr(microCodeGenOp.r);
        if (op == 5)
        {
            workParam1 = tasmRegisters[intr(tasmRegisters[REG_SP]) - 1];
            workParam2 = tasmRegisters[intr(tasmRegisters[REG_SP]) - 2];
            workParam3 = tasmRegisters[intr(tasmRegisters[REG_SP]) - 3];
            workParam4 = tasmRegisters[intr(tasmRegisters[REG_SP]) - 4];

            resultVec = generateBumpNormal(workParam4, workParam3, workParam2, workParam1);
            
            tasmRegisters[intr(tasmRegisters[REG_SP]) - 4] = resultVec.x;
            tasmRegisters[intr(tasmRegisters[REG_SP]) - 3] = resultVec.y;
            tasmRegisters[intr(tasmRegisters[REG_SP]) - 2] = resultVec.z;
        }
        else if (op == 6)
        {
            workParam1 = tasmRegisters[intr(tasmRegisters[REG_SP]) - 1];
            workParam2 = tasmRegisters[intr(tasmRegisters[REG_SP]) - 2];
            workParam3 = tasmRegisters[intr(tasmRegisters[REG_SP]) - 3];

            resultVec = vec3(generateMipMap(vec2(workParam3, workParam2), int(workParam1)), 0.0);
            
            tasmRegisters[intr(tasmRegisters[REG_SP]) - 3] = resultVec.x;
            tasmRegisters[intr(tasmRegisters[REG_SP]) - 2] = resultVec.y;

        }
        else if (op > 0)
        {
            workParam1 = tasmRegisters[intr(tasmRegisters[REG_SP]) - 1];
            workParam2 = tasmRegisters[intr(tasmRegisters[REG_SP]) - 2];
            workParam3 = tasmRegisters[intr(tasmRegisters[REG_SP]) - 3];
            workParam4 = tasmRegisters[intr(tasmRegisters[REG_SP]) - 4];

            tasmRegisters[REG_PARAM1] = (op == 1 ? generateLine(vec2(workParam4, workParam3), workParam2, workParam1) : 0.0) +
                                        (op == 2 ? generateCheckerboard(vec2(workParam2, workParam1)) : 0.0) + 
                                        (op == 3 ? generateWhiteNoise(vec2(workParam2, workParam1)) : 0.0) + 
                                        (op == 4 ? generateCellularNoise2D(vec2(workParam4, workParam3), int(workParam2), workParam1) : 0.0);
        }

        // add const value to a register (avoid with void pointer)
        srcPointer = int(tasmRegisters[int(microCodeAddReg.r)]);
        tasmRegisters[srcPointer] += microCodeAddReg.g;

        // copy n registers from *source to *dest (avoid with batch size = 0)
        batchSize = intr(microCodeCopyReg2.r);
        srcPointer = intr(tasmRegisters[int(microCodeCopyReg2.g)]);
        destPointer = intr(tasmRegisters[int(microCodeCopyReg2.b)]);
        offsets = int(microCodeCopyReg2.a);
        srcOffset = (offsets >> 4) - 8;
        destOffset = (offsets & 15) - 8;
        for (ri = 0; ri < 3; ++ri)
        {
            tasmRegisters[destPointer + ri + destOffset] = ri < batchSize ? tasmRegisters[srcPointer + ri + srcOffset]
                                                                          : tasmRegisters[destPointer + ri + destOffset];
        }
    }

    return mat3(
        vec3(
            programTooLong || stackOutOfBounds ? 1.0 : tasmRegisters[REG_COLOR_R],
            programTooLong ? 0.0 : stackOutOfBounds ? 1.0 : tasmRegisters[REG_COLOR_G],
            programTooLong ? 1.0 : stackOutOfBounds ? 0.0 : tasmRegisters[REG_COLOR_B]
        ),
        vec3(tasmRegisters[REG_NORMAL_X], tasmRegisters[REG_NORMAL_Y], tasmRegisters[REG_NORMAL_Z]),
        vec3(tasmRegisters[REG_ATTRIB_1], tasmRegisters[REG_ATTRIB_2], tasmRegisters[REG_ATTRIB_3])
    );
}

vec3 getSurfaceNormal(int obj, vec3 p)
{
    // p is in object space

    int type = BoxType; //getObjectType(obj);
    if (type == PlaneType)
    {
        return vec3(0.0, 1.0, 0.0);
    }
    else if (type == SphereType)
    {
        return normalize(p);
    }
    else if (type == BoxType)
    {
        vec3[3] normals = vec3[3](
            vec3(1.0, 0.0, 0.0),
            vec3(0.0, 1.0, 0.0),
            vec3(0.0, 0.0, 1.0)
        );

        float maxDot = 0.0;
        vec3 n;
        for (int i = 0; i < 3; ++i)
        {
            float dp = dot(p, normals[i]);
            if (abs(dp) > abs(maxDot))
            {
                n = normals[i];
                maxDot = dp;
            }
        }
        return n * sign(maxDot);
    }

    float epsilon = 0.00001;
    return normalize(
        vec3(
            sdf(obj, p + vec3(epsilon, 0.0, 0.0)) - sdf(obj, p + vec3(-epsilon, 0.0, 0.0)),
            sdf(obj, p + vec3(0.0, epsilon, 0.0)) - sdf(obj, p + vec3(0.0, -epsilon, 0.0)),
            sdf(obj, p + vec3(0.0, 0.0, epsilon)) - sdf(obj, p + vec3(0.0, 0.0, -epsilon))
        )
    );
}

/* Returns the surface material at the given location as a mat3:
 *
 * - vec3: color
 * - vec3: normal vector (z pointing upwards)
 * - vec3: roughness, ior, volumetric
 */
mat3 getObjectMaterial(int obj, vec3 p, float travelDist)
{
    int materialId = objectDataEntry(obj, 2);
    vec2 st = p.xy;

    // position texture on cube
    vec3 n = getSurfaceNormal(obj, p);
    vec3 p2 = abs(n.y) > 0.0 ? n.zxy : n.zyx;
    float dp = dot(n, p2);
    vec3 axis1 = normalize(p2 - dp * n);
    vec3 axis2 = normalize(cross(n, axis1));
    float x = dot(p, axis1);
    float y = dot(p, axis2);
    st = 0.5 + vec2(x, y);

    /*
    float steps = 32.0;
    float maxDist = 100.0;
    float stepWidth = maxDist / steps;
    float section = floor(travelDist / stepWidth);
    float units = pow(2.0, min(31.0, max(0.0, steps - section)));
    float halfUnit = (1.0 / units) / 2.0;

    mat3 m1 = processTasm(materialId, clamp(st - vec2(halfUnit, 0.0), 0.0, 1.0), p, travelDist);
    mat3 m2 = processTasm(materialId, clamp(st + vec2(0.0, halfUnit), 0.0, 1.0), p, travelDist);
    //mat3 m3 = processTasm(materialId, st - vec2(0.0, halfUnit), p, travelDist);
    //mat3 m4 = processTasm(materialId, st + vec2(0.0, halfUnit), p, travelDist);

    return mat3(
        (m1[0] + m2[0]) / 2.0,
        (m1[1] + m2[1]) / 2.0,
        m1[2]
    );
    */

    return enableTasm ? processTasm(materialId, st, p, travelDist)
                      : mat3(
                            vec3(1.0),
                            vec3(0.0, 0.0, 1.0),
                            vec3(1.0, 0.0, 0.0)
                        );

    /*
    return mat3(
        vec3(1.0),
        vec3(0.0, 0.0, 1.0),
        vec3(1.0, 0.0, 0.0)
    );
    */
}

/* Returns if an object may be hit by a ray.
 */
bool mayHitObject(vec3 origin, vec3 rayDirection)
{
    // vectors are expected to be in object space already

    vec3 hit = hitAabb(origin, rayDirection);
    return hit.z > 0.001 &&
           (hit.x >= 0.0 ||
            hit.y >= 0.0);
}

/* Loads the objects hit by a ray in a cube onto the objectsOnRay array.
 */
void findObjectsInCube(int cube, vec3 origin, vec3 rayDirection, float maxDist)
{
    ivec2 offset = cubeDataOffset(cube);
    int numObjects = intr(texelFetch(sekaiData, offset, 0).r);
    vec4 pattern = texelFetch(sekaiData, offset + 1, 0);
    int patternHi = intr(pattern.r);
    int patternHiMid = intr(pattern.g);
    int patternLoMid = intr(pattern.b);
    int patternLo = intr(pattern.a);


    for (int objLoc = 0; objLoc < 64; ++objLoc)
    {
        if (objLoc >= numObjects)
        {
            break;
        }

        /*
        bool haveObject = false;
        if (objLoc < 16)
        {
            haveObject = (patternLo & (1 << objLoc)) > 0;
        }
        else if (objLoc < 32)
        {
            haveObject = (patternLoMid & (1 << (objLoc - 16))) > 0;
        }
        else if (objLoc < 48)
        {
            haveObject = (patternHiMid & (1 << (objLoc - 32))) > 0;
        }
        else
        {
            haveObject = (patternHi & (1 << (objLoc - 48))) > 0;
        }

        if (! haveObject)
        {
            //continue;
        }
        */

        //int objLoc = objNr; //intr(texelFetch(sekaiData, ivec2(10 + objNr * 10, cube), 0).r);
        int obj = makeWorldLocator(cube, objLoc);

        // convert origin and ray into object space
        vec3 rayP = origin + rayDirection;
        vec3 rayPT = transformPoint(rayP, obj);

        vec3 originT = transformPoint(origin, obj);
        vec3 rayDirectionT = rayPT - originT;

        if (mayHitObject(originT, rayDirectionT))
        {
            float dist = sdf(obj, originT);
            if (dist < maxDist && numObjectsOnRay < objectsOnRay.length())
            {
                objectsOnRay[numObjectsOnRay] = obj;
                ++numObjectsOnRay;
            }
        }
    }
}

/* Loads the objects hit by a ray onto the objectsOnRay array.
 */
void findObjectsOnRay(vec3 origin, vec3 rayDirection, float maxDist)
{
    // find the cubes on the ray
    // test the objects in the cube
    // load object IDs into an array

    // perform DDA ray casting to find the cubes on the ray

    numObjectsOnRay = 0;

    float cubeSize = 4.0;
    vec3 begin = vec3(0.0);
    vec3 p = origin;

    int originCube = makeCubeLocator(p);
    findObjectsInCube(originCube, origin, rayDirection, maxDist);

    vec3 scale = vec3(
        rayDirection.x != 0.0 ? cubeSize / abs(rayDirection.x) : 9999.0,
        rayDirection.y != 0.0 ? cubeSize / abs(rayDirection.y) : 9999.0,
        rayDirection.z != 0.0 ? cubeSize / abs(rayDirection.z) : 9999.0
    );

    vec3 grid = begin + cubeSize * floor((p - begin) / cubeSize);

    grid += vec3(
        rayDirection.x > 0.0 ? cubeSize : 0.0,
        rayDirection.y > 0.0 ? cubeSize : 0.0,
        rayDirection.z > 0.0 ? cubeSize : 0.0
    );

    vec3 dist = abs(grid - p);

    for (int i = 0; i < marchingDepth; ++i)
    {
        if (numObjectsOnRay > 0)
        {
            break;
        }

        vec3 rayLength = dist * scale;
        bool advanceX = rayLength.x <= rayLength.y && rayLength.x <= rayLength.z;
        bool advanceY = rayLength.y <= rayLength.x && rayLength.y <= rayLength.z;
        bool advanceZ = rayLength.z <= rayLength.x && rayLength.z <= rayLength.y;

        vec3 advanceVec = vec3(
            advanceX ? sign(rayDirection.x) * cubeSize : 0.0,
            advanceY ? sign(rayDirection.y) * cubeSize : 0.0,
            advanceZ ? sign(rayDirection.z) * cubeSize : 0.0
        );

        p += advanceVec;
        dist += vec3(
            advanceX ? cubeSize : 0.0,
            advanceY ? cubeSize : 0.0,
            advanceZ ? cubeSize : 0.0
        );

        int cube = makeCubeLocator(p);
        findObjectsInCube(cube, origin, rayDirection, maxDist);
    }
}

/*
void findObjectsOnRay2(vec3 origin, vec3 rayDirection, float maxDist)
{
    // find the cubes on the ray
    // test the objects in the cube
    // load object IDs into an array

    // perform DDA ray casting to find the cubes on the ray

    numObjectsOnRay = 0;

    float cubeSize = 4.0;
    vec3 begin = vec3(0.0);
    vec3 p = origin;

    int originCube = makeCubeLocator(p);
    findObjectsInCube(originCube, origin, rayDirection, maxDist);

    vec3 scale = vec3(
        rayDirection.x != 0.0 ? cubeSize / abs(rayDirection.x) : 9999.0,
        rayDirection.y != 0.0 ? cubeSize / abs(rayDirection.y) : 9999.0,
        rayDirection.z != 0.0 ? cubeSize / abs(rayDirection.z) : 9999.0
    );

    vec3 grid = begin + cubeSize * floor((p - begin) / cubeSize);

    if (rayDirection.x > 0.0)
    {
        grid.x += cubeSize;
    }
    if (rayDirection.y > 0.0)
    {
        grid.y += cubeSize;
    }
    if (rayDirection.z > 0.0)
    {
        grid.z += cubeSize;
    }

    vec3 dist = abs(grid - p);

    int level = 0;
    int parentNr = makeSuperCubeLocator(p, level + 1);
    bool advance = true;
    for (int i = 0; i < 20; ++i)
    {
        if (numObjectsOnRay > 0)
        {
            break;
        }

        cubeSize = float(4 << level);

        if (advance)
        {
            scale = vec3(
                rayDirection.x != 0.0 ? cubeSize / abs(rayDirection.x) : 9999.0,
                rayDirection.y != 0.0 ? cubeSize / abs(rayDirection.y) : 9999.0,
                rayDirection.z != 0.0 ? cubeSize / abs(rayDirection.z) : 9999.0
            );

            vec3 rayLength = dist * scale;

            if (rayLength.x <= rayLength.y && rayLength.x <= rayLength.z)
            {
                p.x += sign(rayDirection.x) * cubeSize;
                dist.x += cubeSize;
            }
            else if (rayLength.y <= rayLength.x && rayLength.y <= rayLength.z)
            {
                p.y += sign(rayDirection.y) * cubeSize;
                dist.y += cubeSize;
            }
            else if (rayLength.z <= rayLength.x && rayLength.z <= rayLength.y)
            {
                p.z += sign(rayDirection.z) * cubeSize;
                dist.z += cubeSize;
            }
        }

        int superCubeNr = makeSuperCubeLocator(p, level);
        int currentParentNr = makeSuperCubeLocator(p, level + 1);

        if (parentNr != currentParentNr)
        {
            // parent super cube changed -> ascend one layer
            if (level < 3)
            {
                ++level;
                advance = false;
            }
            parentNr = makeSuperCubeLocator(p, level + 1);
        }
        else if (level == 0)
        {
            // find objects on lowest level
            findObjectsInCube(superCubeNr, origin, rayDirection, maxDist);
            advance = true;
        }
        else if (isSuperCubeEmpty(superCubeNr, level))
        {
            // empty -> continue
            advance = true;
        }
        else
        {
            // super cube not empty -> descend one layer
            parentNr = superCubeNr;
            --level;
            advance = false;
        }
    }
}
*/

void discardNthObjectOnRay(int n)
{
    objectsOnRay[n] = objectsOnRay[numObjectsOnRay - 1];
    --numObjectsOnRay;
}

void discardObjectOnRay(int obj)
{
    for (int i = 0; i < numObjectsOnRay; ++i)
    {
        if (objectsOnRay[i] == obj && numObjectsOnRay > 1)
        {
            discardNthObjectOnRay(i);
            break;
        }
    }
}

/* Returns the object nearest to the given point in world coordinates
 * and the distance to it.
 */
vec2 nearestObject(vec3 p)
{
    int foundObject = -1;
    float d = 9999.0;    

    for (int i = 0; i < 32; ++i)
    {
        if (i == numObjectsOnRay)
        {
            break;
        }

        int obj = objectsOnRay[i];

        vec3 pT = transformPoint(p, obj);
        float dist = sdf(obj, pT);
        if (dist < d)
        {
            d = dist;
            foundObject = obj;
        }
    }

    return vec2(float(foundObject), d);
}

vec2 nearestBox(vec3 origin, vec3 rayDirection)
{
    int foundObject = -1;
    float d = 9999.0;    

    for (int i = 0; i < 16; ++i)
    {
        if (i == numObjectsOnRay)
        {
            break;
        }

        int obj = objectsOnRay[i];

        // convert origin and ray into object space
        vec3 rayP = origin + rayDirection;
        vec3 rayPT = transformPoint(rayP, obj);

        vec3 originT = transformPoint(origin, obj);
        vec3 rayDirectionT = rayPT - originT;

        vec3 hit = hitAabb(originT, rayDirectionT);
        if (hit.z > 0.0001)
        {
            float dist = hit.x < 0.0 ? hit.y
                                     : hit.y < 0.0 ? 9999.0
                                                   : min(hit.x, hit.y);
            if (dist < d)
            {
                d = dist;
                foundObject = obj;
            }
        }
    }

    return vec2(float(foundObject), d);
}

/*
vec2 rayMarch(vec3 origin, vec3 rayDirection, bool insideObject, float maxDistance, float accuracy)
{
    // this is an essential optimization to reduce the number of objects to check
    findObjectsOnRay(origin, rayDirection, maxDistance);
    objectsOnRayCount = numObjectsOnRay;

    float distance = 0.0;
    for (int i = 0; i < marchingDepth; ++i)
    {
        ++sdfCount;
        if (distance > maxDistance)
        {
            break;
        }

        vec3 checkPoint = origin + rayDirection * distance;
        vec2 objectAndDistance = nearestObject(checkPoint);
        int obj = intr(objectAndDistance.x);
        float safeDist = objectAndDistance.y;

        if (insideObject)
        {
            safeDist *= -1.0;
        }

        if (safeDist > accuracy)
        {
            // no hit
            distance += safeDist;
        }
        else
        {
            return vec2(objectAndDistance.x, distance);
        }
    }
    return vec2(-1.0, 9999.0);
}
*/

vec2 rayWarp(vec3 origin, vec3 rayDirection, bool insideObject, float maxDistance, float accuracy)
{
    findObjectsOnRay(origin, rayDirection, maxDistance);
    objectsOnRayCount = numObjectsOnRay;

    vec2 objectAndDistance = nearestBox(origin, rayDirection);
    return objectAndDistance;
}

vec3 simplePhongShading(vec3 checkPoint)
{
    vec3 lighting = vec3(0.0);

    for (int i = 0; i < 3; ++i)
    {
        if (i == numLights)
        {
            break;
        }

        vec3 lightLoc = getLightLocation(i);
        vec3 lightCol = getLightColor(i);
        float lightRange = getLightRange(i);

        vec3 directionToLight = normalize(lightLoc - checkPoint);
        float lightDistance = length(checkPoint - lightLoc);

        // light attenuation based on distance and strength of the light source
        float attenuation = clamp(1.0 - lightDistance / lightRange, 0.0, 1.0);
        attenuation *= attenuation;
        vec3 attenuatedLight = lightCol * attenuation;

        // diffuse light
        float diffuseImpact = 1.0;
        vec3 diffuse = attenuatedLight * diffuseImpact;

        lighting += diffuse;
    }

    return lighting.rgb;
}

vec3 phongShading(vec3 origin, vec3 checkPoint, vec3 surfaceNormal, float roughness)
{
    // Phong shading: lighting = ambient + diffuse + specular
    //                color = modelColor * lighting

    vec3 viewDirection = normalize(origin - checkPoint);
    vec3 lighting = vec3(0.2);
    float shininess = (1.0 - roughness) * 64.0;

    // is the sky visible (the direction vector should depend on the time of day)
    /* This is too slow!
    vec3 v = reflect(-viewDirection, surfaceNormal);
    if (dot(v, surfaceNormal) > 0.0)
    {
        float travelDistToSky = rayWarp(checkPoint + v * 0.1, v, false, 9999.0, 0.0001).y;
        if (travelDistToSky > 1000.0)
        {
            // ambient light
            lighting += vec3(0.5);
        }
    }
    */

    for (int i = 0; i < 3; ++i)
    {
        if (i == numLights)
        {
            break;
        }

        vec3 lightLoc = getLightLocation(i);
        vec3 lightCol = getLightColor(i);
        float lightRange = getLightRange(i);

        vec3 directionToLight = normalize(lightLoc - checkPoint);
        float lightDistance = length(checkPoint - lightLoc);

        float impact = dot(directionToLight, surfaceNormal);

        // does the light reach?
        if (impact <= 0.001)
        {
            // nope
            continue;
        }
        if (enableShadows)
        {
            float travelDist = rayWarp(checkPoint + directionToLight * 0.1, directionToLight, false, lightDistance, 0.0001).y;
            if (travelDist < lightDistance)
            {
                // nope
                continue;
            }
        }

        // light attenuation based on distance and strength of the light source
        float attenuation = clamp(1.0 - lightDistance / lightRange, 0.0, 1.0);
        attenuation *= attenuation;
        vec3 attenuatedLight = lightCol * attenuation;

        // diffuse light
        vec3 diffuse = attenuatedLight * impact;

        // specular highlight (Blinn-Phong)
        vec3 halfDirection = normalize(directionToLight + viewDirection);
        float specularStrength = pow(max(0.0, dot(surfaceNormal, halfDirection)), shininess) * 0.5;
        vec3 specular = attenuatedLight * specularStrength;

        lighting += diffuse + specular;
    }

    return lighting.rgb;
}

/* Returns the color at the given screen pixel plus the ID of the object that was
 * hit in the a component. The object ID is added to the amount of traces multiplied
 * by 1000.
 */
vec4 shootRayThroughScreen(vec2 uv, vec3 origin, float aspect)
{
    // transform the camera location and orientation
    vec3 currentOrigin = (cameraTrafo * vec4(origin, 1.0)).xyz;
    vec3 screenPoint = (cameraTrafo * vec4(uv.x, uv.y / aspect, 1.0, 1.0)).xyz;

    int currentObject = -1;

    // shoot a ray from origin onto the near Plain (screen)
    vec3 rayDirection = normalize(screenPoint - currentOrigin);

    float travelDistance = 0.0;

    vec3 color = vec3(1.0);
    vec3 light = vec3(0.0);
    vec3 volumetricColor = vec3(0.8);
    vec3 volumetricLight = vec3(0.0);
    float volumetricDensity = 0.0;

    bool insideObject = false;

    int traceCount = 0;
    for (; traceCount < 8; ++traceCount)
    {
        if (traceCount == tracingDepth)
        {
            break;
        }

        vec2 objectAndDist = rayWarp(currentOrigin, rayDirection, insideObject, 9999.0, 0.0001);
        //vec2 objectAndDist = rayMarch(currentOrigin, rayDirection, insideObject, 50.0, 0.0001);
        int obj = intr(objectAndDist.x);
        float dist = objectAndDist.y;
        
        travelDistance += dist;
        currentObject = obj;

        if (obj >= 0 && dist < 10000.0)
        {
            // hit something
            vec3 checkPoint = currentOrigin + rayDirection * dist;
            vec3 checkPointT = transformPoint(checkPoint, obj);

            mat3 materialData = getObjectMaterial(obj, checkPointT, travelDistance);
            //mat3 materialData = mat3(vec3(1.0), vec3(0.0, 0.0, 1.0), vec3(1.0, 0.0, 0.0));
            vec3 materialColor = materialData[0];
            float roughness = materialData[2].x;
            float reflectivity = 1.0 - roughness;
            float ior = materialData[2].y;
            float volumetric = materialData[2].z;

            vec3 surfaceNormal = getSurfaceNormal(obj, checkPointT);
            //vec3 surfaceNormal = transformNormalOW(surfaceNormalT, obj);
            mat4 surfaceTrafo = createSurfaceTrafo(surfaceNormal);
            vec3 bumpNormalM = materialData[1];
            vec3 bumpNormalT = (surfaceTrafo * vec4(bumpNormalM, 1.0)).xyz;
            /*
            if (pathTracingDepth > 0)
            {
                bumpNormalT += roughness * (-0.5 + seededRandom(checkPointT.st) * 1.0);
            }
            */
            vec3 bumpNormal = transformNormalOW(bumpNormalT, obj);
           
            // for debugging: show normals
            //materialColor = vec3(0.5) + bumpNormal * 0.5;

            if (volumetric < 0.1 && ior < 0.001)
            {
                vec3 lightIntensity = phongShading(currentOrigin, checkPoint, bumpNormal, roughness);
                light += lightIntensity;
                color *= materialColor;
                color *= lightIntensity;
            }
            
            // measure volumetrics
            for (int i = 0; i < 100; ++i)
            {
                if (float(i) * 0.1 > dist)
                {
                    break;
                }
                vec3 samplePoint = currentOrigin + rayDirection * float(i) * 0.1;
                float f = float((timems / 100000) % 500);
                float v = max(0.0, 8.0 - samplePoint.y) * fogDensity * (1.0 - generateCellularNoise3D(samplePoint / 16.0 + f * vec3(0.0, -0.005, 0.0), 20));
                volumetricDensity += v;
                if (v > 0.0001)
                {
                    //volumetricColor *= 0.9 * simplePhongShading(samplePoint);
                    //volumetricLight += exp(-v) * simplePhongShading(samplePoint) * vec3(0.01);
                }
            }

            if (ior > 0.01)
            {
                // we're not finished yet - refract the ray and enter or exit the object
                vec3 refractedRay = refr(rayDirection, bumpNormal, ior);
                if (length(refractedRay) < 0.0001)
                {
                    // total internal reflection
                    rayDirection = normalize(reflect(rayDirection, bumpNormal));
                }
                else
                {
                    insideObject = ! insideObject;
                    rayDirection = normalize(refractedRay);
                }
                currentOrigin = checkPoint + rayDirection * 0.01;
                //light *= 0.5;
            }
            else if (reflectivity > 0.1)
            {
                // we're not finished yet - reflect the ray
                float fresnel = pow(clamp(1.0 - dot(bumpNormal, rayDirection * -1.0), 0.5, 1.0), 1.0);
                //checkPoint = currentOrigin + rayDirection * (dist - 0.5);
                rayDirection = reflect(rayDirection, bumpNormal);
                // epsilon must be small for corners, or you'll get reflected far into another block
                //currentOrigin = checkPoint; //bumpNormal * 0.1 + /* bump off the surface a bit */
                currentOrigin = checkPoint + rayDirection * 0.001;
                //light *= fresnel * reflectivity;
            }
            else if (volumetric > 0.1)
            {
                // convert origin and ray into object space
                vec3 rayP = currentOrigin + rayDirection;
                vec3 rayPT = transformPoint(rayP, obj);

                vec3 originT = transformPoint(currentOrigin, obj);
                vec3 rayDirectionT = rayPT - originT;

                vec3 checkPointT = transformPoint(checkPoint, obj);
                light *= distance(checkPointT, vec3(0.0)) / 1.0;

                vec3 entryExit = hitAabb(originT, rayDirectionT);
                vec3 entryPoint = currentOrigin + rayDirection * entryExit.s;
                vec3 entryPointT = originT + rayDirectionT * entryExit.s; // + vec3(0.1, 0.1, 0.1);
                vec3 exitPointT = originT + rayDirectionT * entryExit.t;
                float len = entryExit.t - entryExit.s;

                float sdfDist = 0.0;
                for (int i = 0; i < 50; ++i)
                {
                    vec3 samplePointT = entryPointT + rayDirectionT * sdfDist;
                    float s1 = length(samplePointT - vec3(-0.5 / 2.0, 0.0, 0.0)) - 0.5;
                    float s2 = length(samplePointT - vec3(+0.5 / 2.0, 0.0, 0.0)) - 0.5;
                    float d = max(s1, s2);
                    if (d < 0.001)
                    {
                        vec3 li = phongShading(currentOrigin, entryPoint + rayDirection * d, surfaceNormal, roughness);
                        light += li;
                        color *= vec3(1.0, 0.0, 0.0);
                        color *= li;
                        travelDistance += d;
                        break;
                    }
                    else if (d > 2.0)
                    {
                        currentOrigin = checkPoint + (len + 0.01) * rayDirection;
                        travelDistance += len + 0.01;
                    }
                    else
                    {
                        sdfDist += d;
                    }
                }

                /*
                volumetricColor *= vec3(0.3, 0.9, 0.9);
                for (float step = 0.0; step < 1.0; step += 0.01)
                {
                    vec3 samplePointT = entryPointT + step * length * rayDirectionT;
                    if (distance(samplePointT, vec3(0.0)) < 0.4)
                    {
                        volumetricDensity += 1.0 - procCellularNoise3D(0.5 + samplePointT / 2.0, 15);
                        break;
                    }
                }
                currentOrigin = checkPoint + (length + 0.01) * rayDirection;
                travelDistance += length + 0.01;
                */
            }
            else
            {
                break;
            }
        }
        else
        {
            // hit the sky box
            //lightColor += vec3(0.5, 0.8, 1.0) * 1.0;

            //vec3 shadedColor = vec3(0.5, 0.8, 1.0);
            //color *= shadedColor; // * intensity;
            vec3 hitPoint = abs(rayDirection.y) > 0.001 ? currentOrigin + rayDirection * ((1000.0 - currentOrigin.y) / rayDirection.y) : currentOrigin + rayDirection;
            hitPoint += vec3(float(timems) / 10000.0, 0.0, 0.0);
            vec4 skyBox = texture(skyTexture, (hitPoint.xz / 10000.0));
            currentObject = -1;
            color *= skyBox.rgb;
            //light += 0.5 * skyBox.rgb; //vec3(0.1); //vec3(0.6, 0.7, 0.9);
            light += vec3(0.5);

            // measure volumetrics
            volumetricDensity = 0.0;
            for (int i = 0; i < 100; ++i)
            {
                vec3 samplePoint = currentOrigin + rayDirection * float(i) * 0.1;
                float f = float((timems / 100000) % 500);
                float v = max(0.0, 8.0 - samplePoint.y) * fogDensity * (1.0 - generateCellularNoise3D(samplePoint / 16.0 + f * vec3(0.0, -0.005, 0.0), 20));
                volumetricDensity += v;
            }
            break;
        }

    }

    // finally apply the light and fog
    //color *= light;

    // lerp between volumetric color and color according to the density factor
    if (volumetricDensity > 0.00001)
    {
        volumetricDensity = clamp(volumetricDensity, 0.0, 1.0);
        volumetricColor = clamp(volumetricColor, 0.0, 1.0);
        color = vec3(
            lerp(color.r, volumetricColor.r, volumetricDensity),
            lerp(color.g, volumetricColor.g, volumetricDensity),
            lerp(color.b, volumetricColor.b, volumetricDensity)
        );
    }

    // fog is an ubiquituous volumetric body
    /*
    if (fogDensity > 0.00001)
    {
        float fogFactor = exp(-travelDistance * fogDensity);
        vec3 fogColor = vec3(0.6 + 0.1 * random2(uv));

        color = vec3(
            lerp(fogColor.r, color.r, fogFactor),
            lerp(fogColor.g, color.g, fogFactor),
            lerp(fogColor.b, color.b, fogFactor)
        );
        //volumetricDensity += 1.0 - fogFactor;
        //volumetricColor = volumetricColor * fogColor * (1.0 - fogFactor);
    }
    */
    

    // debugging: show the depth information
    //color.r = clamp(light, 0.0, 1.0);
    //color.r = 1.0 - travelDistance / 10.0;
    return vec4(color, float(traceCount * 1000 + currentObject));    
}

void main()
{
    // initialize const TASM registers
    tasmRegisters[REG_PTR_END_VALUE] = float(REG_END_VALUE);
    tasmRegisters[REG_PTR_VOID] = float(REG_VOID);
    tasmRegisters[REG_PTR_PC] = float(REG_PC);
    tasmRegisters[REG_PTR_SP] = float(REG_SP);
    tasmRegisters[REG_PTR_PARAM1] = float(REG_PARAM1);
    tasmRegisters[REG_PTR_PARAM2] = float(REG_PARAM2);
    tasmRegisters[REG_PTR_COLOR] = float(REG_COLOR_R);
    tasmRegisters[REG_PTR_NORMAL] = float(REG_NORMAL_X);
    tasmRegisters[REG_PTR_ATTRIBUTES] = float(REG_ATTRIB_1);
    tasmRegisters[REG_PTR_ATTRIB_2] = float(REG_ATTRIB_2);
    tasmRegisters[REG_PTR_ATTRIB_3] = float(REG_ATTRIB_3);
    tasmRegisters[REG_END_VALUE] = -1.0;



    float aspect = screenWidth / screenHeight;
    vec2 pixelSize = vec2(1.0) / vec2(screenWidth, screenHeight);

    float exposure = 1.0;

    vec3 origin = vec3(0, 0, -0.1);

    // MSAA
    int sampleCount = 0;
    float msaa = sqrt(msaaLevel);
    vec4 samplePoint;
    vec3 pixel = vec3(0.0);
    for (float subX = 0.0; subX < 1.0; subX += 1.0 / msaa)
    {
        for (float subY = 0.0; subY < 1.0; subY += 1.0 / msaa)
        {
            vec2 delta = vec2((-0.5 + subX) * pixelSize.x, (-0.5 + subY) * pixelSize.y);
            samplePoint = shootRayThroughScreen(uv + delta * 2.0, origin, aspect);
            pixel += samplePoint.rgb;
            ++sampleCount;
        }
    }
    pixel /= float(sampleCount);
    pixel *= exposure;
    pixel = gammaCorrection(pixel);

    if (enableToonEffect)
    {   
        //pixel = flattenColor(pixel, 4);
        int obj1 = intr(samplePoint.w);
        int obj2 = intr(shootRayThroughScreen(uv + pixelSize, origin, aspect).w);

        if (obj1 != obj2)
        {
            pixel = vec3(0.0, 0.0, 0.0);
        }
        else
        {
            pixel = flattenColor(pixel, 8);
        }
    }

    //fragColor = vec4(pixel, 1.0);
    //fragColor = vec4(objectsOnRayCount >= 5 ? 1.0 : pixel.r, pixel.g, pixel.b, 1.0);
    if (numObjectsOnRay > 16)
    {
        fragColor = vec4(1.0, 0.0, 0.0, 1.0);
    }
    else
    {
        fragColor = vec4(pixel, 1.0);
    }
}
