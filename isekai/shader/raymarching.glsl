#version 300 es
precision highp float;  /* Android needs this for sufficient precision */
precision highp int;    /* Android needs this for sufficient precision */
precision highp usampler2D;

// world configuration
const int HORIZON_SIZE = 15;
const int WORLD_PAGE_SIZE = 4096;

// the side-length of a cube in voxels
const int[7] LOD_CUBE_SIZE =   int[]( 4,  2,  1, 1, 1, 1, 1);
// the side-length of a sector in cubes
const int[7] LOD_SECTOR_SIZE = int[](16, 16, 16, 8, 4, 2, 1);
// the division factor from nominal voxels to LOD voxels
const int[7] LOD_CUBE_DIV = int[](1, 2, 4, 4, 4, 4, 4);
// the division factor from nominal cubes to LOD cubes
const int[7] LOD_SECTOR_DIV = int[](1, 1, 1, 2, 4, 8, 16);

// we use the INVALID_SECTOR_ADDRESS to mark sectors with pending content
const uint INVALID_SECTOR_ADDRESS = 1u;

in vec2 uv;
out vec4 fragColor;

uniform int timems;
uniform vec3 universeLocation;

uniform int tracingDepth;

uniform int renderChannel;
uniform bool enableShadows;
uniform bool enableAmbientOcclusion;
uniform bool enableOutlines;
uniform bool enableTasm;

uniform mat4 cameraTrafo;

uniform float screenWidth;
uniform float screenHeight;

uniform usampler2D worldData;
uniform sampler2D lightsData;
uniform sampler2D tasmData;

struct Channels
{
    bool final;
    int bounces;
    float totalDistance;
    vec3 origin;
    vec3 rayDirection;
    vec3 p;
    vec3 indirectP;
    vec3 light;
    vec3 albedo;
    vec3 surfaceNormal;
    bool outline;
};

struct SectorMapEntry
{
    uint address;
    int lod;
};

struct Material
{
    vec3 color;
    vec3 normal;
    float roughness;
    float ior;
};

bool freeEdge = false;
bool aoEdge = false;
int debug = 0;
vec3 debugColor = vec3(1.0, 0.0, 0.0);
int skipCount = 0;

float randomSeed = 0.0;

bool tasmProgramTooLong = false;
bool tasmStackOutOfBounds = false;

const int IMAGE_CHANNEL = 0;
const int DEPTH_BUFFER_CHANNEL = 1;
const int NORMALS_CHANNEL = 2;
const int LIGHTING_CHANNEL = 3;
const int COLORS_CHANNEL = 4;
const int OUTLINES_CHANNEL = 5;

const vec3 DISTANCE_FOG_COLOR = vec3(0.6, 0.5, 0.5);

float[72] tasmRegisters;
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
const int REG_ENV_P_X = 30;
const int REG_ENV_P_Y = 31;
const int REG_ENV_P_Z = 32;
const int REG_ENV_UNIVERSE_X = 33;
const int REG_ENV_UNIVERSE_Y = 34;
const int REG_ENV_UNIVERSE_Z = 35;
const int REG_STACK = 36;
const int REG_USER = 56;

const int TASM_TEST_BEGIN = 30;
const int TASM_TEST_END = 39;
const int TASM_BIN_BEGIN = 50;
const int TASM_BIN_END = 79;
const int TASM_GEN_BEGIN = 100;
const int TASM_GEN_END = 109;


/* Fast but less accurate sqrt approximation.
 */
float fastSqrt(float v)
{
    return v * inversesqrt(v + 0.00001);
}

/* Not as fast as fastSqrt, but gives better accuracy in the range [0, 1].
 */
float approxSqrt(float v)
{
    // 2nd-order minimax approximation of sqrt(x) on [0, 1]
    return v * (0.41731 + v * (0.59016 - 0.06757 * v));
}

float squaredDist(vec3 p1, vec3 p2)
{
    vec3 diff = p1 - p2;
    return dot(diff, diff);
}

float fastDistance(vec3 p1, vec3 p2)
{
  return fastSqrt(squaredDist(p1, p2));
}

/* Convers a linear address to a pixel location in the data texture.
 */
ivec2 textureAddress(uint address)
{
    return ivec2(
        address % uint(WORLD_PAGE_SIZE),
        address / uint(WORLD_PAGE_SIZE)
    );
}

/* Returns the data offset for the given sector.
 */
SectorMapEntry readSectorMapEntry(int sector)
{
    int value = int(texelFetch(worldData, ivec2(sector / 4, WORLD_PAGE_SIZE - 1), 0)[sector % 4]);
    return SectorMapEntry(
        uint(value >> 3),
        value & 7
    );
}

/* Returns the sector at the given world point.
 */
int sectorAt(vec3 p)
{
    int sectorLength = LOD_SECTOR_SIZE[0] * LOD_CUBE_SIZE[0];
    ivec3 sectorLoc = ivec3(p) / sectorLength;
    return sectorLoc.y * (HORIZON_SIZE * HORIZON_SIZE) +
           sectorLoc.z * HORIZON_SIZE +
           sectorLoc.x;
}

/* Returns the cube at the given point in a sector.
 */
int cubeAt(int sector, vec3 pInSector)
{
    const int sectorSize0 = LOD_SECTOR_SIZE[0];
    const int cubeSize0 = LOD_CUBE_SIZE[0];

    ivec3 cubeLoc = ivec3(pInSector) / cubeSize0;
    int cube = cubeLoc.x * (sectorSize0 * sectorSize0) +
               cubeLoc.y * sectorSize0 +
               cubeLoc.z;
    return cube;
}

/* Returns the origin of the given sector in world coordinates.
 */
vec3 sectorOrigin(int sector)
{
    const int sectorSize = LOD_SECTOR_SIZE[0];
    const int cubeSize = LOD_CUBE_SIZE[0];

    int y = sector / (HORIZON_SIZE * HORIZON_SIZE);
    int z = (sector % (HORIZON_SIZE * HORIZON_SIZE)) / HORIZON_SIZE;
    int x = sector % HORIZON_SIZE;

    return vec3(float(x), float(y), float(z)) * float(sectorSize * cubeSize);
}

/* Returns the origin of the given cube in world coordinates.
 */
vec3 cubeOrigin(int sector, int cube)
{
    SectorMapEntry sectorMapEntry = readSectorMapEntry(sector);
    int lod = sectorMapEntry.lod;

    // get the cube at p - sectorOrigin
    int sectorSize = LOD_SECTOR_SIZE[lod];
    int cubeSize = LOD_CUBE_SIZE[0];

    int x = cube / (sectorSize * sectorSize);
    int y = (cube % (sectorSize * sectorSize) / sectorSize);
    int z = cube % sectorSize;

    vec3 origin = sectorOrigin(sector) +
                  vec3(float(x), float(y), float(z)) * float(cubeSize);

    return origin;
}

uvec4 readCubeEntry(vec3 p)
{
    const int sectorSize0 = LOD_SECTOR_SIZE[0];
    const int cubeSize0 = LOD_CUBE_SIZE[0];
    const int sectorLength = sectorSize0 * cubeSize0;

    p = clamp(p, 0.0, float(HORIZON_SIZE * sectorLength - 1));

    // get the sector at p
    ivec3 sectorLoc = ivec3(p) / sectorLength;
    int sector = sectorLoc.y * (HORIZON_SIZE * HORIZON_SIZE) +
                 sectorLoc.z * HORIZON_SIZE +
                 sectorLoc.x;
    vec3 sectorOrigin = vec3(sectorLoc * sectorLength);

    SectorMapEntry sectorMapEntry = readSectorMapEntry(sector);
    int lod = sectorMapEntry.lod;

    // get the cube at p - sectorOrigin
    int sectorSize = LOD_SECTOR_SIZE[lod];
    int cubeSize = LOD_CUBE_SIZE[lod];
    vec3 pInSector = p - sectorOrigin;
    ivec3 cubeLoc = ivec3(pInSector) / cubeSize0;
    ivec3 lodCubeLoc = cubeLoc / LOD_SECTOR_DIV[lod];
    int lodCube = lodCubeLoc.x * (sectorSize * sectorSize) +
                  lodCubeLoc.y * sectorSize +
                  lodCubeLoc.z;

    // get the cube data offset and voxel payload address
    uint cubeDataOffset = sectorMapEntry.address + uint(lodCube * 1);
    return texelFetch(worldData, textureAddress(cubeDataOffset), 0);
}

/* Returns the type of voxel at the given world position, or 0 if the
 * location is empty.
 */
uint voxelType(vec3 p)
{
    const int sectorSize0 = LOD_SECTOR_SIZE[0];
    const int cubeSize0 = LOD_CUBE_SIZE[0];
    const int sectorLength = sectorSize0 * cubeSize0;

    p = clamp(p, 0.0, float(HORIZON_SIZE * sectorLength - 1));

    // get the sector at p
    ivec3 sectorLoc = ivec3(p) / sectorLength;
    int sector = sectorLoc.y * (HORIZON_SIZE * HORIZON_SIZE) +
                 sectorLoc.z * HORIZON_SIZE +
                 sectorLoc.x;
    vec3 sectorOrigin = vec3(sectorLoc * sectorLength);

    SectorMapEntry sectorMapEntry = readSectorMapEntry(sector);
    if (sectorMapEntry.address == INVALID_SECTOR_ADDRESS)
    {
        //debug = 2;
        return 0u;
    }
    int lod = sectorMapEntry.lod;

    // get the cube at p - sectorOrigin
    int sectorSize = LOD_SECTOR_SIZE[lod];
    int cubeSize = LOD_CUBE_SIZE[lod];
    vec3 pInSector = p - sectorOrigin;
    ivec3 cubeLoc = ivec3(pInSector) / cubeSize0;
    ivec3 lodCubeLoc = cubeLoc / LOD_SECTOR_DIV[lod];
    int cube = cubeLoc.x * (sectorSize0 * sectorSize0) +
               cubeLoc.y * sectorSize0 +
               cubeLoc.z;
    int lodCube = lodCubeLoc.x * (sectorSize * sectorSize) +
                  lodCubeLoc.y * sectorSize +
                  lodCubeLoc.z;
    vec3 cubeOrigin = vec3(cubeLoc * cubeSize0);

    // get the voxel payload address
    uint cubeDataAddress = readCubeEntry(p).b;

    vec3 pInCube = pInSector - cubeOrigin; 
    ivec3 voxelLoc = ivec3(pInCube);
    ivec3 lodVoxelLoc = voxelLoc / LOD_CUBE_DIV[lod];
    int voxel = voxelLoc.x * (cubeSize0 * cubeSize0) +
                voxelLoc.y * cubeSize0 +
                voxelLoc.z;
    int lodVoxel = lodVoxelLoc.x * cubeSize * cubeSize +
                   lodVoxelLoc.y * cubeSize +
                   lodVoxelLoc.z;

    int cubesPerSector = sectorSize * sectorSize * sectorSize;
    int voxelsPerCube = cubeSize * cubeSize * cubeSize;
    if (cubeSize == 4)
    {
        uint size = cubeSize == 4 ? 16u
                                  : 2u;
        uint voxelAddress = sectorMapEntry.address +
                            uint(cubesPerSector) +
                            cubeDataAddress * 16u +
                            uint(lodVoxel / 4);
        return texelFetch(worldData, textureAddress(voxelAddress), 0)[lodVoxel % 4];
    }
    else if (cubeSize == 2)
    {
        uint voxelAddress = sectorMapEntry.address +
                            uint(cubesPerSector) +
                            cubeDataAddress * 2u +
                            uint(lodVoxel / 4);
        return texelFetch(worldData, textureAddress(voxelAddress), 0)[lodVoxel % 4];
    }
    else
    {
        uint voxelAddress = sectorMapEntry.address +
                            uint(cubesPerSector) +
                            cubeDataAddress  / 4u;
        return texelFetch(worldData, textureAddress(voxelAddress), 0)[cubeDataAddress % 4u];
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

vec3 hitCubeAabb(vec3 origin, vec3 rayDirection, vec3 pos)
{
    vec2 tx = aabbMinMax(origin.x, rayDirection.x, pos.x, pos.x + 4.0);
    vec2 ty = aabbMinMax(origin.y, rayDirection.y, pos.y, pos.y + 4.0);
    vec2 tz = aabbMinMax(origin.z, rayDirection.z, pos.z, pos.z + 4.0);

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

/* Creates a transformation matrix for transforming to surface space.
 */
mat4 createSurfaceTrafo(vec3 normal)
{
    vec3 up = abs(normal.y) < 0.999 ? vec3(0.0, 1.0, 0.0)
                                    : vec3(1.0, 0.0, 0.0);

    vec3 tangent = normalize(cross(up, normal));
    vec3 bitangent = cross(normal, tangent);

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

float generateWhiteNoise(vec2 st)
{
    return fract(sin(dot(st.xy, vec2(12.9898,78.233))) * 43758.5453123);
}

float generateCellularNoise2D(vec2 p, int size, float variant)
{
    float fsize = float(size);
    float cubeSize = 1.0 / fsize;

    // in which section am I?
    ivec2 q = ivec2(floor(p * fsize));

    // check the surroundings
    float minSquaredDist = 1e10;
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

            float squaredDist = dot(samplePoint - p, samplePoint - p);
            minSquaredDist = min(squaredDist, minSquaredDist);
        }
    }
    return fastSqrt(minSquaredDist) / cubeSize;
}

float generateCellularNoise3D(vec3 p, int size)
{
    float fsize = float(size);
    float cubeSize = 1.0 / fsize;

    // in which section am I?
    ivec3 q = ivec3(floor(p * fsize));

    // check the surroundings
    float minSquaredDist = 1e10;
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

                float squaredDist = dot(samplePoint - p, samplePoint - p);
                minSquaredDist = min(squaredDist, minSquaredDist);
            }
        }
    }
    return fastSqrt(minSquaredDist) / cubeSize;
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
}

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
        float t2 = cosi * eta + fastSqrt(k);
        return t1 - surfaceNormal * t2;
    }
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

float sdfBox(vec3 p)
{
    vec3 halfSides = vec3(0.5);
    vec3 pt = p - vec3(0.0);
    vec3 q = abs(pt) - halfSides;
    return length(max(q, 0.0)) - min(0.0, max(max(q.x, q.y), q.z));
}

/* Processes a set of TASM instructions to generate a texture.
 */
Material processTasm(int program, vec2 st, vec3 p, float travelDist)
{
    // Since the GPU is quite limited on what it can do, implementing the
    // TASM instruction set might be too heavy for it. Therefore, all TASM
    // instructions are broken down into microcode defined by the TASM firmware.
    // The GPU processes the microcode only.

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
    tasmRegisters[REG_ENV_P_X] = p.x;
    tasmRegisters[REG_ENV_P_Y] = p.y;
    tasmRegisters[REG_ENV_P_Z] = p.z;

    const int maxSteps = 128;
    int i = 0;
    while (i < maxSteps)
    {
        tasmProgramTooLong = i == maxSteps - 1;
        tasmStackOutOfBounds = tasmRegisters[REG_SP] < float(REG_STACK) ||
                               tasmRegisters[REG_SP] >= float(REG_USER);

        ++i;

        int pc = int(tasmRegisters[REG_PC]);
        int stackPointer = int(tasmRegisters[REG_SP]);

        if (pc < 0 || tasmProgramTooLong || tasmStackOutOfBounds)
        {
            // exit
            break;
        }

        instruction = texelFetch(tasmData, ivec2(pc, program), 0);

        opCode = int(instruction.r);
        instructionSize = instruction.g;
        tasmRegisters[REG_PARAM1] = instruction.b;
        tasmRegisters[REG_PARAM2] = instruction.a;


        // caching these appears to add too much overhead and memory spilling, and we're generally
        // better off without caching
        microCodeCopyReg1 = texelFetch(tasmData, ivec2(0, 3000 + opCode), 0);
        microCodeTest = opCode >= TASM_TEST_BEGIN && opCode <= TASM_TEST_END ? texelFetch(tasmData, ivec2(1, 3000 + opCode), 0) : vec4(0.0);
        microCodeBinOp = opCode >= TASM_BIN_BEGIN && opCode <= TASM_BIN_END ? texelFetch(tasmData, ivec2(2, 3000 + opCode), 0) : vec4(0.0);
        microCodeGenOp = opCode >= TASM_GEN_BEGIN && opCode <= TASM_GEN_END ? texelFetch(tasmData, ivec2(3, 3000 + opCode), 0) : vec4(0.0);
        microCodeAddReg = texelFetch(tasmData, ivec2(4, 3000 + opCode), 0);
        microCodeCopyReg2 = texelFetch(tasmData, ivec2(5, 3000 + opCode), 0);

        // advance program counter
        tasmRegisters[REG_PC] += instructionSize;

        // copy n registers from *source to *dest (avoid with batchSize = 0)
        batchSize = int(microCodeCopyReg1.r);
        srcPointer = int(tasmRegisters[int(microCodeCopyReg1.g)]);
        destPointer = int(tasmRegisters[int(microCodeCopyReg1.b)]);
        offsets = int(microCodeCopyReg1.a);
        srcOffset = (offsets >> 4) - 8;
        destOffset = (offsets & 15) - 8;

        if (batchSize >= 1)
            tasmRegisters[destPointer + destOffset] = tasmRegisters[srcPointer + srcOffset];
        if (batchSize >= 2)
            tasmRegisters[destPointer + 1 + destOffset] = tasmRegisters[srcPointer + 1 + srcOffset];
        if (batchSize >= 3)
            tasmRegisters[destPointer + 2 + destOffset] = tasmRegisters[srcPointer + 2 + srcOffset];

        // test
        op = int(microCodeTest.r);
        if (op > 0)
        {
            workParam1 = tasmRegisters[stackPointer - 2];
            workParam2 = tasmRegisters[stackPointer - 1];

            bool testResult = false;
            if (op == 1) testResult = workParam1 < workParam2;
            else if (op == 2) testResult = workParam1 <= workParam2;
            else if (op == 3) testResult = abs(workParam1 - workParam2) < 0.0001;
            else if (op == 4) testResult = workParam1 > workParam2;
            else if (op == 5) testResult = workParam1 >= workParam2;

            if (! testResult)
            {
                tasmRegisters[REG_PC] = tasmRegisters[REG_PARAM1];
            }
        }

        // binop
        op = int(microCodeBinOp.r);
        if (op > 0)
        {
            batchSize = int(microCodeBinOp.g);
            for (ri = 0; ri < 3; ++ri)
            {
                workParam1 = tasmRegisters[stackPointer - 2 * batchSize + ri];
                workParam2 = tasmRegisters[stackPointer - batchSize + ri];
                v = (op == 1) ? workParam1 + workParam2 : v;
                v = (op == 2) ? workParam1 - workParam2 : v;
                v = (op == 3) ? workParam1 * workParam2 : v;
                v = (op == 4) ? workParam1 / workParam2 : v;
                v = (op == 5) ? min(workParam1, workParam2) : v;
                v = (op == 6) ? max(workParam1, workParam2) : v;
                v = (op == 7) ? workParam1 + exp(workParam2) : v;
                tasmRegisters[stackPointer - 2 * batchSize + ri] = ri < batchSize ? v
                                                                                                 : workParam1;
            }
        }

        // gen
        op = int(microCodeGenOp.r);
        if (op == 5)
        {
            workParam1 = tasmRegisters[stackPointer - 1];
            workParam2 = tasmRegisters[stackPointer - 2];
            workParam3 = tasmRegisters[stackPointer - 3];
            workParam4 = tasmRegisters[stackPointer - 4];

            resultVec = generateBumpNormal(workParam4, workParam3, workParam2, workParam1);
            
            tasmRegisters[stackPointer - 4] = resultVec.x;
            tasmRegisters[stackPointer - 3] = resultVec.y;
            tasmRegisters[stackPointer - 2] = resultVec.z;
        }
        else if (op == 6)
        {
            workParam1 = tasmRegisters[stackPointer - 1];
            workParam2 = tasmRegisters[stackPointer - 2];
            workParam3 = tasmRegisters[stackPointer - 3];

            resultVec = vec3(generateMipMap(vec2(workParam3, workParam2), int(workParam1)), 0.0);
            
            tasmRegisters[stackPointer - 3] = resultVec.x;
            tasmRegisters[stackPointer - 2] = resultVec.y;

        }
        else if (op == 7)
        {
            workParam1 = tasmRegisters[stackPointer - 1];
            workParam2 = tasmRegisters[stackPointer - 2];
            workParam3 = tasmRegisters[stackPointer - 3];
            tasmRegisters[REG_PARAM1] = lerp(workParam3, workParam2, workParam1);
        }
        else if (op > 0)
        {
            workParam1 = tasmRegisters[stackPointer - 1];
            workParam2 = tasmRegisters[stackPointer - 2];
            workParam3 = tasmRegisters[stackPointer - 3];
            workParam4 = tasmRegisters[stackPointer - 4];

            tasmRegisters[REG_PARAM1] = (op == 1 ? generateLine(vec2(workParam4, workParam3), workParam2, workParam1) : 0.0) +
                                        (op == 2 ? generateCheckerboard(vec2(workParam2, workParam1)) : 0.0) + 
                                        (op == 3 ? generateWhiteNoise(vec2(workParam2, workParam1)) : 0.0) + 
                                        (op == 4 ? generateCellularNoise2D(vec2(workParam4, workParam3), int(workParam2), workParam1) : 0.0);
        }

        // add const value to a register (avoid with void pointer)
        srcPointer = int(tasmRegisters[int(microCodeAddReg.r)]);
        tasmRegisters[srcPointer] += microCodeAddReg.g;

        // copy n registers from *source to *dest (avoid with batch size = 0)
        batchSize = int(microCodeCopyReg2.r);
        srcPointer = int(tasmRegisters[int(microCodeCopyReg2.g)]);
        destPointer = int(tasmRegisters[int(microCodeCopyReg2.b)]);
        offsets = int(microCodeCopyReg2.a);
        srcOffset = (offsets >> 4) - 8;
        destOffset = (offsets & 15) - 8;

        if (batchSize >= 1)
            tasmRegisters[destPointer + destOffset] = tasmRegisters[srcPointer + srcOffset];
        if (batchSize >= 2)
            tasmRegisters[destPointer + 1 + destOffset] = tasmRegisters[srcPointer + 1 + srcOffset];
        if (batchSize >= 3)
            tasmRegisters[destPointer + 2 + destOffset] = tasmRegisters[srcPointer + 2 + srcOffset];
    }

    return Material(
        vec3(tasmRegisters[REG_COLOR_R], tasmRegisters[REG_COLOR_G], tasmRegisters[REG_COLOR_B]),
        vec3(tasmRegisters[REG_NORMAL_X], tasmRegisters[REG_NORMAL_Y], tasmRegisters[REG_NORMAL_Z]),
        tasmRegisters[REG_ATTRIB_1],
        tasmRegisters[REG_ATTRIB_2]
    );
}

bool isEdgeZ(vec3 p, float epsilon)
{
    // p is in object space
    p = abs(p);
    return p.x > 0.5 - epsilon && abs(p.x - p.y) < epsilon;
}

bool isEdgeY(vec3 p, float epsilon)
{
    // p is in object space
    p = abs(p);
    return p.x > 0.5 - epsilon && abs(p.x - p.z) < epsilon;
}

bool isEdgeX(vec3 p, float epsilon)
{
    // p is in object space
    p = abs(p);
    return p.y > 0.5 - epsilon && abs(p.y - p.z) < epsilon;
}

/* Returns the surface material at the given location.
 */
Material getObjectMaterial(vec3 p, float dist, vec3 surfaceNormal)
{
    int materialId = int(voxelType(p));

    vec2 st = p.xy;

    // position texture on cube
    vec3 n = surfaceNormal;
    vec3 p2 = abs(n.y) > 0.0 ? n.zxy : n.zyx;
    float dp = dot(n, p2);
    vec3 axis1 = normalize(p2 - dp * n);
    vec3 axis2 = normalize(cross(n, axis1));
    float x = dot(p, axis1);
    float y = dot(p, axis2);
    st = 0.5 + vec2(x, y);

    if (enableTasm)
    {
        return processTasm(materialId, st, p, dist);
    }
    else
    {
        return Material(
            vec3(1.0),
            vec3(0.0, 0.0, 1.0),
            1.0,
            0.0
        );
    }
}

bool cubeHasVoxel(vec3 pInCube, uvec2 pattern, int lod)
{
    int cubeSize = LOD_CUBE_SIZE[lod];

    uint patternHi = pattern.r;
    uint patternLo = pattern.g;

    if (cubeSize > 1)
    {
        ivec3 loc = ivec3(pInCube);
        loc /= LOD_CUBE_DIV[lod];
        int bitsPerCoord = cubeSize == 4 ? 2 : 1;
        int n = (loc.x << (bitsPerCoord + bitsPerCoord)) +
                (loc.y << bitsPerCoord) +
                loc.z;
        return n < 32 ? (patternLo & uint(1 << n)) > 0u
                      : (patternHi & uint(1 << (n - 32))) > 0u;
    }
    else
    {
        return patternLo > 0u;
    }
}

/* Checks the cube's bit pattern to see if the ray may hit any voxel.
 */
bool mayHitVoxels(vec3 entryPoint, vec3 exitPoint, uvec2 pattern, int lod)
{
    // entryPoint and exitPoints are in cube-local coordinates (between vec3(0.0) and vec3(4.0))

    if (pattern.r == 0u && pattern.g == 0u)
    {
        return false;
    }
    if (lod > 0)
    {
        return true;
    }

    const uvec2[4] xSlices = uvec2[](
        uvec2(0x00000000, 0x0000ffff),   // 0: 0000000000000000 0000000000000000 0000000000000000 1111111111111111
        uvec2(0x00000000, 0xffff0000),   // 1: 0000000000000000 0000000000000000 1111111111111111 0000000000000000
        uvec2(0x0000ffff, 0x00000000),   // 2: 0000000000000000 1111111111111111 0000000000000000 0000000000000000
        uvec2(0xffff0000, 0x00000000)    // 3: 1111111111111111 0000000000000000 0000000000000000 0000000000000000
    );

    const uvec2[4] ySlices = uvec2[](
        uvec2(0x000f000f, 0x000f000f),   // 0: 0000000000001111 0000000000001111 0000000000001111 0000000000001111
        uvec2(0x00f000f0, 0x00f000f0),   // 1: 0000000011110000 0000000011110000 0000000011110000 0000000011110000
        uvec2(0x0f000f00, 0x0f000f00),   // 2: 0000111100000000 0000111100000000 0000111100000000 0000111100000000
        uvec2(0xf000f000, 0xf000f000)    // 3: 1111000000000000 1111000000000000 1111000000000000 1111000000000000
    );

    const uvec2[4] zSlices = uvec2[](
        uvec2(0x11111111, 0x11111111),   // 0: 0001000100010001 0001000100010001 0001000100010001 0001000100010001
        uvec2(0x22222222, 0x22222222),   // 1: 0010001000100010 0010001000100010 0010001000100010 0010001000100010
        uvec2(0x44444444, 0x44444444),   // 2: 0100010001000100 0100010001000100 0100010001000100 0100010001000100
        uvec2(0x88888888, 0x88888888)    // 3: 1000100010001000 1000100010001000 1000100010001000 1000100010001000
    );

    int minX = clamp(int(min(entryPoint.x, exitPoint.x)), 0, 3);
    int minY = clamp(int(min(entryPoint.y, exitPoint.y)), 0, 3);
    int minZ = clamp(int(min(entryPoint.z, exitPoint.z)), 0, 3);
    
    int maxX = clamp(int(max(entryPoint.x, exitPoint.x)), 0, 3);
    int maxY = clamp(int(max(entryPoint.y, exitPoint.y)), 0, 3);
    int maxZ = clamp(int(max(entryPoint.z, exitPoint.z)), 0, 3); 

    uvec2 bitsX = uvec2(0);
    uvec2 bitsY = uvec2(0);
    uvec2 bitsZ = uvec2(0);

    for (int i = 0; i < 4; ++i)
    {
        bitsX = bitsX | ((i >= minX && i <= maxX) ? xSlices[i] : uvec2(0));
        bitsY = bitsY | ((i >= minY && i <= maxY) ? ySlices[i] : uvec2(0));
        bitsZ = bitsZ | ((i >= minZ && i <= maxZ) ? zSlices[i] : uvec2(0));
    }

    uvec2 bits = bitsX & bitsY & bitsZ & pattern;

    return bits.r > 0u || bits.g > 0u;
}

bool hasVoxelAt(vec3 p)
{
    int sector = sectorAt(p);
    vec3 sectorOrgn = sectorOrigin(sector);
    int cubeIdx = cubeAt(sector, p - sectorOrgn);
    vec3 cubeOrgn = cubeOrigin(sector, cubeIdx);
    vec3 pT = p - cubeOrgn;

    SectorMapEntry sectorMapEntry = readSectorMapEntry(sector);
    if (sectorMapEntry.address == INVALID_SECTOR_ADDRESS)
    {
        //debug = 2;
        return false;
    }
    uvec4 patternAndAddress = readCubeEntry(p);

    return cubeHasVoxel(pT, patternAndAddress.rg, sectorMapEntry.lod);
}

vec4 raymarchVoxels(vec3 initProbePoint, vec3 origin, vec3 entryPoint, vec3 rayDirection)
{
    int sector = sectorAt(initProbePoint);
    vec3 sectorOrgn = sectorOrigin(sector);
    int cubeIdx = cubeAt(sector, initProbePoint - sectorOrgn);
    vec3 cubeOrgn = cubeOrigin(sector, cubeIdx);

    SectorMapEntry sectorMapEntry = readSectorMapEntry(sector);
    int lod = sectorMapEntry.lod;
    if (sectorMapEntry.address == INVALID_SECTOR_ADDRESS)
    {
        // this sector is empty
        //debug = 2;
        return vec4(entryPoint, 0.0);
    }
    uvec4 patternAndAddress = readCubeEntry(initProbePoint);
    
    vec3 exitPoint = entryPoint + rayDirection * 8.0;

    vec3 entryPointT = entryPoint - cubeOrgn;
    vec3 exitPointT = exitPoint - cubeOrgn;

    if (! mayHitVoxels(entryPointT, exitPointT, patternAndAddress.rg, lod))
    {
        ++skipCount;
        return vec4(entryPoint, 0.0);
    }

    vec3 p = entryPoint;
    vec3 pT = entryPointT;
    vec3 probePoint = pT;

    const float gridSize = 1.0;
    //float gridSize = 1.0 * float(LOD_CUBE_SIZE[0] / LOD_CUBE_SIZE[lod]);

    if (cubeHasVoxel(probePoint, patternAndAddress.rg, lod))
    {
        return vec4(p, 1.0);
    }

    vec3 invRayDirection = 1.0 / abs(rayDirection);
    vec3 rayDirectionSigns = sign(rayDirection);

    vec3 scalingsOnGrid = vec3(
        rayDirection.x != 0.0 ? invRayDirection.x : 9999.0,
        rayDirection.y != 0.0 ? invRayDirection.y : 9999.0,
        rayDirection.z != 0.0 ? invRayDirection.z : 9999.0
    );

    vec3 disabler = step(9990.0, scalingsOnGrid) * 9999.0;
    /*
    vec3 disabler = vec3(
        scalingsOnGrid.x < 9990.0 ? 0.0 : 9999.0,
        scalingsOnGrid.y < 9990.0 ? 0.0 : 9999.0,
        scalingsOnGrid.z < 9990.0 ? 0.0 : 9999.0
    );
    */

    vec3 gridPoint = floor(p / gridSize) * gridSize;

    gridPoint += vec3(
        rayDirection.x > 0.0 ? 1.0 : 0.0,
        rayDirection.y > 0.0 ? 1.0 : 0.0,
        rayDirection.z > 0.0 ? 1.0 : 0.0
    ) * gridSize;

    vec3 distsOnGrid = abs(gridPoint - p);
    for (int i = 0; i < 12; ++i)
    {
        vec3 rayLengths = distsOnGrid * scalingsOnGrid + disabler;
        bool advanceX = rayLengths.x <= rayLengths.y && rayLengths.x <= rayLengths.z;
        bool advanceY = rayLengths.y <= rayLengths.x && rayLengths.y <= rayLengths.z;
        bool advanceZ = rayLengths.z <= rayLengths.x && rayLengths.z <= rayLengths.y;

        if (advanceX && advanceZ)
        {
            advanceY = false;
            advanceZ = false;
        }
        if (advanceX && advanceY)
        {
            advanceY = false;
            advanceZ = false;
        }
        if (advanceY && advanceZ)
        {
            advanceX = false;
            advanceZ = false;
        }

        vec3 advanceVec = vec3(
            advanceX ? rayDirectionSigns.x : 0.0,
            advanceY ? rayDirectionSigns.y : 0.0,
            advanceZ ? rayDirectionSigns.z : 0.0
        ) * gridSize;

        distsOnGrid += abs(advanceVec);
        probePoint += advanceVec;

        // be sure to take only one of the rayLengths
        p = entryPoint + rayDirection * ((advanceX ? rayLengths.x : 0.0) +
                                         (advanceY ? rayLengths.y : 0.0) +
                                         (advanceZ ? rayLengths.z : 0.0));

        if (rayDirectionSigns.x < 0.0 && probePoint.x < 0.0 ||
            rayDirectionSigns.y < 0.0 && probePoint.y < 0.0 ||
            rayDirectionSigns.z < 0.0 && probePoint.z < 0.0 ||
            rayDirectionSigns.x > 0.0 && probePoint.x >= 4.0 ||
            rayDirectionSigns.y > 0.0 && probePoint.y >= 4.0 ||
            rayDirectionSigns.z > 0.0 && probePoint.z >= 4.0)
        {
            // leaving the cube
            break;
        }

        if (cubeHasVoxel(probePoint, patternAndAddress.rg, lod))
        {
            return vec4(p + advanceVec * 0.01, 1.0);
        }
    }

    return vec4(p, 0.0);
}

vec4 raymarchCubes(vec3 origin, vec3 rayDirection, float maxDistance)
{
    const int sectorSize = LOD_SECTOR_SIZE[0];
    const int cubeSize = LOD_CUBE_SIZE[0];

    float maxDistanceSquared = maxDistance * maxDistance;
    vec3 p = origin;
    vec3 probePoint = origin;

    vec4 result = raymarchVoxels(probePoint, origin, p, rayDirection);
    if (result.w > 0.0)
    {
        return result;
    }

    int sector = sectorAt(probePoint);
    int lod = readSectorMapEntry(sector).lod;
    float gridSize = lod == 0 ? 4.0
                              : 4.0 * float(LOD_SECTOR_SIZE[0] / LOD_SECTOR_SIZE[lod - 1]);

    vec3 invRayDirection = 1.0 / abs(rayDirection);
    vec3 rayDirectionSigns = sign(rayDirection);

    vec3 scalingsOnGrid = vec3(
        rayDirection.x != 0.0 ? invRayDirection.x : 9999.0,
        rayDirection.y != 0.0 ? invRayDirection.y : 9999.0,
        rayDirection.z != 0.0 ? invRayDirection.z : 9999.0
    );

    vec3 disabler = step(9990.0, scalingsOnGrid) * 9999.0;
    /*
    vec3 disabler = vec3(
        scalingsOnGrid.x < 9990.0 ? 0.0 : 9999.0,
        scalingsOnGrid.y < 9990.0 ? 0.0 : 9999.0,
        scalingsOnGrid.z < 9990.0 ? 0.0 : 9999.0
    );
    */

    vec3 gridPoint = floor(p / gridSize) * gridSize;

    gridPoint += vec3(
        rayDirection.x > 0.0 ? 1.0 : 0.0,
        rayDirection.y > 0.0 ? 1.0 : 0.0,
        rayDirection.z > 0.0 ? 1.0 : 0.0
    ) * gridSize;

    vec3 distsOnGrid = abs(gridPoint - p);
    for (int i = 0; i < 32; ++i)
    {
        vec3 rayLengths = distsOnGrid * scalingsOnGrid + disabler;
        bool advanceX = rayLengths.x <= rayLengths.y && rayLengths.x <= rayLengths.z;
        bool advanceY = rayLengths.y <= rayLengths.x && rayLengths.y <= rayLengths.z;
        bool advanceZ = rayLengths.z <= rayLengths.x && rayLengths.z <= rayLengths.y;

        if (advanceX && advanceZ)
        {
            advanceY = false;
            advanceZ = false;
        }
        if (advanceX && advanceY)
        {
            advanceY = false;
            advanceZ = false;
        }
        if (advanceY && advanceZ)
        {
            advanceX = false;
            advanceZ = false;
        }

        vec3 advanceVec = vec3(
            advanceX ? rayDirectionSigns.x : 0.0,
            advanceY ? rayDirectionSigns.y : 0.0,
            advanceZ ? rayDirectionSigns.z : 0.0
        ) * gridSize;

        distsOnGrid += abs(advanceVec);
        probePoint += advanceVec;

        // be sure to take only one of the rayLengths
        p = origin + rayDirection * ((advanceX ? rayLengths.x : 0.0) +
                                     (advanceY ? rayLengths.y : 0.0) +
                                     (advanceZ ? rayLengths.z : 0.0));

        if (probePoint.x < 0.0 || probePoint.y < 0.0 || probePoint.z < 0.0 ||
            probePoint.x >= float(sectorSize * cubeSize * HORIZON_SIZE) ||
            probePoint.y >= float(sectorSize * cubeSize * HORIZON_SIZE) ||
            probePoint.z >= float(sectorSize * cubeSize * HORIZON_SIZE) ||
            squaredDist(p, origin) > maxDistanceSquared)
        {
            // out of range
            result = vec4(p, 0.0);
            break;
        }

        // clamp point to within the cube
        vec3 probePointFloored = floor(probePoint / float(cubeSize)) * float(cubeSize);
        p = vec3(
            clamp(p.x, probePointFloored.x, probePointFloored.x + 3.9999),
            clamp(p.y, probePointFloored.y, probePointFloored.y + 3.9999),
            clamp(p.z, probePointFloored.z, probePointFloored.z + 3.9999)
        );
        result = raymarchVoxels(probePoint, origin, p, rayDirection);
        if (result.w > 0.0)
        {
            break;
        }
    }

    return result;
}

vec4 raymarch(vec3 origin, vec3 rayDirection, float maxDistance)
{
    return raymarchCubes(origin, rayDirection, maxDistance);
}

vec4 skyBox(vec3 origin, vec3 rayDirection)
{
    vec3 hitPoint = abs(rayDirection.y) > 0.001 ? origin + rayDirection * ((1000.0 - origin.y) / rayDirection.y)
                                                : origin + rayDirection;

    vec3 color = enableTasm && rayDirection.y > 0.0 ? processTasm(0, hitPoint.xz, hitPoint, fastDistance(origin, hitPoint)).color
                                                    : DISTANCE_FOG_COLOR;

    return vec4(color, 1.0);
}

vec3 phongShading(int lightSource, vec3 origin, vec3 checkPoint, vec3 ambience, vec3 surfaceNormal, float roughness)
{
    // Phong shading: lighting = ambient + diffuse + specular
    //                color = modelColor * lighting

    vec3 viewDirection = normalize(origin - checkPoint);
    vec3 lighting = ambience;
    float shininess = (1.0 - roughness) * 64.0;

    vec3 lightLoc = getLightLocation(lightSource);
    vec3 lightCol = getLightColor(lightSource);
    float lightRange = getLightRange(lightSource);

    vec3 directionToLight = normalize(lightLoc - checkPoint);
    float lightDistance = length(checkPoint - lightLoc);

    float impact = dot(directionToLight, surfaceNormal);

    // does the light reach?
    if (impact <= 0.001)
    {
        // nope
        return lighting;
    }
    if (enableShadows)
    {
        // we may not have to go all the way to the light to know if it reaches (it may be far far away)
        float optimizedLightDistance = min(lightDistance, 100.0);
        vec4 hitSample = raymarch(checkPoint + directionToLight * 0.1, directionToLight, optimizedLightDistance);
        if (hitSample.w > 0.0)
        {
            // nope
            return lighting;
        }
    }

    // light attenuation based on distance and strength of the light source
    float attenuation = clamp(1.0 - lightDistance / lightRange, 0.0, 1.0);

    if (lightLoc.y > 1000.0)
    {
        // light attenuation based on clouds
        vec4 skyColor = skyBox(checkPoint, directionToLight);
        attenuation *= lerp(0.1, 1.0, 1.0 - skyColor.r);
    }

    attenuation *= attenuation;
    vec3 attenuatedLight = lightCol * attenuation;

    // diffuse light
    vec3 diffuse = attenuatedLight * impact;

    // specular highlight (Blinn-Phong)
    vec3 halfDirection = normalize(directionToLight + viewDirection);
    float specularStrength = pow(max(0.0, dot(surfaceNormal, halfDirection)), shininess) * 0.5;
    vec3 specular = attenuatedLight * specularStrength;

    lighting += diffuse + specular;

    return lighting.rgb;
}

float ambientOcclusion(vec3 p, mat4 surfaceTrafo, float size)
{
    // p is in world space

    vec3 tangent = surfaceTrafo[0].xyz;
    vec3 bitangent = surfaceTrafo[1].xyz;
    vec3 surfaceNormal = surfaceTrafo[2].xyz; 

    // move p away from the surface a bit
    p += surfaceNormal * 0.1;

    // p in surface space for distance computations
    vec3 surfacePoint = (inverse(surfaceTrafo) * vec4(mod(p, 1.0), 1.0)).xyz;

    // build direction vectors
    vec3 v1 = tangent * size;
    vec3 v2 = bitangent * size;

    // compute sample points in world space
    vec3 samplePoints[8];
    samplePoints[0] = p + v1; 
    samplePoints[1] = p - v1;
    samplePoints[2] = p + v2;
    samplePoints[3] = p - v2;

    // diagonal samples
    samplePoints[4] = p + v1 + v2;
    samplePoints[5] = p - v1 - v2; 
    samplePoints[6] = p + v1 - v2;
    samplePoints[7] = p - v1 + v2;

    // check for neighbors
    bool samples[8];
    for (int i = 0; i < 8; ++i)
    {
        samples[i] = hasVoxelAt(samplePoints[i]);
    }

    // the distance to the surface edges specifies the occlusion strength
    float dist1 = 0.1; //abs(1.0 - surfacePoint.x);
    float dist2 = 0.1; //abs(0.0 - surfacePoint.x);
    float dist3 = 0.1; //abs(1.0 - surfacePoint.y);
    float dist4 = 0.1; //abs(0.0 - surfacePoint.y);

    //if (abs(0.5 - surfacePoint.x) < 0.1 || abs(0.5 - surfacePoint.y) < 0.1 || abs(0.5 - surfacePoint.z) < 0.1) debug = 2;

    float shadow = ((samples[0] ? size - dist1 : 0.0) +
                    (samples[1] ? size - dist2 : 0.0) +
                    (samples[2] ? size - dist3 : 0.0) +
                    (samples[3] ? size - dist4 : 0.0)) / (2.0 * size);

    // the corners in surface space
    vec2 corners[4] = vec2[](
        vec2(1.0, 1.0),
        vec2(0.0, 0.0),
        vec2(1.0, 0.0),
        vec2(0.0, 1.0)
    );

    float cornerShadow = 0.0;
    bool cornerLine = false;
    for (int i = 0; i < 4; ++i)
    {
        if (samples[i + 4])
        {
            float dx = corners[i].x - surfacePoint.x;
            float dy = corners[i].y - surfacePoint.y;
            float dist = 0.1; //fastSqrt(dx * dx + dy * dy);
            cornerShadow += size - dist;
            cornerLine = dist < 0.05;
        }
    }
    cornerShadow /= (2.0 * size);

    shadow = max(abs(shadow), abs(cornerShadow));

    aoEdge = shadow > 0.01 &&
             (cornerLine ||
              samples[0] && dist1 < 0.05 ||
              samples[1] && dist2 < 0.05 ||
              samples[2] && dist3 < 0.05 ||
              samples[3] && dist4 < 0.05);

    return clamp(1.0 - shadow, 0.0, 1.0);
}

/* Determining the box normals is tricky around the edges and corners.
 * To get this right, we have to check their surroundings.
 */
vec3 getCorrectedBoxNormals(vec3 p, vec3 rayDirection)
{
    // p is in world space

    // we have to check the neighbors to resolve surface normal ambiguities at the edges
    vec3 centerPoint = vec3(ivec3(p)) + 0.5;
    vec3 pT = p - centerPoint;

    // the three normals facing towards the camera
    vec3 surfaceNormalX = vec3(sign(pT.x), 0.0, 0.0);
    vec3 surfaceNormalY = vec3(0.0, sign(pT.y), 0.0);
    vec3 surfaceNormalZ = vec3(0.0, 0.0, sign(pT.z));

    // we have to check the dot products for negativity to filter out normals facing away
    float dotX = dot(surfaceNormalX, -rayDirection);
    float dotY = dot(surfaceNormalY, -rayDirection);
    float dotZ = dot(surfaceNormalZ, -rayDirection);

    // check if there are neighbors on sides facing to the camera,
    // as these sides cannot be visible
    bool hasXNeighbor = hasVoxelAt(centerPoint + surfaceNormalX);
    bool hasYNeighbor = hasVoxelAt(centerPoint + surfaceNormalY);
    bool hasZNeighbor = hasVoxelAt(centerPoint + surfaceNormalZ);

    // the interesting (because ambiguous) places are the edges and corners
    bool nearEdgeX = isEdgeX(pT, 0.05);
    bool nearEdgeY = isEdgeY(pT, 0.05);
    bool nearEdgeZ = isEdgeZ(pT, 0.05);

    bool edgeX = nearEdgeX && isEdgeX(pT, 0.03);
    bool edgeY = nearEdgeY && isEdgeY(pT, 0.03);
    bool edgeZ = nearEdgeZ && isEdgeZ(pT, 0.03);

    // side-computation: only free edges may show outlines
    freeEdge = nearEdgeX && (! hasYNeighbor && ! hasZNeighbor) ||
               nearEdgeY && (! hasXNeighbor && ! hasZNeighbor) ||
               nearEdgeZ && (! hasXNeighbor && ! hasYNeighbor);

    vec3 surfaceNormal;

    bool mayUseX = true;
    bool mayUseY = true;
    bool mayUseZ = true;

    // sort out the obvious cases
    if (hasXNeighbor) mayUseX = false;
    if (hasYNeighbor) mayUseY = false;
    if (hasZNeighbor) mayUseZ = false;

    // sort ouf what's facing away
    if (dotX <= 0.0) mayUseX = false;
    if (dotY <= 0.0) mayUseY = false;
    if (dotZ <= 0.0) mayUseZ = false;

    if (! edgeX && ! edgeY && ! edgeZ)
    {
        // sort out non-edges
        vec3 d = abs(pT);
        if (d.x > d.y && d.x > d.z)
        {
            mayUseY = false;
            mayUseZ = false;
        }
        else if (d.y > d.x && d.y > d.z)
        {
            mayUseX = false;
            mayUseZ = false;
        }
        else if (d.z > d.x && d.z > d.y)
        {
            mayUseX = false;
            mayUseY = false;
        }
    }
    else if (edgeX && edgeY || edgeY && edgeZ || edgeX && edgeZ)
    {
        // sort out corners
        if (hasXNeighbor)
        {
            if (! hasYNeighbor && dotY > 0.0) mayUseZ = false;
            else mayUseY = false;
        }
        if (hasYNeighbor)
        {
            if (! hasXNeighbor && dotX > 0.0) mayUseZ = false;
            else mayUseX = false;
        }
        if (hasZNeighbor)
        {
            if (! hasXNeighbor && dotX > 0.0) mayUseY = false;
            else mayUseX = false;
        }
    }
    else
    {
        // sort out edges
        if (edgeX) mayUseX = false;
        if (edgeY) mayUseY = false;
        if (edgeZ) mayUseZ = false;
    }

    if (mayUseX) return surfaceNormalX;
    if (mayUseY) return surfaceNormalY;
    if (mayUseZ) return surfaceNormalZ;

    // this should not happen (no normal left to use)
    debug = 2;
    return surfaceNormalX;
}

vec3 computeLighting(vec3 origin, vec3 rayDirection, vec3 p, vec3 surfaceNormal)
{
    vec3 ambience = vec3(0.2) * (enableAmbientOcclusion && fastDistance(p, origin) < 100.0
                    ? ambientOcclusion(p, createSurfaceTrafo(surfaceNormal), 0.2)
                    : 1.0);
    vec3 light = phongShading(0, origin, p, ambience, surfaceNormal, 1.0);

    return light;
}

/* Processes the channels for composing the final image. This method can be called repeatedly
 * for deeper ray tracing and viewing depth.
 */
Channels processChannels(Channels channels)
{
    if (channels.final)
    {
        // this pixel is final and doesn't need further processing
        return channels;
    }

    vec3 origin = channels.indirectP;
    vec3 incomingRayDirection = channels.rayDirection;

    vec4 depthPoint = raymarch(origin, channels.rayDirection, 9999.0);
    if (channels.bounces == 0)
    {
        channels.p = depthPoint.xyz;
        channels.indirectP = depthPoint.xyz;
    }
    else
    {
        channels.indirectP = depthPoint.xyz;
    }
    channels.totalDistance += fastDistance(origin, depthPoint.xyz);

    if (depthPoint.w > 0.0)
    {
        uint vtype = voxelType(depthPoint.xyz);

        float dist = channels.totalDistance;
        vec3 surfaceNormal = getCorrectedBoxNormals(depthPoint.xyz, channels.rayDirection);
        Material material = getObjectMaterial(depthPoint.xyz, dist, surfaceNormal);
        vec3 bumpNormal = (createSurfaceTrafo(surfaceNormal) * vec4(material.normal, 1.0)).xyz;

        channels.surfaceNormal = bumpNormal;
        channels.light += computeLighting(channels.origin, channels.rayDirection, depthPoint.xyz, bumpNormal);
        channels.albedo *= material.color;

        if (material.roughness < 1.0)
        {
            // reflect ray
            channels.rayDirection = reflect(channels.rayDirection, channels.surfaceNormal);

            // epsilon must be small for corners, or you'll get reflected far into another block
            channels.indirectP += channels.rayDirection * 0.01;

            if (hasVoxelAt(channels.indirectP))
            {
                // stuck in an object, not good...
                //debug = 2;
                // this is a workaround suitable for water...
                channels.indirectP -= channels.rayDirection * 0.01;
                channels.rayDirection.y *= -1.0;
                channels.indirectP += channels.rayDirection * 0.01;
            }

            // TODO: do something with the roughness
            float fresnel = pow(clamp(1.0 - dot(channels.surfaceNormal, channels.rayDirection * -1.0), 0.5, 1.0), 1.0);
            channels.light *= fresnel;

            channels.origin = channels.indirectP;
            ++channels.bounces;
        }
        else if (material.ior > 0.0)
        {
            vec3 refractedRay = refr(channels.rayDirection, channels.surfaceNormal, material.ior);
            if (length(refractedRay) < 0.0001)
            {
                // total internal reflection
                channels.rayDirection = normalize(reflect(channels.rayDirection, channels.surfaceNormal));
            }
            else
            {
                //insideObject = ! insideObject;
                channels.rayDirection = normalize(refractedRay);
            }
            channels.indirectP += channels.rayDirection * 0.01;

            channels.origin = channels.indirectP;
            ++channels.bounces;
        }
        else
        {
            channels.final = true;
        }
    }

    return channels;
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
    tasmRegisters[REG_ENV_UNIVERSE_X] = universeLocation.x;
    tasmRegisters[REG_ENV_UNIVERSE_Y] = universeLocation.y;
    tasmRegisters[REG_ENV_UNIVERSE_Z] = universeLocation.z;

    if (screenWidth < 16.0 || screenHeight < 16.0)
    {
        fragColor = vec4(1.0, 0.0, 0.0, 1.0);
        return;
    }

    float aspect = screenWidth / screenHeight;
    vec2 pixelSize = vec2(1.0) / vec2(screenWidth, screenHeight);
    float exposure = 1.0;

    vec3 cameraPosition = vec3(0, 0, -0.1);

    // transform the camera location and orientation
    vec3 viewOrigin = (cameraTrafo * vec4(cameraPosition, 1.0)).xyz;
    vec3 screenPoint = (cameraTrafo * vec4(uv.x, uv.y / aspect, 1.0, 1.0)).xyz;
    // shoot a ray from the camera onto the near plane (screen)
    vec3 rayDirection = normalize(screenPoint - viewOrigin);

    Channels channels;
    vec3 currentOrigin = viewOrigin;

    channels.origin = viewOrigin;
    channels.indirectP = viewOrigin;
    channels.rayDirection = rayDirection;
    channels.albedo = vec3(1.0);
    channels.light = vec3(0.0);

    for (int i = 0; i < 16; ++i)
    {
        if (channels.final)
        {
            break;
        }
        else if (i >= tracingDepth)
        {
            // hit the sky
            channels.albedo *= skyBox(channels.indirectP, channels.rayDirection).rgb;
            channels.light += getLightColor(0);

            break;
        }

        channels = processChannels(channels);
    }
    channels.outline = freeEdge || aoEdge;

    float dist = fastDistance(viewOrigin, channels.indirectP);

    if (tasmProgramTooLong)
    {
        fragColor = vec4(1.0, 0.0, 0.0, 1.0);
    }
    else if (tasmStackOutOfBounds)
    {
        fragColor = vec4(1.0, 1.0, 0.0, 1.0);
    }
    else if (debug != 0)
    {
        fragColor = vec4(1.0, 0.0, 1.0, 1.0);
    }
    else if (renderChannel == DEPTH_BUFFER_CHANNEL)
    {
        float depthLevel = min(dist, 150.0) / 150.0;
        fragColor = vec4(vec3(depthLevel), 1.0);
    }
    else if (renderChannel == NORMALS_CHANNEL)
    {
        fragColor = vec4(vec3(0.5) + channels.surfaceNormal * 0.5, 1.0);
    }
    else if (renderChannel == OUTLINES_CHANNEL)
    {
        //fragColor = vec4(vec3(float(skipCount) / 128.0), 1.0);
        //fragColor = vec4(vec3(channels.final ? 1.0 : 0.0), 1.0);
        fragColor = vec4(vec3(channels.outline ? 0.0 : 1.0), 1.0);
    }
    else if (renderChannel == LIGHTING_CHANNEL)
    {
        fragColor = vec4(channels.light, 1.0);
    }
    else if (renderChannel == COLORS_CHANNEL)
    {
        fragColor = vec4(channels.albedo, 1.0);
    }
    else
    {
        vec3 composed = enableOutlines && channels.outline ? vec3(0.0)
                                                           : (channels.light * channels.albedo);

        // apply distance fog
        float clampedDist = clamp(dist, 0.0, 400.0);
        float heightDensity = max(0.0, (500.0 - abs(viewOrigin.y - channels.indirectP.y)) / 500.0);
        float fogDensity = min(1.0, max(0.0, clampedDist - 350.0) * 0.02 * heightDensity);
        composed = vec3(
            lerp(composed.r, DISTANCE_FOG_COLOR.r, fogDensity),
            lerp(composed.g, DISTANCE_FOG_COLOR.g, fogDensity),
            lerp(composed.b, DISTANCE_FOG_COLOR.b, fogDensity)
        );

        fragColor = vec4(gammaCorrection(composed * exposure), 1.0);
    }

}

