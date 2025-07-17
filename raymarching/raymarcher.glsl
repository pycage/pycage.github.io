#version 300 es
precision mediump float;

in vec2 uv;
out vec4 fragColor;

uniform int timems;

uniform int marchingDepth;
uniform int tracingDepth;
uniform int pathTracingDepth;

uniform bool enablePhongShading;
uniform bool enableToonEffect;

uniform mat4 cameraTrafo;

uniform float aspect;

uniform int numLights;
uniform int numObjects;
uniform sampler2D materialsData;
uniform sampler2D lightsData;
uniform sampler2D worldData;

uniform sampler2D skyTexture;

#define PlaneType 0
#define SphereType 1
#define BoxType 2
#define LensType 3

int[1000] typeCache;
float[1000] radiusCache;
int sdfCount = 0;
int objectsOnRayCount = 0;
int[1000] objectsOnRay;
int numObjectsOnRay = 0;

float randomSeed = 0.0;

float random (vec2 st)
{
    st += vec2(randomSeed);
    randomSeed += 1.0;
    return fract(sin(dot(st.xy, vec2(12.9898,78.233))) * 43758.5453123);
}

float random2 (vec2 st)
{
    return fract(sin(dot(st.xy, vec2(12.9898,78.233))) * 43758.5453123);
}

mat2 rotate2d(float angle)
{
    return mat2(cos(angle), -sin(angle),
                sin(angle), cos(angle));
}

vec2 wrapSt(vec2 st)
{
    float s = st.s;
    float t = st.t;
    /*
    */
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

vec2 mosaic(vec2 st, float size)
{
    return vec2(ceil(st.x * size) / size, ceil(st.y * size) / size);
}

float procLine(vec2 st, float start, float end)
{
    return step(st.y, end) * (1.0 - step(st.y, start));
}

float procLines(vec2 pos, float b)
{
    float scale = 10.0;
    pos *= scale;
    return smoothstep(0.0, 0.5 + b * 0.5, abs((sin(pos.x * 3.1415) + b * 2.0)) * 0.5);
}

float procWhiteNoise(vec2 st)
{
    return fract(sin(dot(st.xy, vec2(12.9898,78.233))) * 43758.5453123);
}

float procNoise(vec2 st)
{
    vec2 i = floor(st);
    vec2 f = fract(st);
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(random2(i  + vec2(0.0, 0.0)),
                   random2(i + vec2(1.0, 0.0)), u.x),
               mix(random2(i + vec2(0.0, 1.0)),
                   random2(i + vec2(1.0, 1.0)), u.x), u.y);
}

float procCheckerboard(vec2 st)
{
    float value1 = step(st.x, 0.5);
    float value2 = step(st.y, 0.5);
    return min(value1, value2) + (1.0 - max(value1, value2));
}

float procSteppedSin(vec2 st, float steps)
{
    return floor(steps * sin(st.x * 3.14195)) / steps;
}

float procSteppedPyramid(vec2 st, float steps)
{
    float value1 = procSteppedSin(st, steps);
    float value2 = procSteppedSin(st.yx, steps);
    return min(value1, value2);
}

float procTriangle(vec2 st)
{
    return step(clamp(st.x - st.y, 0.0, 1.0), 0.0);
}

float procRipple(vec2 st, float p)
{
    return 0.5 + sin(
        pow(
            pow(abs(st.s), p) + pow(abs(st.t), p),
            (1.0 / p)
        )
    ) / 2.0;
}

float procWaves(vec2 st)
{
    float e = 2.7183;
    return (pow(e, sin(st.s) * cos(st.t)) / (e * e));
}

mat3 proceduralTextureCheckerboard(vec2 st, vec3 colorA, vec3 colorB)
{
    //float value1 = procWhiteNoise(st) * 0.3;
    float value2 = procCheckerboard(st) * 0.7;
    float value = value2; //value1 + value2;

    vec3 color = value < 0.5 ? colorA * (value + 0.5) : colorB * value;
    return mat3(color, vec3(0.0, 1.0, 0.0), vec3(0.0));
}

mat3 proceduralTextureBricks(vec2 st)
{
    float value1 = procLine(st, 0.0, 0.025);
    float value2 = procLine(st, 0.975, 1.0);
    float value3 = procLine(st, 0.47, 0.52);

    float value4 = procLine(st.yx, 0.2, 0.25) * procLine(st, 0.0, 0.47);
    float value5 = procLine(st.yx, 0.65, 0.7) * procLine(st, 0.52, 1.0);

    float mask = min(1.0, value1 + value2 + value3 + value4 + value5);

    float noise1 = 0.8 + procWhiteNoise(st) * 0.2;
    float noise2 = 0.5 + procWhiteNoise(st) * 0.3;

    vec3 color1 = vec3(0.95, 0.95, 0.77) * noise1;
    vec3 color2 = vec3(0.67, 0.44, 0.44) * noise2;
    vec3 color = mask > 0.0 ? color1 : color2;

    return mat3(color, vec3(0.0, 1.0, 0.0), vec3(0.0));
}

mat3 proceduralTextureNoise(vec2 st)
{
    vec3 color = vec3(fract(sin(dot(st.xy, vec2(12.9898,78.233))) * 43758.5453123));
    return mat3(color, vec3(0.0, 1.0, 0.0), vec3(0.0));
}

mat3 proceduralTextureRipple(vec2 st, vec3 color)
{
    //float v = procRipple(st, 3.0);
    float v = procWaves(st * 100.0);
    return mat3(color * v, vec3(v, v, 0.0), vec3(0.0));
}

mat3 proceduralTextureWood(vec2 st, vec3 colorA, vec3 colorB)
{
    // from https://thebookofshaders.com/edit.php#11/wood.frag
    vec2 pos = st.yx * vec2(10.0, 3.0);

    float pattern = pos.x;

    // Add noise
    pos = rotate2d(procNoise(pos)) * pos;

    // Draw lines
    pattern = procLines(pos, 0.5);

    return mat3(colorA * pattern, vec3(0.0, 1.0, 0.0), vec3(0.0));
}

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

vec4 getMaterialColor(int n)
{
    /*
    switch (n)
    {
    case 0:
        return vec4(1.0, 0.0, 0.0, 0.0);
    case 1:
        return vec4(0.0, 1.0, 0.0, 0.0);
    case 2:
        return vec4(0.0, 0.0, 1.0, 0.0);
    default:
        return vec4(1.0, 0.0, 1.0, 0.0);
    }
    */

    int pos = n * 2;
    return texelFetch(materialsData, ivec2(pos, 0), 0);
}

int getMaterialTexture(int n)
{
    int pos = n * 2;
    return int(texelFetch(materialsData, ivec2(pos, 0), 0).a);
}

vec4 getMaterialProperties(int n)
{
    int pos = n * 2;
    return texelFetch(materialsData, ivec2(pos + 1, 0), 0);
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

int getObjectType(int n)
{
    int pos = n * 9;
    return int(round(texelFetch(worldData, ivec2(pos, 0), 0).r));
}

int getObjectMaterial(int n)
{
    int pos = n * 9;
    return int(texelFetch(worldData, ivec2(pos, 0), 0).z);
}

vec4 getObjectProperties(int n)
{
    int pos = n * 9;
    return texelFetch(worldData, ivec2(pos, 0), 0);
}

mat4 getObjectTrafo(int n)
{
    int pos = n * 9;
    vec4 vx = texelFetch(worldData, ivec2(pos + 1, 0), 0);
    vec4 vy = texelFetch(worldData, ivec2(pos + 2, 0), 0);
    vec4 vz = texelFetch(worldData, ivec2(pos + 3, 0), 0);
    vec4 vt = texelFetch(worldData, ivec2(pos + 4, 0), 0);

    mat4 rotY = rotationY(texelFetch(worldData, ivec2(pos, 0), 0).w);

    return mat4(vx, vy, vz, vt) * rotY;
}

mat4 getObjectInverseTrafo(int n)
{
    int pos = n * 9;
    vec4 vx = texelFetch(worldData, ivec2(pos + 5, 0), 0);
    vec4 vy = texelFetch(worldData, ivec2(pos + 6, 0), 0);
    vec4 vz = texelFetch(worldData, ivec2(pos + 7, 0), 0);
    vec4 vt = texelFetch(worldData, ivec2(pos + 8, 0), 0);

    mat4 rotY = rotationY(-texelFetch(worldData, ivec2(pos, 0), 0).w);

    return rotY * mat4(vx, vy, vz, vt);
}

vec3 refl(vec3 ray, vec3 surfaceNormal)
{
    float dp = dot(ray, surfaceNormal);
    return normalize(ray - surfaceNormal * 2.0 * dp);
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

float sdf(int obj, vec3 p)
{
    /* SDF are courtesy of I~nigo Quilez:
     * https://iquilezles.org/articles/distfunctions/
     */

    float radius = radiusCache[obj]; //getObjectProperties(obj).y;
    //switch (getObjectType(obj))
    switch (typeCache[obj])
    {
    case BoxType:
        vec3 halfSides = vec3(radius) * 0.5;
        vec3 pt = p - vec3(0.0);
        vec3 q = abs(pt) - halfSides;
        float dist = length(max(q, 0.0)) - min(0.0, max(max(q.x, q.y), q.z));
        // create a hollow shell
        //return abs(dist) - 0.05;
        return dist;

    case PlaneType:
        return p.y;

    case SphereType:
        return length(p - vec3(0.0, 0.0, 0.0)) - radius;

    case LensType:
        float s1 = length(p - vec3(-radius / 2.0, 0.0, 0.0)) - radius;
        float s2 = length(p - vec3(+radius / 2.0, 0.0, 0.0)) - radius;
        return max(s1, s2);

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

    default:
        return 9999.0;
    }
}

vec3 getSurfaceNormal(int obj, vec3 p)
{
    // p is in object space

    int type = getObjectType(obj);
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

vec3 getSurfaceMaterial(int obj, vec3 p)
{
    int material = getObjectMaterial(obj);
    vec4 color = getMaterialColor(material);
    int textureId = int(color.a);

    if (textureId != 0)
    {
        vec2 st = p.xy;

        int type = getObjectType(obj);
        if (type == PlaneType)
        {
            float x = p.x;
            float z = p.z;
            st = vec2(
                0.5 + x / 10.0,
                -(0.5 + z / 10.0)
            );
        }
        else if (type == BoxType)
        {
            vec3 n = getSurfaceNormal(obj, p);
            vec3 p2 = abs(n.y) > 0.0 ? n.zxy : n.zyx;
            float dp = dot(n, p2);
            vec3 axis1 = normalize(p2 - dp * n);
            vec3 axis2 = normalize(cross(n, axis1));
            float x = dot(p, axis1);
            float y = dot(p, axis2);
            st = vec2(x, y);
        }

        switch (textureId)
        {
        case -1:
            color = vec4(proceduralTextureCheckerboard(wrapSt(st * 3.0), vec3(0.93, 0.72, 0.19), vec3(0.29, 0.22, 0.06))[0], 1.0);
            break;
        case -2:
            color = vec4(proceduralTextureBricks(wrapSt(st * 2.0)));
            break;
        case -3:
            color = vec4(proceduralTextureNoise(wrapSt(st))[0], 1.0);
            break;
        case -4:
            color = vec4(proceduralTextureWood(wrapSt(st / 10.0), vec3(0.63, 0.49, 0.14), vec3(0.0))[0], 1.0);
            break;
        }
        
        return gammaCorrectionInverse(color.rgb);
    }
    else
    {
        return color.rgb;
    }
}

bool mayHitObject(int obj, vec3 origin, vec3 rayDirection)
{
    // vectors are in object space already

    int type = getObjectType(obj);
    if (type == PlaneType)
    {
        return (origin + rayDirection * 100.0).y < 0.0;
    }
    else
    {
        vec3 objectDirection = vec3(0.0) - origin;
        float rayDist = dot(rayDirection, objectDirection);
        if (rayDist >= 0.0)
        {
            vec3 checkPoint = origin + rayDirection * rayDist;
            float objRadius = getObjectProperties(obj).g;

            if (type == SphereType)
            {
                float dist = distance(checkPoint, vec3(0.0));
                return dist < objRadius;
            }
            else if (type == BoxType)
            {
                // non-euklidean distance
                return abs(checkPoint.x) < objRadius &&
                       abs(checkPoint.y) < objRadius &&
                       abs(checkPoint.z) < objRadius;
            }
            else
            {
                float dist = distance(checkPoint, vec3(0.0));
                return dist < objRadius * sqrt(2.0);
            }
        }
        else
        {
            return false;
        }
    }
}

void findObjectsOnRay(vec3 origin, vec3 rayDirection)
{
    numObjectsOnRay = 0;
    for (int obj = 0; obj < numObjects; ++obj)
    {
        // convert origin and ray into object space
        vec3 rayP = origin + rayDirection;
        vec3 rayPT = transformPoint(rayP, obj);

        vec3 originT = transformPoint(origin, obj);
        vec3 rayDirectionT = rayPT - originT;

        /*
        vec3 objectDirection = vec3(0.0) - originT;
        float rayDist = dot(rayDirection, objectDirection);
        vec3 checkPoint = originT + rayDirectionT * rayDist;

        float dist = distance(checkPoint, vec3(0.0));

        float objRadius = getObjectProperties(obj).g;
        if (dist < objRadius * 2.0)
        */
        if (mayHitObject(obj, originT, rayDirectionT))
        {
            objectsOnRay[numObjectsOnRay] = obj;
            ++numObjectsOnRay;
        }
    }
}

vec2 nearestDistance(vec3 p)
{
    int foundObject = -1;
    float d = 9999.0;    

    for (int i = 0; i < numObjectsOnRay; ++i)
    {
        int obj = objectsOnRay[i];
        float dist = sdf(obj, transformPoint(p, obj));
        if (dist < d)
        {
            d = dist;
            foundObject = obj;
        }
    }

    /*
    for (int obj = 0; obj < numObjects; ++obj)
    {
    }
    */

    return vec2(float(foundObject), d);
}

vec2 rayMarch(vec3 origin, vec3 rayDirection, bool insideObject, float maxDistance, float accuracy)
{
    // this is an essential optimization to reduce the number of objects to check
    findObjectsOnRay(origin, rayDirection);
    objectsOnRayCount = numObjectsOnRay;

    float distance = 0.0;
    float previousObjectDistance = 9999.0;
    int previousObject = -1;
    for (int i = 0; i < marchingDepth; ++i)
    {
        ++sdfCount;
        if (distance > maxDistance)
        {
            break;
        }

        vec3 checkPoint = origin + rayDirection * distance;
        vec2 objectAndDistance = nearestDistance(checkPoint);
        int obj = int(objectAndDistance.x);
        float safeDist = objectAndDistance.y;

        if (insideObject)
        {
            safeDist *= -1.0;
        }

        // optimization for walking along convex objects:
        // if we don't come closer, forget about it
        if (obj == previousObject && safeDist >= previousObjectDistance)
        {
            // remove object from list
            for (int i = 0; i < numObjectsOnRay; ++i)
            {
                if (objectsOnRay[i] == obj && numObjectsOnRay > 1)
                {
                    objectsOnRay[i] = objectsOnRay[numObjectsOnRay - 1];
                    --numObjectsOnRay;
                    break;
                }
            }
        }

        previousObject = obj;
        previousObjectDistance = safeDist;

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

vec3 phongShading(vec3 origin, vec3 checkPoint, vec3 surfaceNormal)
{
    // Phong shading: lighting = ambient + diffuse + specular
    //                color = modelColor * lighting

    vec3 viewDirection = normalize(origin - checkPoint);
    vec3 ambience = vec3(0.0);
    //vec3 ambience = vec3(0.0, 0.0, 0.0);

    vec3 lighting = ambience;
    float shininess = 64.0;

    for (int i = 0; i < numLights; ++i)
    {
        vec3 lightLoc = getLightLocation(i);
        vec3 lightCol = getLightColor(i);
        float lightRange = getLightRange(i);

        vec3 lightDirection = normalize(lightLoc - checkPoint);
        float lightDistance = length(checkPoint - lightLoc);

        // does the light reach?
        if (lightDistance < lightRange)
        {
            float travelDist = rayMarch(checkPoint + lightDirection * 0.001, lightDirection, false, lightDistance, 0.0001).y;
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
        float diffuseImpact = max(0.0, dot(lightDirection, surfaceNormal));
        vec3 diffuse = attenuatedLight * diffuseImpact;

        // specular highlight
        vec3 specular = vec3(0.0, 0.0, 0.0);
        if (diffuseImpact > 0.0)
        {
            // Blinn-Phong
            vec3 halfDirection = normalize(lightDirection + viewDirection);
            float specularStrength = pow(max(0.0, dot(surfaceNormal, halfDirection)), shininess) * 0.5;
            specular = attenuatedLight * specularStrength;
        }

        lighting += diffuse * 0.75 + specular * 0.25;
    }

    return lighting.rgb;
}

/* Returns the color at the given screen pixel plus the ID of the object that was
 * hit in the a component. The object ID is added to the amount of traces multiplied
 * by 1000.
 */
vec4 shootRay(vec2 uv, vec3 origin, float aspect)
{
    // transform the camera location and orientation
    vec3 currentOrigin = (cameraTrafo * vec4(origin, 1.0)).xyz;
    vec3 screenPoint = (cameraTrafo * vec4(uv.x, uv.y / aspect, 1.0, 1.0)).xyz;

    int currentObject = -1;

    // shoot a ray from origin onto the near Plain (screen)
    vec3 rayDirection = normalize(screenPoint - currentOrigin);

    vec3 color = vec3(1.0);
    vec3 light = vec3(0.1);

    bool insideObject = false;

    int traceCount = 0;
    for (; traceCount < tracingDepth; ++traceCount)
    {
        vec2 objectAndDist = rayMarch(currentOrigin, rayDirection, insideObject, 1000.0, 0.0001);
        //sdfCount += numObjectsOnRay;
        float dist = objectAndDist.y;

        if (objectAndDist.x >= 0.0 && objectAndDist.y < 100.0)
        {
            // hit something
            vec3 checkPoint = currentOrigin + rayDirection * dist;
            int obj = int(objectAndDist.x);

            currentObject = obj;

            vec3 checkPointT = transformPoint(checkPoint, obj);

            int material = getObjectMaterial(obj);
            vec4 materialProperties = getMaterialProperties(material);
            vec3 materialColor = getSurfaceMaterial(obj, checkPointT);
            float roughness = materialProperties.r;
            float reflectivity = 1.0 - roughness;
            float ior = materialProperties.g;

            vec3 surfaceNormalT = getSurfaceNormal(obj, checkPointT);
            if (pathTracingDepth > 0)
            {
                surfaceNormalT += roughness * (-0.5 + random(uv) * 1.0);
            }
            vec3 surfaceNormal = transformNormalOW(surfaceNormalT, obj);
            // for debugging: show normals
            //materialColor = 0.5 + surfaceNormal * 0.5;

            vec3 lightIntensity = enablePhongShading ? phongShading(currentOrigin, checkPoint, surfaceNormal)
                                                     : vec3(1.0);

            color *= materialColor;
            light += lightIntensity;
            
            if (ior > 0.01)
            {
                // we're not finished yet - refract the ray and enter or exit the object
                vec3 refractedRay = refr(rayDirection, surfaceNormal, ior);
                if (length(refractedRay) < 0.01)
                {
                    // it's a reflection
                    rayDirection = normalize(reflect(rayDirection, surfaceNormal));
                }
                else
                {
                    insideObject = ! insideObject;
                    rayDirection = normalize(refractedRay);
                }
                currentOrigin = checkPoint + rayDirection * 0.001;
                light *= 0.5;
            }
            else if (reflectivity > 0.1)
            {
                // we're not finished yet - reflect the ray
                float fresnel = pow(clamp(1.0 - dot(surfaceNormal, rayDirection * -1.0), 0.5, 1.0), 1.0);
                rayDirection = reflect(rayDirection, surfaceNormal);
                currentOrigin = checkPoint + rayDirection * 0.001;
                light *= fresnel * reflectivity;
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
            vec3 hitPoint = currentOrigin + rayDirection * ((1000.0 - currentOrigin.y) / rayDirection.y);
            hitPoint += vec3(float(timems) / 10000.0, 0.0, 0.0);
            vec4 skyBox = texture(skyTexture, (hitPoint.xz / 10000.0));
            currentObject = -1;
            light += 0.5 * skyBox.rgb; //vec3(0.1); //vec3(0.6, 0.7, 0.9);
            break;
        }

    }

    return vec4(color * light, float(traceCount * 1000 + currentObject));
}

void main()
{
    for (int i = 0; i < numObjects; ++i)
    {
        vec4 props = getObjectProperties(i);
        typeCache[i] = int(props.x);
        radiusCache[i] = props.y;
    }

    /*
    if (! enablePhongShading)
    {
        vec2 st = 0.5 + uv / 2.0;
        //st = wrapSt(st * 12.0);

        float value1 = procLine(st, 0.0, 0.025);
        float value2 = procLine(st, 0.975, 1.0);
        float value3 = procLine(st, 0.47, 0.52);

        float value4 = procLine(st.yx, 0.2, 0.25) * procLine(st, 0.0, 0.47);
        float value5 = procLine(st.yx, 0.65, 0.7) * procLine(st, 0.52, 1.0);

        float mask1 = min(1.0, value1 + value2 + value3 + value4 + value5);
        float mask2 = 1.0 - mask1;

        float noise1 = 0.8 + procWhiteNoise(st) * 0.2;
        float noise2 = 0.5 + procWhiteNoise(st) * 0.3;

        //float value2 = procLine(wrapSt(st.yx - 0.5));
        //float value2 = procCheckerboard(st) * 0.7;
        float value = min(1.0, noise1 * mask1 + noise2 * mask2);
        vec3 color1 = vec3(0.95, 0.95, 0.77) * noise1;
        vec3 color2 = vec3(0.67, 0.44, 0.44) * noise2;
        fragColor = vec4(mask1 > 0.0 ? color1 : color2, 1.0);
        return;
    }
    */


    float exposure = 1.0;

    vec3 origin = vec3(0, 0, -0.1);
    float probeDist = 0.0001;

    vec4 probe = shootRay(uv, origin, aspect);
    vec3 pixel = probe.rgb;

    for (int i = 0; i < pathTracingDepth; ++i)
    {
        //pixel += shootRay(uv + vec2(probeDist, 0.0), origin, aspect);
        pixel += shootRay(uv, origin, aspect).rgb;
    }
    pixel /= float(pathTracingDepth + 1);

    pixel *= exposure;

    if (enableToonEffect)
    {   
        int obj1 = int(probe.w);
        int obj2 = int(shootRay(uv + 0.005, origin, aspect).w);

        if (obj1 != obj2)
        {
            pixel = vec3(0.0, 0.0, 0.0);
        }
        else
        {
            pixel = flattenColor(pixel, 4);
        }
    }

    pixel = gammaCorrection(pixel);
    fragColor = vec4(pixel, 1.0);
    //fragColor = vec4(objectsOnRayCount >= 20 ? 1.0 : pixel.r, pixel.g, pixel.b, 1.0);
    //fragColor = vec4(sdfCount >= marchingDepth ? 1.0 : pixel.r, pixel.g, pixel.b, 1.0);
}
