const mat = await shRequire("shellfish/core/matrix");

/* SDF functions are courtesy of Iñigo Quilez:
 * https://iquilezles.org/articles/distfunctions/
 */

function sdfBox(p)
{
    // a box of side-length 1
    const halfSides = mat.vec(1.0, 1.0, 1.0);
    const pt = mat.sub(p, mat.vec(0, 0, 0));
    const q = mat.sub(mat.elementWise(pt, 0, mat.ABS), halfSides);
    return mat.length(mat.elementWise(q, 0.0, mat.MAX)) + Math.min(Math.max(q[0][0], Math.max(q[1][0], q[2][0])), 0.0);
}
exports.sdfBox = sdfBox;

function sdfPlane(p)
{
    return mat.swizzle(p, "y") - mat.swizzle(mat.vec(0, 0, 0), "y");
}
exports.sdfPlane = sdfPlane;

function sdfSphere(p)
{
    // a unit sphere of radius 1
    return mat.distance(p, mat.vec(0, 0, 0)) - 1.0;
}
exports.sdfSphere = sdfSphere;


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