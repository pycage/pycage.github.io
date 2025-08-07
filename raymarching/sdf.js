const mat = await shRequire("shellfish/core/matrix");

function sdf(shape, radius, p)
{
    if (shape === 0)
    {
        // plane
        return mat.swizzle(p, "y") - mat.swizzle(mat.vec(0, 0, 0), "y");
    }
    else if (shape === 1)
    {
        // sphere
        return mat.distance(p, mat.vec(0, 0, 0)) - radius;
    }
    else if (shape === 2)
    {
        // box
        const halfSides = mat.vec(radius, radius, radius);
        const pt = mat.sub(p, mat.vec(0, 0, 0));
        const q = mat.sub(mat.elementWise(pt, 0, mat.ABS), halfSides);
        return mat.length(mat.elementWise(q, 0.0, mat.MAX)) + Math.min(Math.max(q[0][0], Math.max(q[1][0], q[2][0])), 0.0);
    }
    else
    {
        // generic/unknown
        return mat.distance(p, mat.vec(0, 0, 0)) - radius;
    }
}
exports.sdf = sdf;
