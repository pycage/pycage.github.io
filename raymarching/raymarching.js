const mat = await shRequire("shellfish/core/matrix");

/* Transforms a world-space point into object space.
 */
function transformPoint(world, p, obj)
{
    // transform p around the object
    const objTrafoInv = mat.t(
        mat.fromArray(
            world.data.slice(obj * world.objectSize + 20, obj * world.objectSize + 36),
            4)
    );

    return mat.swizzle(mat.mul(objTrafoInv, mat.vec(p, 1.0)), "xyz");
}
exports.transformPoint = transformPoint;

function sdf(world, obj, p)
{
    const objType = world.data[obj * world.objectSize];
    const objRadius = world.data[obj * world.objectSize + 1];
    if (objType === 0)
    {
        // plane
        return mat.swizzle(p, "y") - mat.swizzle(mat.vec(0, 0, 0), "y");
    }
    else if (objType === 1)
    {
        // sphere
        return mat.distance(p, mat.vec(0, 0, 0)) - objRadius;
        
    }
    else if (objType === 2)
    {
        // box
        const halfSides = mat.vec(objRadius, objRadius, objRadius);
        const pt = mat.sub(p, mat.vec(0, 0, 0));
        const q = mat.sub(mat.elementWise(pt, 0, mat.ABS), halfSides);
        return mat.length(mat.elementWise(q, 0.0, mat.MAX)) + Math.min(Math.max(q[0][0], Math.max(q[1][0], q[2][0])), 0.0);
    }
    else
    {
        // generic/unknown
        return mat.distance(p, mat.vec(0, 0, 0)) - objRadius;
    }
}
exports.sdf = sdf;

function nearestObject(world, p)
{
    let foundObject = -1;
    let d = 9999.0;

    for (let i = 0; i < world.size; ++i)
    {
        const obj = i;
        const dist = exports.sdf(world, obj, exports.transformPoint(world, p, obj));
        if (dist < d)
        {
            d = dist;
            foundObject = obj;
        }
    }

    return { obj: foundObject, distance: d };
}
exports.nearestObject = nearestObject;

/*
function rayMarch(world, origin, rayDirection, maxDistance)
{
    let distance = 0.0;

    for (int i = 0; i < 128; ++i)
    {
        if (distance > maxDistance)
        {
            break;
        }

        const checkPoint = mat.add(origin, mat.mul(rayDirection * distance));
        const objectAndDistance = nearestObject(world, checkPoint);

        const safeDist = objectAndDistance.distance;

        if (safeDist > 0.001)
        {
            // no hit
            distance += safeDist;
        }
        else
        {
            return { obj: objectAndDistance.obj, distance: distance };
        }
    }

    return { obj: -1, distance: 9999.0 };
}
*/