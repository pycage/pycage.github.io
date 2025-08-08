const core = await shRequire("shellfish/core");
const mat = await shRequire("shellfish/core/matrix");

function readArrayAt(arr, pos, size)
{
    const valueArr = [];
    for (let i = 0; i < size; ++i)
    {
        valueArr[i] = arr[pos + i];
    }
    return valueArr;
}

function writeArrayAt(arr, pos, valueArr)
{
    for (let i = 0; i < valueArr.length; ++i)
    {
        arr[pos + i] = valueArr[i];
    }
}

function makeCubeLocator(loc, cubeSize)
{
    const pathX = Math.floor((loc[0][0]) / cubeSize);
    const pathY = Math.floor((loc[1][0]) / cubeSize);
    const pathZ = Math.floor((loc[2][0]) / cubeSize);
    return (pathX << 12) + (pathY << 6) + pathZ;
}

function resolveCubeLocator(cubeNr, cubeSize)
{
    const pathX = cubeNr >> 12;
    const pathY = (cubeNr >> 6) % 64;
    const pathZ = cubeNr % 64;

    return mat.vec(
        pathX * cubeSize,
        pathY * cubeSize,
        pathZ * cubeSize
    );
}

function makeObjectLocator(locInCube)
{
    const ox = Math.floor(locInCube[0][0]);
    const oy = Math.floor(locInCube[1][0]);
    const oz = Math.floor(locInCube[2][0]);
    return (ox << 4) + (oy << 2) + oz;
}

function resolveObjectLocator(objLoc)
{
    return mat.vec(
        (objLoc >> 4) + 0.5,
        ((objLoc >> 2) & 3) + 0.5,
        (objLoc & 3) + 0.5
    );
}

function makeWorldLocator(cubeNr, objLoc)
{
    return (cubeNr << 6) + objLoc;
}

function resolveWorldLocator(wl)
{
    return {
        cube: wl >> 6,
        object: wl % 64
    };
}


const d = new WeakMap();

class World extends core.Object
{
    constructor()
    {
        super();
        d.set(this, {
            // - num objects (1)
            // - link radius (1)
            // - cube trafo (16)
            // - inv cube trafo (16)
            // - objects ...  [<type, material, radius, any> (1) + trafo (16) + invTrafo (16)]
            worldData: new Float32Array(4096 * 4096 * 4),
            cubeSize: 4.0,
            cubeDataStride: 64 * 4,
            objectSize: 2
        });

        const priv = d.get(this);
    }

    get worldData() { return d.get(this).worldData; }

    cubeOf(loc)
    {
        return makeCubeLocator(loc, d.get(this).cubeSize);
    }

    cubeLocation(cubeNr)
    {
        return resolveCubeLocator(cubeNr, d.get(this).cubeSize);
    }

    cubeTrafo(cubeNr)
    {
        return mat.translationM(this.cubeLocation(cubeNr));
    }

    cubeTrafoInverse(cubeNr)
    {
        return mat.translationM(mat.mul(this.cubeLocation(cubeNr), -1));
    }

    cubesOnRay(origin, rayDirection)
    {
        const cubeSize = d.get(this).cubeSize;

        const cubes = [];

        let p = origin;
        const originCube = this.cubeOf(p);
        cubes.push(originCube);

        //console.log("ray: " + rayDirection.flat().map(c => c.toFixed(2)));

        const scaleX = rayDirection[0][0] != 0.0 ? cubeSize / Math.abs(rayDirection[0][0]) : 9999999.0;
        const scaleY = rayDirection[1][0] != 0.0 ? cubeSize / Math.abs(rayDirection[1][0]) : 9999999.0;
        const scaleZ = rayDirection[2][0] != 0.0 ? cubeSize / Math.abs(rayDirection[2][0]) : 9999999.0;

        //console.log("scale: " + mat.vec(scaleX, scaleY, scaleZ).flat().map(c => c.toFixed(2)));

        let gridX = cubeSize * Math.floor((p[0][0]) / cubeSize);
        let gridY = cubeSize * Math.floor((p[1][0]) / cubeSize);
        let gridZ = cubeSize * Math.floor((p[2][0]) / cubeSize);

        if (rayDirection[0][0] > 0.0)
        {
            gridX += cubeSize;
        }
        if (rayDirection[1][0] > 0.0)
        {
            gridY += cubeSize;
        }
        if (rayDirection[2][0] > 0.0)
        {
            gridZ += cubeSize;
        }
        //console.log("grid: " + mat.vec(gridX, gridY, gridZ).flat().map(c => c.toFixed(2)));

        let distX = Math.abs(gridX - p[0][0]);
        let distY = Math.abs(gridY - p[1][0]);
        let distZ = Math.abs(gridZ - p[2][0]);

        //console.log("dist: " + mat.vec(distX, distY, distZ).flat().map(c => c.toFixed(2)));

        for (let i = 0; i < 5; ++i)
        {
            const rayLengthX = distX * scaleX;
            const rayLengthY = distY * scaleY;
            const rayLengthZ = distZ * scaleZ;

            let moveX = 0.0;
            let moveY = 0.0;
            let moveZ = 0.0;

            if (rayLengthX <= rayLengthY && rayLengthX <= rayLengthZ)
            {
                moveX = Math.sign(rayDirection[0][0]) * cubeSize;
                distX += cubeSize;
            }
            else if (rayLengthY <= rayLengthX && rayLengthY <= rayLengthZ)
            {
                moveY = Math.sign(rayDirection[1][0]) * cubeSize;
                distY += cubeSize;
            }
            else if (rayLengthZ <= rayLengthX && rayLengthZ <= rayLengthY)
            {
                moveZ = Math.sign(rayDirection[2][0]) * cubeSize;
                distZ += cubeSize;
            }

            p = mat.add(p, mat.vec(moveX, moveY, moveZ));

            const cube = this.cubeOf(p);
            cubes.push(cube);
        }

        return cubes;
    }

    /* Returns the list of objects in the given cube.
     */
    objectsInCube(cubeNr)
    {
        const priv = d.get(this);
        
        const cubeOffset = cubeNr * priv.cubeDataStride;
        const numObjects = priv.worldData[cubeOffset];
        
        let objects = [];
        for (let i = 0; i < numObjects; ++i)
        {
            const objOffset = cubeOffset + 8 + i * priv.objectSize;

            const objLoc = priv.worldData[objOffset];
            const p = resolveObjectLocator(objLoc);
            const objTrafo = mat.translationM(p);
            const objTrafoInverse = mat.translationM(mat.mul(p, -1));

            objects.push({
                type: 2, //priv.worldData[objOffset],
                material: priv.worldData[objOffset + 1],
                radius: 1.0, //priv.worldData[objOffset + 2],
                trafo: objTrafo,
                trafoInverse: objTrafoInverse
            });
        }

        return objects;
    }

    setSuperCubeEmpty(loc, level, empty)
    {
        const priv = d.get(this);
        const superCubeNr = makeCubeLocator(loc, 4 << level);
        const cubeOffset = (3000 + level) * 4096 * 4 + superCubeNr;
        priv.worldData[cubeOffset] = empty ? 0 : 1;
    }

    isSuperCubeEmpty(loc, level)
    {

    }

    addObject(type, material, loc, radius)
    {
        const priv = d.get(this);

        const cubeNr = this.cubeOf(loc);
        
        const cubeOffset = cubeNr * priv.cubeDataStride;
        const numObjects = priv.worldData[cubeOffset];
        let patternHi = priv.worldData[cubeOffset + 4];
        let patternHiMid = priv.worldData[cubeOffset + 5];
        let patternLoMid = priv.worldData[cubeOffset + 6];
        let patternLo = priv.worldData[cubeOffset + 7];
        //console.log("Loc: " + JSON.stringify(loc) + ", Cube: " + cubeNr + ", num: " + numObjects);
        if (numObjects === undefined)
        {
            console.log("world capacity exceeded");
        }
        
        const cubeLoc = this.cubeLocation(cubeNr);
        const locInCube = mat.sub(loc, cubeLoc);
        const objLoc = makeObjectLocator(locInCube);
        //console.log("CubeLoc: " + JSON.stringify(cubeLoc));
        
        if (numObjects < 16)
        {
            patternLo |= 1 << numObjects;
        }
        else if (numObjects < 32)
        {
            patternLoMid |= 1 << (numObjects - 16);
        }
        else if (numObjects < 48)
        {
            patternHiMid |= 1 << (numObjects - 32);
        }
        else
        {
            patternHi |= 1 << (numObjects - 48);
        }

        const objOffset = cubeOffset + 2 * 4 + numObjects * priv.objectSize;

        priv.worldData[objOffset] = objLoc;
        //priv.worldData[objOffset] = type;
        priv.worldData[objOffset + 1] = material;
        //priv.worldData[objOffset + 2] = 1.0;
        //writeArrayAt(priv.worldData, objOffset + 4, mat.flat(mat.t(objTrafo)));
        //writeArrayAt(priv.worldData, objOffset + 20, mat.flat(mat.t(mat.inv(objTrafo))));

        priv.worldData[cubeOffset] = numObjects + 1;
        //console.log("lo: " + patternLo + " hi: " + patternHi);
        priv.worldData[cubeOffset + 4] = patternHi;
        priv.worldData[cubeOffset + 5] = patternHiMid;
        priv.worldData[cubeOffset + 6] = patternLoMid;
        priv.worldData[cubeOffset + 7] = patternLo;

        // update cubetree
        this.setSuperCubeEmpty(loc, 1, false);
        this.setSuperCubeEmpty(loc, 2, false);
        this.setSuperCubeEmpty(loc, 3, false);

        const id = makeWorldLocator(cubeNr, objLoc);
        return id;
    }

    removeObject(id)
    {
        const r = resolveWorldLocator(id);
        const objIdx = r.object;
        const cubeNr = r.cube;

        const cubeOffset = cubeNr * priv.cubeDataStride;
        const objOffset = (cubeOffset + 4) * objIdx * priv.objectSize;
        priv.worldData[objOffset] = 0;
    }
};
exports.World = World;