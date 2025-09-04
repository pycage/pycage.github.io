const core = await shRequire("shellfish/core");
const mat = await shRequire("shellfish/core/matrix");
const sdf = await shRequire("./sdf.js");
const terrain = await shRequire("./wasm/terrain.wasm");

// the side-length of the horizone cube in sectors (must be odd so there is a center)
const HORIZON_SIZE = 5;
// the side-length of a sector in cubes
const SECTOR_SIZE = 16;
// the side-length of a cube in voxels
const CUBE_SIZE = 4;
const SECTOR_LINES = 17;
const VOXEL_DATA_OFFSET = SECTOR_SIZE * SECTOR_SIZE * SECTOR_SIZE;
const CUBE_VOXEL_STRIDE = CUBE_SIZE * CUBE_SIZE * CUBE_SIZE;

function readUint32Array(ptr, memory)
{
  const memU32 = new Uint32Array(memory.buffer);
  
  const bufPtr = memU32[ptr / 4];
  const length = memU32[(ptr + 8) / 4];

  const uint32Array = new Uint32Array(memory.buffer, bufPtr, length / 4);
  return uint32Array;
}

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

function uniteRanges(ranges)
{
    let result = [];
    for (let i = 0; i < ranges.length; ++i)
    {
        const [begin, end] = ranges[i];

        let haveIntersections = false;

        // remove all ranges that lie between begin and end
        result = result.filter(r => r[0] < begin || r[1] > end)
                       .map(r =>
        {
            if (end >= r[0] && begin <= r[1])
            {
                // intersection
                haveIntersections = true;
                return [Math.min(r[0], begin), Math.max(r[1], end)];
            }
            else
            {
                return r;
            }
        });

        if (! haveIntersections)
        {
            // add, if no intersections where found
            result.push([begin, end]);
        }
    }

    return result;
}

/* Returns the sector index at the given horizon cube coordinates.
 */
function makeSectorIndex(x, y, z)
{
    return y * HORIZON_SIZE * HORIZON_SIZE +
           z * HORIZON_SIZE +
           x;
}

/* Returns the horizon cube coordinates of the given sector.
 */
function sectorLocation(sector)
{
    const y = Math.floor(sector / (HORIZON_SIZE * HORIZON_SIZE));
    const z = Math.floor(sector % (HORIZON_SIZE * HORIZON_SIZE) / HORIZON_SIZE);
    const x = sector % HORIZON_SIZE;

    return mat.vec(x, y, z);
}

/* Returns the level of detail for the given sector.
 */
function lodOfSector(sector)
{
    const [x, y, z] = sectorLocation(sector).flat();
    const center = Math.floor(HORIZON_SIZE / 2);
    const dist = Math.max(Math.abs(x - center), Math.abs(y - center), Math.abs(z - center));

    return dist < 3 ? 0 : 1;
}

function sectorWorldLocation(sector, cubeSize)
{
    const sectorLength = SECTOR_SIZE * cubeSize;
    return mat.mul(sectorLocation(sector), sectorLength);
}

function makeCubeLocator(loc, cubeSize)
{
    const sectorLength = SECTOR_SIZE * cubeSize;

    const x = Math.floor(loc[0][0] / sectorLength);
    const y = Math.floor(loc[1][0] / sectorLength);
    const z = Math.floor(loc[2][0] / sectorLength);

    const sector = makeSectorIndex(x, y, z);

    const t = mat.mul(mat.sub(loc, sectorWorldLocation(sector, cubeSize)), 1 / cubeSize);

    return {
        x: Math.floor(t[0][0]),
        y: Math.floor(t[1][0]),
        z: Math.floor(t[2][0]),
        sector
    };
}

function resolveCubeLocator(cube, cubeSize)
{
    return mat.add(
        sectorWorldLocation(cube.sector, cubeSize),
        mat.vec(cube.x * cubeSize, cube.y * cubeSize, cube.z * cubeSize)
    );
}

function makeObjectLocator(locInCube)
{
    const ox = Math.floor(locInCube[0][0]);
    const oy = Math.floor(locInCube[1][0]);
    const oz = Math.floor(locInCube[2][0]);
    return {
        x: ox,
        y: oy,
        z: oz
    };
}

function resolveObjectLocator(objLoc)
{
    return mat.vec(objLoc.x, objLoc.y, objLoc.z);
}

function makeWorldLocator(cube, objLoc)
{
    return {
        cube,
        object: objLoc
    };
}

function uploadLinearData(canvas, begin, data)
{
    const lineLength = 4096 * 4;
    const end = begin + data.length - 1;  // inclusive
    const line1 = Math.floor(begin / lineLength);
    const col1 = begin % lineLength;
    const line2 = Math.floor(end / lineLength);
    const col2 = end % lineLength;

    if (line1 === line2)
    {
        const width = col2 - col1 + 1;
        const height = 1;
        canvas.updateSampler("worldData", col1 / 4, line1, width / 4, height, data);
    }
    else
    {
        // first line: col1 -> lineLength
        let width = lineLength - col1;
        let height = 1;
        let subBegin = 0;
        let subEnd = subBegin + width;
        //console.log("1: " + subBegin + " -> " + subEnd + ", " + width + " x " + height + ", " + data.subarray(subBegin, subEnd).length);
        canvas.updateSampler("worldData", col1 / 4, line1, width / 4, height, data.subarray(subBegin, subEnd));

        // last line: 0 -> col2
        width = col2 + 1;
        height = 1;
        subBegin = data.length - width;
        subEnd = data.length;
        //console.log("2: " + subBegin + " -> " + subEnd + ", " + width + " x " + height + ", " + data.subarray(subBegin, subEnd).length);
        canvas.updateSampler("worldData", 0, line2, width / 4, height, data.subarray(subBegin, subEnd));

        if (line1 + 1 < line2 - 1)
        {
            // inbetween
            width = lineLength;
            height = line2 - line1 - 1;
            subBegin = lineLength * 1;
            subEnd = subBegin + width * height;
            //console.log("3: " + subBegin + " -> " + subEnd + ", " + width + " x " + height + ", " + data.subarray(subBegin, subEnd).length);
            canvas.updateSampler("worldData", 0, line1 + 1, width / 4, height, data.subarray(subBegin, subEnd));
        }
    }
}


const d = new WeakMap();

class World extends core.Object
{
    constructor()
    {
        super();

        const horizonCenter = Math.floor(HORIZON_SIZE / 2);

        d.set(this, {
            worldData: new Uint32Array(4096 * 4096 * 4),
            updateQueue: [],
            sectorMap: [ ],  // array of { address, universeLocation }
            centerSector: makeSectorIndex(horizonCenter, horizonCenter, horizonCenter)
        });

        const priv = d.get(this);

        // init sector map
        for (let i = 0; i < HORIZON_SIZE * HORIZON_SIZE * HORIZON_SIZE; ++i)
        {
            priv.sectorMap.push({ address: i, uloc: mat.vec(0, 0, 0) });
        }

        this.writeSectorMap();
    }

    get worldData() { return d.get(this).worldData; }
    get sectorMap() { return d.get(this).sectorMap.map(s => s.address); }
    get centerSector() { return d.get(this).centerSector; }

    /* Maps a sector to its actual physical location.
     */
    mapSector(sector)
    {
        return d.get(this).sectorMap[sector].address;
    }

    /* Returns the sector data offset.
     */
    sectorDataOffset(sector)
    {
        return SECTOR_LINES * this.mapSector(sector) * 4096 * 4;
    }

    /* Returns the data offset for accessing a cube, relative to the sector offset.
     */
    cubeDataOffset(cubeIndex)
    {
        return cubeIndex * 4;
    }

    /* Returns the offset into the cube voxel data for the given address, relative to the
     * sector offset;
     */
    voxelDataOffset(address)
    {
        return VOXEL_DATA_OFFSET + address * CUBE_VOXEL_STRIDE;
    }

    cubeIndex(cube)
    {
        const lod = lodOfSector(cube.sector);
        const sectorLod = Math.max(0, lod - 2);
        const sectorSizeWithLod = SECTOR_SIZE / (1 << sectorLod);

        const cx = Math.floor(cube.x / (1 << sectorLod));
        const cy = Math.floor(cube.y / (1 << sectorLod));
        const cz = Math.floor(cube.z / (1 << sectorLod));

        return cx * sectorSizeWithLod * sectorSizeWithLod + cy * sectorSizeWithLod + cz;
    }

    /* Returns the sector at the given world location.
     */
    sectorAt(loc)
    {
        const sectorLength = SECTOR_SIZE * CUBE_SIZE;

        const v = mat.mul(loc, 1 / sectorLength);
        const x = Math.floor(v[0][0]);
        const y = Math.floor(v[1][0]);
        const z = Math.floor(v[2][0]);

        return makeSectorIndex(x, y, z);
    }

    /* Returns the distance from the horizon cube center to the given sector.
     */
    sectorDistance(sector)
    {
        const center = Math.floor(HORIZON_SIZE / 2);
        const [x, y, z] = sectorLocation(sector).flat();
        console.log("sector coords of " + sector + " = " + x + ", " + y + ", " + z);
        return mat.vec(x - center, y - center, z - center);
    }

    /* Returns the cube at the given world location.
     */
    cubeOf(loc)
    {
        return makeCubeLocator(loc, CUBE_SIZE);
    }

    /* Returns the world location of the given cube.
     */
    cubeLocation(cube)
    {
        return resolveCubeLocator(cube, CUBE_SIZE);
    }

    /* Returns the cube to world transformation matrix of the given cube.
     */
    cubeTrafo(cube)
    {
        return mat.translationM(this.cubeLocation(cube));
    }

    /* Returns the world to cube transformation matrix of the given cube.
     */
    cubeTrafoInverse(cube)
    {
        return mat.translationM(mat.mul(this.cubeLocation(cube), -1));
    }

    /* Returns a list of cubes on the given ray.
     */
    cubesOnRay(origin, rayDirection)
    {
        const cubeSize = CUBE_SIZE;

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

        let distX = Math.abs(gridX - p[0][0]) / cubeSize;
        let distY = Math.abs(gridY - p[1][0]) / cubeSize;
        let distZ = Math.abs(gridZ - p[2][0]) / cubeSize;

        //console.log("dist: " + mat.vec(distX, distY, distZ).flat().map(c => c.toFixed(2)));

        for (let i = 0; i < 50; ++i)
        {
            const rayLengthX = distX * scaleX;
            const rayLengthY = distY * scaleY;
            const rayLengthZ = distZ * scaleZ;

            let moveX = 0.0;
            let moveY = 0.0;
            let moveZ = 0.0;

            let newOrigin = origin;
            if (rayLengthX <= rayLengthY && rayLengthX <= rayLengthZ)
            {
                moveX = Math.sign(rayDirection[0][0]) * cubeSize;
                distX += 1.0;
                //console.log("rayLength " + rayLengthX);
                newOrigin = mat.add(origin, mat.mul(rayDirection, rayLengthX));
            }
            else if (rayLengthY <= rayLengthX && rayLengthY <= rayLengthZ)
            {
                moveY = Math.sign(rayDirection[1][0]) * cubeSize;
                distY += 1.0;
                //console.log("rayLength " + rayLengthY);
                newOrigin = mat.add(origin, mat.mul(rayDirection, rayLengthY));
            }
            else if (rayLengthZ <= rayLengthX && rayLengthZ <= rayLengthY)
            {
                moveZ = Math.sign(rayDirection[2][0]) * cubeSize;
                distZ += 1.0;
                //console.log("rayLength " + rayLengthZ);
                newOrigin = mat.add(origin, mat.mul(rayDirection, rayLengthZ));
            }

            //console.log(i + ": " + JSON.stringify(newOrigin));

            p = mat.add(p, mat.vec(moveX, moveY, moveZ));

            const cube = this.cubeOf(p);
            const cube2 = this.cubeOf(mat.add(newOrigin, mat.mul(rayDirection, 0.00001)));
            if (cube !== cube2) console.log(i + " CUBE MISMATCH: " + cube + " vs " + cube2);
            //console.log("cube: " + cube + " vs " + this.cubeOf(mat.add(newOrigin, mat.mul(rayDirection, 0.00001))));
            cubes.push(cube);
        }

        return cubes;
    }

    /* Returns the list of objects in the given cube.
     */
    objectsInCube(cube)
    {
        const priv = d.get(this);

        const lod = lodOfSector(cube.sector);
        const cubeLod = Math.min(lod, 2);
        const bitsPerCoord = 2 / (1 << cubeLod);
        
        const cubeIndex = this.cubeIndex(cube);
        const sectorOffset = this.sectorDataOffset(cube.sector);
        const cubeOffset = sectorOffset + this.cubeDataOffset(cubeIndex);
        let patternHi = priv.worldData[cubeOffset];
        let patternLo = priv.worldData[cubeOffset + 1];
        const address = priv.worldData[cubeOffset + 2];

        const voxelOffset = this.voxelDataOffset(address);

        let objects = [];
        for (let x = 0; x < 4; ++x)
        {
            for (let y = 0; y < 4; ++y)
            {
                for (let z = 0; z < 4; ++z)
                {
                    const lx = x / (1 << cubeLod);
                    const ly = y / (1 << cubeLod);
                    const lz = z / (1 << cubeLod);

                    const idx = (lx << (bitsPerCoord + bitsPerCoord)) +
                                (ly << bitsPerCoord) +
                                lz;

                    let haveObject = false;
                    if (idx < 32)
                    {
                        haveObject = patternLo & (1 << idx);
                    }
                    else
                    {
                        haveObject = patternHi & (1 << (idx - 32));
                    }
                    if (! haveObject)
                    {
                        continue;
                    }

                    const objOffset = sectorOffset + voxelOffset + idx;

                    const p = mat.vec(x, y, z);
                    const objTrafo = mat.translationM(p);
                    const objTrafoInverse = mat.translationM(mat.mul(p, -1));

                    objects.push({
                        material: priv.worldData[objOffset],
                        trafo: objTrafo,
                        trafoInverse: objTrafoInverse
                    });
                }
            }
        }

        return objects;
    }

    isLocationFree(p)
    {
        const cube = this.cubeOf(p);
        const cm = this.cubeTrafoInverse(cube);

        const hits = this.objectsInCube(cube).map(obj =>
        {
            const pT = mat.swizzle(mat.mul(mat.mul(cm, obj.trafoInverse), mat.vec(p, 1.0)), "xyz");
            return sdf.sdfBox(pT);
        })
        .filter(dist => dist <= 0.0);

        return hits.length === 0;
    }

    setSuperCubeEmpty(loc, level, empty)
    {
        const priv = d.get(this);
        const superCube = makeCubeLocator(loc, 4 << level);
        const cubeOffset = (3000 + level) * 4096 * 4 + superCube;
        priv.worldData[cubeOffset] = empty ? 0 : 1;
    }

    isSuperCubeEmpty(loc, level)
    {

    }

    generateSector(sector, universeLocation, lod)
    {
        //console.log("Generating sector " + sector + " around " + JSON.stringify(universeLocation) + " with LOD " + lod);
        const ptr = terrain.generateSector(universeLocation[0][0], universeLocation[1][0], universeLocation[2][0], lod);
        const sectorData = readUint32Array(ptr, terrain.memory);
        const sectorLine = SECTOR_LINES * this.mapSector(sector);
        d.get(this).worldData.set(sectorData, sectorLine * 4096 * 4);
        return {
            begin: sectorLine * 4096 * 4,
            end: sectorLine * 4096 * 4 + sectorData.length,
            data: sectorData
        };
    }

    /* Updates the horizon cube around the given universe location.
     */
    updateHorizon(universeLocation, canvas)
    {
        const priv = d.get(this);

        // flush pending uploads first
        this.uploadData(canvas, true);

        // make a deep copy
        const sectorMap = priv.sectorMap.map(entry =>
        {
            return { address: entry.address, uloc: mat.vec(...entry.uloc.flat()) };
        });

        console.log("Updating horizon around: " + JSON.stringify(universeLocation));
        const halfSize = Math.floor(HORIZON_SIZE / 2);
        const requiredSectors = [];
        const freedAddressesPerLod = [[], [], [], [], []];

        for (let y = 0; y < HORIZON_SIZE; ++y)
        {
            for (let z = 0; z < HORIZON_SIZE; ++z)
            {
                for (let x = 0; x < HORIZON_SIZE; ++x)
                {
                    const sector = makeSectorIndex(x, y, z);
                    const lod = lodOfSector(sector);
                    const loc = mat.add(
                        universeLocation,
                        mat.vec(x - halfSize, y - halfSize, z - halfSize)
                    );
                    requiredSectors.push({ sector, loc, lod });
                }
            }
        }

        // collect the addresses that became free
        for (let i = 0; i < priv.sectorMap.length; ++i)
        {
            const lod = lodOfSector(i);
            const uloc = priv.sectorMap[i].uloc;
            const idx = requiredSectors.findIndex(s => "" + s.loc === "" + uloc && s.lod === lod);
            if (idx === -1)
            {
                // this address is free
                freedAddressesPerLod[lod].push(priv.sectorMap[i].address);
            }
        }

        //console.log(JSON.stringify(freedAddressesPerLod));
        //console.log(freedAddressesPerLod[0].length);
        //console.log(JSON.stringify(priv.sectorMap));

        // either move or create the sectors
        let dataRanges = [];
        for (let i = 0; i < requiredSectors.length; ++i)
        {
            const entry = requiredSectors[i];

            const idx = sectorMap.findIndex((s, idx) => idx !== entry.sector && "" + s.uloc === "" + entry.loc && lodOfSector(idx) === entry.lod);
            if (idx === -1)
            {
                // this is a new entry
                //console.log("New Entry, sector: " + entry.sector + ", uloc: " + entry.loc);
                priv.sectorMap[entry.sector].uloc = entry.loc;
                priv.sectorMap[entry.sector].address = freedAddressesPerLod[entry.lod].shift() + 100000 /* mark as empty until uploaded */;
                //console.log("use free address: " + sectorMap[entry.sector].address);
                priv.updateQueue.push(entry);
            }
            else
            {
                // move entry
                //console.log("Move Entry, sector: " + idx + " -> " + entry.sector);
                priv.sectorMap[entry.sector].uloc = sectorMap[idx].uloc;
                priv.sectorMap[entry.sector].address = sectorMap[idx].address;
            }
        }
               
        //console.log("AFTER: " + JSON.stringify(freedAddressesPerLod));
        //console.log(JSON.stringify(priv.sectorMap.map((m, idx) => [idx, m])));
        
        this.uploadData(canvas, false);

        // write sector map
        //this.writeSectorMap();
        //canvas.updateSampler("worldData", 0, 4095, 4096, 1, priv.worldData.subarray(4095 * 4096 * 4));
    }

    uploadData(canvas, flush)
    {
        const priv = d.get(this);

        if (priv.updateQueue.length === 0)
        {
            return;
        }

        const now = Date.now();
        let duration = 0;
        let count = 0;
        while (priv.updateQueue.length > 0)
        {
            const entry = priv.updateQueue.shift();
            priv.sectorMap[entry.sector].address -= 100000;
            const sectorData = this.generateSector(entry.sector, entry.loc, entry.lod);
           
            uploadLinearData(canvas, sectorData.begin, sectorData.data);

            duration += Date.now() - now;
            ++count;

            if (! flush && duration > 10)
            {
                break;
            }
        }

        this.writeSectorMap();
        uploadLinearData(canvas, 4095 * 4096 * 4, priv.worldData.subarray(4095 * 4096 * 4));
        console.log("Uploaded " + count + " sectors in " + (Date.now() - now) + "ms");
    }

    writeSectorMap()
    {
        const priv = d.get(this);
        writeArrayAt(priv.worldData, 4096 * 4 * 4095, priv.sectorMap.map(s => s.address));
    }
};
exports.World = World;