const core = await shRequire("shellfish/core");
const mat = await shRequire("shellfish/core/matrix");
const sdf = await shRequire("./sdf.js");
const terrain = await shRequire("./wasm/terrain.wasm");

// the side-length of the horizon cube in sectors (must be odd so there is a center)
const HORIZON_SIZE = 15;

const DISTANCE_LODS = [0, 0, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5];
// the data stride of a sector
const LOD_SECTOR_STRIDE = [69632 * 4, 12288 * 4, 5120 * 4, 640 * 4, 80 * 4, 10 * 4];
// the side-length of a cube in voxels
const LOD_CUBE_SIZE =   [ 4,  2,  1, 1, 1, 1, 1];
// the side-length of a sector in cubes
const LOD_SECTOR_SIZE = [16, 16, 16, 8, 4, 2, 1];
// the division factor from nominal voxels to LOD voxels
const LOD_CUBE_DIV = [1, 2, 4, 4, 4, 4, 4];
// the division factor from nominal cubes to LOD cubes
const LOD_SECTOR_DIV = [1, 1, 1, 2, 4, 8, 16];

const INVALID_SECTOR_ADDRESS = 1;

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

function locEqual(a, b)
{
    return a[0][0] === b[0][0] &&
           a[1][0] === b[1][0] &&
           a[2][0] === b[2][0];
}

function locHash(loc)
{
  return "" + loc.flat();
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

/* Returns the level-of-detail for the given sector number.
 */
function lodOfSector(sector)
{
    const [x, y, z] = sectorLocation(sector).flat();
    const center = Math.floor(HORIZON_SIZE / 2);
    const dist = Math.max(Math.abs(x - center), Math.abs(y - center), Math.abs(z - center));

    return DISTANCE_LODS[dist];
}

function sectorWorldLocation(sector, cubeSize)
{
    const sectorLength = LOD_SECTOR_SIZE[0] * cubeSize;
    return mat.mul(sectorLocation(sector), sectorLength);
}

function makeCubeLocator(loc, cubeSize)
{
    const sectorLength = LOD_SECTOR_SIZE[0] * cubeSize;

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

        if (line1 + 1 <= line2 - 1)
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
            sectorMap: [],  // array of { address, universeLocation }
            centerSector: makeSectorIndex(horizonCenter, horizonCenter, horizonCenter),
            freedAddressesPerLod: [[], [], [], [], [], []]
        });

        this.initializeSectorMap();
   }

    get worldData() { return d.get(this).worldData; }
    get sectorMap() { return d.get(this).sectorMap.map(s => s.address); }
    get centerSector() { return d.get(this).centerSector; }

    /* Releases a sector address for a LOD.
     */
    releaseSectorAddress(lod, address)
    {
        d.get(this).freedAddressesPerLod[lod].push(address);
    }

    /* Allocates a free sector address for a LOD.
     */
    allocateSectorAddress(lod)
    {
        if (d.get(this).freedAddressesPerLod[lod].length === 0)
        {
            console.error("Out of sector memory for LOD " + lod + ".");
            throw "Out Of Sector Memory";
        }
        return d.get(this).freedAddressesPerLod[lod].pop();
    }

    /* Initializes the sector map.
     */
    initializeSectorMap()
    {
        const priv = d.get(this);

        // count the sectors per LOD
        let lodSectorCounts = [0, 0, 0, 0, 0, 0];
        for (let i = 0; i < HORIZON_SIZE * HORIZON_SIZE * HORIZON_SIZE; ++i)
        {
            const lod = lodOfSector(i);
            ++lodSectorCounts[lod];
        }
        // add some sectors for buffer
        lodSectorCounts = lodSectorCounts.map(c => Math.ceil(c * 2.0));

        console.log("LOD sector counts: " + JSON.stringify(lodSectorCounts));

        // compute the LOD address offsets
        const lodSlotSizes = [
            lodSectorCounts[0] * LOD_SECTOR_STRIDE[0],
            lodSectorCounts[1] * LOD_SECTOR_STRIDE[1],
            lodSectorCounts[2] * LOD_SECTOR_STRIDE[2],
            lodSectorCounts[3] * LOD_SECTOR_STRIDE[3],
            lodSectorCounts[4] * LOD_SECTOR_STRIDE[4],
            lodSectorCounts[5] * LOD_SECTOR_STRIDE[5]
        ];
        const lodSlotOffsets = [0];
        let offset = 0;
        for (let i = 0; i < lodSlotSizes.length; ++i)
        {
            offset += lodSlotSizes[i];
            lodSlotOffsets.push(offset);
        }
        console.log("LOD slot offsets: " + JSON.stringify(lodSlotOffsets));

        // create the slots
        for (let lod = 0; lod < lodSectorCounts.length; ++lod)
        {
            const lodCount = lodSectorCounts[lod];
            for (let i = 0; i < lodCount; ++i)
            {
                const physicalAddress = lodSlotOffsets[lod] + i * LOD_SECTOR_STRIDE[lod];
                priv.freedAddressesPerLod[lod].push(physicalAddress);
            }
        }

        for (let i = 0; i < HORIZON_SIZE * HORIZON_SIZE * HORIZON_SIZE; ++i)
        {
            const lod = lodOfSector(i);
            const physicalAddress = this.allocateSectorAddress(lod);
            priv.sectorMap.push({ address: physicalAddress, uloc: mat.vec(0, 0, 0), lod: lod });
        }

        this.writeSectorMap();
    }

    /* Maps a sector to its actual physical address.
     */
    mapSector(sector)
    {
        return d.get(this).sectorMap[sector].address;
    }

    /* Returns the sector data offset.
     */
    sectorDataOffset(sector)
    {
        return this.mapSector(sector);
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
    voxelDataOffset(address, lod)
    {
        const sectorSize = LOD_SECTOR_SIZE[lod];
        const cubeSize = LOD_CUBE_SIZE[lod];

        return sectorSize * sectorSize * sectorSize * 4 +
               address * cubeSize * cubeSize * cubeSize;
    }

    /* Returns the sector at the given world location.
     */
    sectorAt(loc)
    {
        const sectorLength = LOD_SECTOR_SIZE[0] * LOD_CUBE_SIZE[0];

        const v = mat.mul(loc, 1 / sectorLength);
        const x = Math.floor(v[0][0]);
        const y = Math.floor(v[1][0]);
        const z = Math.floor(v[2][0]);

        return makeSectorIndex(x, y, z);
    }

    /* Returns the world location of the given sector.
     */
    sectorWorldLocation(sector)
    {
        return mat.mul(sectorLocation(sector), LOD_SECTOR_SIZE[0] * LOD_CUBE_SIZE[0]);
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
        return makeCubeLocator(loc, LOD_CUBE_SIZE[0]);
    }

    /* Returns the world location of the given cube.
     */
    cubeLocation(cube)
    {
        return resolveCubeLocator(cube, LOD_CUBE_SIZE[0]);
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
        // FIXME: unused
        const nominalCubeSize = LOD_CUBE_SIZE[0];

        const cubes = [];

        let p = origin;
        const originCube = this.cubeOf(p);
        cubes.push(originCube);

        //console.log("ray: " + rayDirection.flat().map(c => c.toFixed(2)));

        const scaleX = rayDirection[0][0] != 0.0 ? nominalCubeSize / Math.abs(rayDirection[0][0]) : 9999999.0;
        const scaleY = rayDirection[1][0] != 0.0 ? nominalCubeSize / Math.abs(rayDirection[1][0]) : 9999999.0;
        const scaleZ = rayDirection[2][0] != 0.0 ? nominalCubeSize / Math.abs(rayDirection[2][0]) : 9999999.0;

        //console.log("scale: " + mat.vec(scaleX, scaleY, scaleZ).flat().map(c => c.toFixed(2)));

        let gridX = nominalCubeSize * Math.floor((p[0][0]) / nominalCubeSize);
        let gridY = nominalCubeSize * Math.floor((p[1][0]) / nominalCubeSize);
        let gridZ = nominalCubeSize * Math.floor((p[2][0]) / nominalCubeSize);

        if (rayDirection[0][0] > 0.0)
        {
            gridX += nominalCubeSize;
        }
        if (rayDirection[1][0] > 0.0)
        {
            gridY += nominalCubeSize;
        }
        if (rayDirection[2][0] > 0.0)
        {
            gridZ += nominalCubeSize;
        }
        //console.log("grid: " + mat.vec(gridX, gridY, gridZ).flat().map(c => c.toFixed(2)));

        let distX = Math.abs(gridX - p[0][0]) / nominalCubeSize;
        let distY = Math.abs(gridY - p[1][0]) / nominalCubeSize;
        let distZ = Math.abs(gridZ - p[2][0]) / nominalCubeSize;

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
                moveX = Math.sign(rayDirection[0][0]) * nominalCubeSize;
                distX += 1.0;
                //console.log("rayLength " + rayLengthX);
                newOrigin = mat.add(origin, mat.mul(rayDirection, rayLengthX));
            }
            else if (rayLengthY <= rayLengthX && rayLengthY <= rayLengthZ)
            {
                moveY = Math.sign(rayDirection[1][0]) * nominalCubeSize;
                distY += 1.0;
                //console.log("rayLength " + rayLengthY);
                newOrigin = mat.add(origin, mat.mul(rayDirection, rayLengthY));
            }
            else if (rayLengthZ <= rayLengthX && rayLengthZ <= rayLengthY)
            {
                moveZ = Math.sign(rayDirection[2][0]) * nominalCubeSize;
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

    /* Returns the list of voxels in the given cube.
     */
    voxelsInCube(cube)
    {
        const priv = d.get(this);

        const lod = priv.sectorMap[cube.sector].lod;
        const nominalCubeSize = LOD_CUBE_SIZE[0];
        const sectorSize = LOD_SECTOR_SIZE[lod];
        const cubeSize = LOD_CUBE_SIZE[lod];
        const bitsPerCoord = cubeSize == 4 ? 2 : 1;

        const cubeX = Math.floor(cube.x / LOD_SECTOR_DIV[lod]);
        const cubeY = Math.floor(cube.y / LOD_SECTOR_DIV[lod]);
        const cubeZ = Math.floor(cube.z / LOD_SECTOR_DIV[lod]);
        const cubeIndex = cubeX * sectorSize * sectorSize +
                          cubeY * sectorSize +
                          cubeZ;
        const sectorOffset = this.sectorDataOffset(cube.sector);
        const cubeOffset = sectorOffset + this.cubeDataOffset(cubeIndex);
        let patternHi = priv.worldData[cubeOffset];
        let patternLo = priv.worldData[cubeOffset + 1];
        const address = priv.worldData[cubeOffset + 2];

        const voxelOffset = this.voxelDataOffset(address, lod);

        let objects = [];
        for (let x = 0; x < nominalCubeSize; ++x)
        {
            for (let y = 0; y < nominalCubeSize; ++y)
            {
                for (let z = 0; z < nominalCubeSize; ++z)
                {
                    const lx = Math.floor(x / LOD_CUBE_DIV[lod]);
                    const ly = Math.floor(y / LOD_CUBE_DIV[lod]);
                    const lz = Math.floor(z / LOD_CUBE_DIV[lod]);

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

        const hits = this.voxelsInCube(cube).map(obj =>
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
        const sectorDataOffset = this.sectorDataOffset(sector);

        d.get(this).worldData.set(sectorData, sectorDataOffset);
        return {
            offset: sectorDataOffset,
            data: sectorData
        };
    }

    /* Updates the horizon cube around the given universe location.
     */
    updateHorizon(universeLocation, viewingDirection, canvas)
    {
        const priv = d.get(this);

        // flush pending uploads first
        let now = Date.now();
        this.uploadData(canvas, true);
        console.log("Flushing queue took " + (Date.now() - now) + "ms");

        // make a deep copy
        now = Date.now();
        const sectorMapIndex = new Map();
        const sectorMap = priv.sectorMap.map((entry, idx) =>
        {
            sectorMapIndex.set(locHash(entry.uloc), idx);
            return { address: entry.address, uloc: entry.uloc.slice(), lod: entry.lod };
        });
        console.log("Making deep copy took " + (Date.now() - now) + "ms");

        console.log("Updating horizon around: " + JSON.stringify(universeLocation));
        const halfSize = Math.floor(HORIZON_SIZE / 2);
        const requiredSectors = [];
        const requiredSectorsIndex = new Map();

        // find the sectors that are required
        now = Date.now();
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
                    requiredSectorsIndex.set(locHash(loc), requiredSectors.length);
                    requiredSectors.push({ sector, loc, lod });
                }
            }
        }
        console.log("Getting required sectors took " + (Date.now() - now) + "ms");

        // collect the addresses that became free
        now = Date.now();
        priv.sectorMap.forEach(entry =>
        {
            const idx = requiredSectorsIndex.get(locHash(entry.uloc));
            if (idx === undefined)
            {
                // this address is free, because it doesn't move from one
                // sector to another
                this.releaseSectorAddress(entry.lod, entry.address);
            }
        });
        console.log("Collecting free addresses took " + (Date.now() - now) + "ms");

        //console.log(JSON.stringify(priv.sectorMap));

        // either move or create the sectors
        now = Date.now();
        requiredSectors.forEach(entry =>
        {
            const idx = sectorMapIndex.get(locHash(entry.loc));
            if (idx === undefined)
            {
                // this is a new entry
                //console.log("New Entry, sector: " + entry.sector + ", uloc: " + entry.loc);
                priv.sectorMap[entry.sector].address = -1;
                priv.updateQueue.push(entry);
            }
            else
            {
                // move entry
                priv.sectorMap[entry.sector] = sectorMap[idx];

                // update LOD
                if (entry.lod !== priv.sectorMap[entry.sector].lod)
                {
                    entry.isUpdate = true;
                    priv.updateQueue.push(entry);
                }
            }
        });
        console.log("Moving/creating sectors took " + (Date.now() - now) + "ms");

        now = Date.now();
        const center = Math.floor(HORIZON_SIZE / 2);
        priv.updateQueue.sort((a, b) =>
        {
            const aDot = mat.dot(viewingDirection, a.loc);
            const bDot = mat.dot(viewingDirection, b.loc);
            if (aDot < 0.0) return 1;
            if (bDot < 0.0) return -1;
            const [x1, y1, z1] = a.loc.flat();
            const [x2, y2, z2] = b.loc.flat();
            const dist1 = Math.max(Math.abs(x1 - center), Math.abs(y1 - center), Math.abs(z1 - center));
            const dist2 = Math.max(Math.abs(x2 - center), Math.abs(y2 - center), Math.abs(z2 - center));

            return dist1 - dist2;
        });
        console.log("Sorting update queue by distance took " + (Date.now() - now) + "ms");
               
        //console.log(JSON.stringify(priv.sectorMap.map((m, idx) => [idx, m])));
        
        now = Date.now();
        this.uploadData(canvas, false);
        console.log("Initial upload took " + (Date.now() - now) + "ms");
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

            if (entry.isUpdate)
            {
                this.releaseSectorAddress(priv.sectorMap[entry.sector].lod, priv.sectorMap[entry.sector].address);
            }
            priv.sectorMap[entry.sector].uloc = entry.loc;
            priv.sectorMap[entry.sector].address = this.allocateSectorAddress(entry.lod);
            priv.sectorMap[entry.sector].lod = entry.lod;

            const sectorData = this.generateSector(entry.sector, entry.loc, entry.lod);
            //console.log("Generated sector " + entry.sector + ", LOD: " + entry.lod + ", offset: " + sectorData.offset + ", size: " + sectorData.data.length);

            uploadLinearData(canvas, sectorData.offset, sectorData.data);

            duration = Date.now() - now;
            ++count;

            if (! flush && duration > 5)
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
        const data = priv.sectorMap.map(s =>
        {
            // divide address by 4 to get pixel address
            const addr = s.address < 0 ? INVALID_SECTOR_ADDRESS : s.address >> 2;
            return (addr << 3) + s.lod;
        });
        writeArrayAt(priv.worldData, 4096 * 4 * 4095, data);
    }
};
exports.World = World;
