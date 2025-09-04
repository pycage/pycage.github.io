const REG_VOID = 0;
const REG_PTR_VOID = 1;
const REG_PC = 2;
const REG_PTR_PC = 3;
const REG_SP = 4;
const REG_PTR_SP = 5;
const REG_PARAM1 = 6;
const REG_PARAM2 = 7;
const REG_PTR_PARAM1 = 8;
const REG_PTR_PARAM2 = 9;
const REG_COLOR_R = 10;
const REG_COLOR_G = 11;
const REG_COLOR_B = 12;
const REG_NORMAL_X = 13;
const REG_NORMAL_Y = 14;
const REG_NORMAL_Z = 15;
const REG_ATTRIB_1 = 16;
const REG_ATTRIB_2 = 17;
const REG_ATTRIB_3 = 18;
const REG_PTR_COLOR = 19;
const REG_PTR_NORMAL = 20;
const REG_PTR_ATTRIBUTES = 21;
const REG_PTR_ATTRIB_2 = 22;
const REG_PTR_ATTRIB_3 = 23;
const REG_END_VALUE = 24;
const REG_PTR_END_VALUE = 25;
const REG_ENV_TIMEMS = 26;
const REG_ENV_ST_X = 27;
const REG_ENV_ST_Y = 28;
const REG_ENV_RAY_DISTANCE = 29;
const REG_ENV_P_X = 30;
const REG_ENV_P_Y = 31;
const REG_ENV_P_Z = 32;
const REG_ENV_UNIVERSE_X = 33;
const REG_ENV_UNIVERSE_Y = 34;
const REG_ENV_UNIVERSE_Z = 35;

const REG_STACK = 36;
const REG_USER = 56;

const TASM_END = 0;
const TASM_JMP = 1;
const TASM_LD = 2;

const TASM_POP = 10;
const TASM_DUP = 11;

const TASM_SREG = 20;
const TASM_LREG = 21;

const TASM_TEST_LT = 30;
const TASM_TEST_LE = 31;
const TASM_TEST_EQ = 32;
const TASM_TEST_GT = 33;
const TASM_TEST_GE = 34;

const TASM_ADD = 50;
const TASM_SUB = 51;
const TASM_MUL = 52;
const TASM_DIV = 53;
const TASM_MIN = 54;
const TASM_MAX = 55;
const TASM_EXP = 56;

const TASM_ADD2 = 60;
const TASM_SUB2 = 61;
const TASM_MUL2 = 62;
const TASM_DIV2 = 63;
const TASM_MIN2 = 64;
const TASM_MAX2 = 65;

const TASM_ADD3 = 70;
const TASM_SUB3 = 71;
const TASM_MUL3 = 72;
const TASM_DIV3 = 73;
const TASM_MIN3 = 74;
const TASM_MAX3 = 75;

const TASM_USE_COLOR = 80;
const TASM_USE_NORMAL = 81;
const TASM_USE_ATTRIBUTES = 82;

const TASM_ENV = 83;

const TASM_GEN_LINE = 100;
const TASM_GEN_CHECKERBOARD = 101;
const TASM_GEN_WHITE_NOISE = 102;
const TASM_GEN_CELLULAR_NOISE_2D = 103;
const TASM_GEN_NORMAL = 104;
const TASM_GEN_MIPMAP = 105;
const TASM_GEN_LERP = 106;

const TestMap = {
    "<": TASM_TEST_LT,
    "<=": TASM_TEST_LE,
    "=": TASM_TEST_EQ,
    ">": TASM_TEST_GT,
    ">=": TASM_TEST_GE
};

function writeMicroCode(arr, opCode,
                        regCopyN, regCopySrcPtr, regCopySrcOffset, regCopyDestPtr, regCopyDestOffset,
                        testOp,
                        binOp, binOpBatchSize,
                        genOp,
                        regAddSrcPtr, regAddAmount,
                        regCopy2N, regCopy2SrcPtr, regCopy2SrcOffset, regCopy2DestPtr, regCopy2DestOffset)
{
    let offset = (3000 + opCode) * 4096 * 4;
    
    arr[offset++] = regCopyN;
    arr[offset++] = regCopySrcPtr;
    arr[offset++] = regCopyDestPtr;
    arr[offset++] = ((regCopySrcOffset + 8) << 4) + (regCopyDestOffset + 8);

    arr[offset++] = testOp;
    ++offset;
    ++offset;
    ++offset;

    arr[offset++] = binOp;
    arr[offset++] = binOpBatchSize;
    ++offset;
    ++offset;

    arr[offset++] = genOp;
    ++offset;
    ++offset;
    ++offset;

    arr[offset++] = regAddSrcPtr;
    arr[offset++] = regAddAmount;
    ++offset;
    ++offset;

    arr[offset++] = regCopy2N;
    arr[offset++] = regCopy2SrcPtr;
    arr[offset++] = regCopy2DestPtr;
    arr[offset++] = ((regCopy2SrcOffset + 8) << 4) + (regCopy2DestOffset + 8);
}

/* Writes the TASM firmware to the given data array.
 * The firmware is used by the shader to emulate the TASM instruction set
 * with simple microcode.
 */
function writeFirmware(arr)
{
    // END: push -1, set register 0
    // regcopy PTR_REG_END_VALUE PTR_PC
    writeMicroCode(arr, TASM_END,
                   1, REG_PTR_END_VALUE, 0, REG_PTR_PC, 0,
                   0, 0, 0, 0,
                   REG_PTR_VOID, 0,
                   0, REG_PTR_VOID, 0, REG_PTR_VOID, 0);

    // JMP:
    // regcopy 1 PTR_PARAM1 PTR_PC
    writeMicroCode(arr, TASM_JMP,
                   1, REG_PTR_PARAM1, 0, REG_PTR_PC, 0,
                   0, 0, 0, 0,
                   REG_PTR_VOID, 0,
                   0, REG_PTR_VOID, 0, REG_PTR_VOID, 0);

    // LD:
    // regcopy 1 PTR_PARAM1 SP
    // regadd PTR_SP 1
    writeMicroCode(arr, TASM_LD,
                   1, REG_PTR_PARAM1, 0, REG_SP, 0,
                   0, 0, 0, 0,
                   REG_PTR_SP, 1,
                   0, REG_PTR_VOID, 0, REG_PTR_VOID, 0);

    // POP:
    // regadd PTR_SP -1
    writeMicroCode(arr, TASM_POP,
                   0, REG_PTR_VOID, 0, REG_PTR_VOID, 0,
                   0, 0, 0, 0,
                   REG_PTR_SP, -1,
                   0, REG_PTR_VOID, 0, REG_PTR_VOID, 0);

    // DUP:
    // regcopy 1 SP -1 SP
    // regadd PTR_SP 1
    writeMicroCode(arr, TASM_DUP,
                   1, REG_SP, -1, REG_SP, 0,
                   0, 0, 0, 0,
                   REG_PTR_SP, 1,
                   0, REG_PTR_VOID, 0, REG_PTR_VOID, 0);

    // SREG:
    // regcopy 1 SP -1 PARAM1
    // regadd PTR_SP -1
    writeMicroCode(arr, TASM_SREG,
                   1, REG_SP, -1, REG_PARAM1, 0,
                   0, 0, 0, 0,
                   REG_PTR_SP, -1,
                   0, REG_PTR_VOID, 0, REG_PTR_VOID, 0);

    // LREG:
    // regcopy 1 PARAM1 SP
    // regadd PTR_SP 1
    writeMicroCode(arr, TASM_LREG,
                   1, REG_PARAM1, 0, REG_SP, 0,
                   0, 0, 0, 0,
                   REG_PTR_SP, 1,
                   0, REG_PTR_VOID, 0, REG_PTR_VOID, 0);

    // TEST:
    // if TEST: PARAM1 -> PC
    // regadd PTR_SP -2
    for (let i = 0; i < 5; ++i)
    {
        writeMicroCode(arr, TASM_TEST_LT + i,
                       0, REG_PTR_VOID, 0, REG_PTR_VOID, 0,
                       1 + i, 0, 0, 0,
                       REG_PTR_SP, -2,
                       0, REG_PTR_VOID, 0, REG_PTR_VOID, 0);
    }

    // BINOP:
    // binop 1 -> SP - 2
    // regadd PTR_SP -1 (-2 + 1)
    for (let i = 0; i < 7; ++i)
    {
        writeMicroCode(arr, TASM_ADD + i,
                       0, REG_PTR_VOID, 0, REG_PTR_VOID, 0,
                       0, 1 + i, 1, 0,
                       REG_PTR_SP, -2 + 1,
                       0, REG_PTR_VOID, 0, REG_PTR_VOID, 0);
    }

    // BINOP2:
    // binop 1 -> SP - 4
    // regadd PTR_SP -1 (-4 + 2)
    for (let i = 0; i < 6; ++i)
    {
        writeMicroCode(arr, TASM_ADD2 + i,
                       0, REG_PTR_VOID, 0, REG_PTR_VOID, 0,
                       0, 1 + i, 2, 0,
                       REG_PTR_SP, -4 + 2,
                       0, REG_PTR_VOID, 0, REG_PTR_VOID, 0);
    }

    // BINOP3:
    // binop 3 -> SP - 6
    // regadd PTR_SP -3 (-6 + 3)
    for (let i = 0; i < 6; ++i)
    {
        writeMicroCode(arr, TASM_ADD3 + i,
                       0, REG_PTR_VOID, 0, REG_PTR_VOID, 0,
                       0, 1 + i, 3, 0,
                       REG_PTR_SP, -6 + 3,
                       0, REG_PTR_VOID, 0, REG_PTR_VOID, 0);
    }

    // GEN:
    // gen PARAM1 -> PARAM1
    // regadd PTR_SP -n
    // regcopy 1 PTR_PARAM1 SP
    // regadd PTR_SP 1
    writeMicroCode(arr, TASM_GEN_LINE,
                   0, REG_PTR_VOID, 0, REG_PTR_VOID, 0,
                   0, 0, 0, 1,
                   REG_PTR_SP, -4 + 1,
                   1, REG_PTR_PARAM1, 0, REG_SP, -1);

    writeMicroCode(arr, TASM_GEN_CHECKERBOARD,
                   0, REG_PTR_VOID, 0, REG_PTR_VOID, 0,
                   0, 0, 0, 2,
                   REG_PTR_SP, -2 + 1,
                   1, REG_PTR_PARAM1, 0, REG_SP, -1);

    writeMicroCode(arr, TASM_GEN_WHITE_NOISE,
                   0, REG_PTR_VOID, 0, REG_PTR_VOID, 0,
                   0, 0, 0, 3,
                   REG_PTR_SP, -2 + 1,
                   1, REG_PTR_PARAM1, 0, REG_SP, -1);

    writeMicroCode(arr, TASM_GEN_CELLULAR_NOISE_2D,
                   0, REG_PTR_VOID, 0, REG_PTR_VOID, 0,
                   0, 0, 0, 4,
                   REG_PTR_SP, -4 + 1,
                   1, REG_PTR_PARAM1, 0, REG_SP, -1);

    writeMicroCode(arr, TASM_GEN_NORMAL,
                   0, REG_PTR_VOID, 0, REG_PTR_VOID, 0,
                   0, 0, 0, 5,
                   REG_PTR_SP, -4 + 3,
                   0, REG_PTR_VOID, 0, REG_PTR_VOID, 0);

    writeMicroCode(arr, TASM_GEN_MIPMAP,
                   0, REG_PTR_VOID, 0, REG_PTR_VOID, 0,
                   0, 0, 0, 6,
                   REG_PTR_SP, -3 + 2,
                   0, REG_PTR_VOID, 0, REG_PTR_VOID, 0);

    writeMicroCode(arr, TASM_GEN_LERP,
                   0, REG_PTR_VOID, 0, REG_PTR_VOID, 0,
                   0, 0, 0, 7,
                   REG_PTR_SP, -3 + 1,
                   1, REG_PTR_PARAM1, 0, REG_SP, -1);


    // USE_COLOR:
    // regadd PTR_SP -3
    // regcopy 3 SP PTR_COLOR
    writeMicroCode(arr, TASM_USE_COLOR,
                   0, REG_PTR_VOID, 0, REG_PTR_VOID, 0,
                   0, 0, 0, 0,
                   REG_PTR_SP, -3,
                   3, REG_SP, 0, REG_PTR_COLOR, 0);

    // USE_NORMAL:
    // regadd PTR_SP -3
    // regcopy 3 SP PTR_NORMAL
    writeMicroCode(arr, TASM_USE_NORMAL,
                   0, REG_PTR_VOID, 0, REG_PTR_VOID, 0,
                   0, 0, 0, 0,
                   REG_PTR_SP, -3,
                   3, REG_SP, 0, REG_PTR_NORMAL, 0);

    // USE_ATTRIBUTES:
    // regadd PTR_SP -3
    // regcopy 3 SP PTR_NORMAL
    writeMicroCode(arr, TASM_USE_ATTRIBUTES,
                   0, REG_PTR_VOID, 0, REG_PTR_VOID, 0,
                   0, 0, 0, 0,
                   REG_PTR_SP, -3,
                   3, REG_SP, 0, REG_PTR_ATTRIBUTES, 0);
}
exports.writeFirmware = writeFirmware;

/**
 * Tasmin compiles texture assembly (tasm) to byte code for processing in the shader.
 * 
 * @param {string} tasm - The tasm code
 * @returns {number[]} The compiled byte code.
 */
function compile(tasm)
{
    const byteCode = [];

    // name -> program counter
    const labelsMap = new Map();
    
    // program counter where to replace -> name
    const jumpLabels = new Map();

    // name -> register
    const registerMap = new Map();
    
    let programCounter = 0;
    const lines = tasm.split("\n");
    lines.push("END");
    const extraLines = [];

    while (lines.length > 0 || extraLines.length > 0)
    {
        let line = extraLines.length > 0 ? extraLines.shift()
                                         : lines.shift();
        line = line.trim();

        // strip-off comment
        const commentIdx = line.indexOf(";");
        if (commentIdx != -1)
        {
            //console.log("is a comment: " + line.substring(commentIdx));
            line = line.substring(0, commentIdx);
        }

        // skip empty instructions
        if (line === "")
        {
            continue;
        }

        // register label
        if (line.endsWith(":"))
        {
            const label = line.substring(0, line.length - 1);
            labelsMap.set(label, programCounter);
            //console.log("registered label: " + label + " @ " + programCounter);
            continue;
        }

        //console.log("instruction: " + line + " @ " + programCounter);

        // commas are syntactic sugar
        line = line.replace(/,/g, " ");

        const parts = line.split(" ").filter(p => p.trim() != "");
        const opCode = parts[0].toUpperCase();        

        let instructionSize = 1;
        let value1 = 0;
        let value2 = 0;

        switch (opCode)
        {
        case "END":
            byteCode.push(TASM_END);
            break;
        case "JMP":
            byteCode.push(TASM_JMP);
            jumpLabels.set(programCounter, parts[1]);
            break;
        case "LD":
            byteCode.push(TASM_LD);
            value1 = parseFloat(parts[1]);
            // extra values can be loaded on the same line as syntactic sugar
            for (let n = 2; n < parts.length; ++n)
            {
                extraLines.push("LD " + parts[n]);
            }
            break;

        case "POP":
            byteCode.push(TASM_POP);
            break;
        case "DUP":
            byteCode.push(TASM_DUP);
            break;

        case "SREG":
        {
            const regName = parts[1];
            if (! registerMap.has(regName))
            {
                registerMap.set(regName, REG_USER + registerMap.size);
            }
            byteCode.push(TASM_SREG);
            value1 = registerMap.get(regName);

            // extra values can be loaded on the same line as syntactic sugar
            for (let n = 2; n < parts.length; ++n)
            {
                extraLines.push("SREG " + parts[n]);
            }
            break;
        }
        case "LREG":
        {
            const regName = parts[1];
            if (! registerMap.has(regName))
            {
                registerMap.set(regName, REG_USER + registerMap.size);
            }
            byteCode.push(TASM_LREG);
            value1 = registerMap.get(regName);

            // extra values can be loaded on the same line as syntactic sugar
            for (let n = 2; n < parts.length; ++n)
            {
                extraLines.push("LREG " + parts[n]);
            }
            break;
        }

        case "ENV":
        {
            const name = parts[1].toUpperCase();
            if (name === "TIMEMS")
            {
                byteCode.push(TASM_LREG);
                value1 = REG_ENV_TIMEMS;
            }
            else if (name === "ST")
            {
                byteCode.push(TASM_LREG);
                value1 = REG_ENV_ST_X;
                extraLines.push("ENV st_y");
            }
            else if (name === "ST.X" || name === "ST_X")
            {
                byteCode.push(TASM_LREG);
                value1 = REG_ENV_ST_X;
            }
            else if (name === "ST.Y" || name === "ST_Y")
            {
                byteCode.push(TASM_LREG);
                value1 = REG_ENV_ST_Y;
            }
            else if (name === "DISTANCE")
            {
                byteCode.push(TASM_LREG);
                value1 = REG_ENV_RAY_DISTANCE;
            }
            else if (name === "P.X" || name === "P_X")
            {
                byteCode.push(TASM_LREG);
                value1 = REG_ENV_P_X;
            }
            else if (name === "P.Y" || name === "P_Y")
            {
                byteCode.push(TASM_LREG);
                value1 = REG_ENV_P_Y;
            }
            else if (name === "P.Z" || name === "P_Z")
            {
                byteCode.push(TASM_LREG);
                value1 = REG_ENV_P_Z;
            }
            else if (name === "UNIVERSE.X" || name === "UNIVERSE_X")
            {
                byteCode.push(TASM_LREG);
                value1 = REG_ENV_UNIVERSE_X;
            }
            else if (name === "UNIVERSE.Y" || name === "UNIVERSE_Y")
            {
                byteCode.push(TASM_LREG);
                value1 = REG_ENV_UNIVERSE_Y;
            }
            else if (name === "UNIVERSE.Z" || name === "UNIVERSE_Z")
            {
                byteCode.push(TASM_LREG);
                value1 = REG_ENV_UNIVERSE_Z;
            }

            // extra values can be loaded on the same line as syntactic sugar
            for (let n = 2; n < parts.length; ++n)
            {
                extraLines.push("ENV " + parts[n]);
            }
            break;
        }

        case "IFNOT":
        case "TEST":
        {
            byteCode.push(TestMap[parts[1]]);
            jumpLabels.set(programCounter, parts[2]);
            break;
        }

        case "ADD":
            byteCode.push(TASM_ADD);
            break;
        case "SUB":
            byteCode.push(TASM_SUB);
            break;
        case "MUL":
            byteCode.push(TASM_MUL);
            break;
        case "DIV":
            byteCode.push(TASM_DIV);
            break;
        case "MIN":
            byteCode.push(TASM_MIN);
            break;
        case "MAX":
            byteCode.push(TASM_MAX);
            break;
        case "EXP":
            byteCode.push(TASM_EXP);
            break;

        case "ADD2":
            byteCode.push(TASM_ADD2);
            break;
        case "SUB2":
            byteCode.push(TASM_SUB2);
            break;
        case "MUL2":
            byteCode.push(TASM_MUL2);
            break;
        case "DIV2":
            byteCode.push(TASM_DIV2);
            break;
        case "MIN2":
            byteCode.push(TASM_MIN2);
            break;
        case "MAX2":
            byteCode.push(TASM_MAX2);
            break;

        case "ADD3":
            byteCode.push(TASM_ADD3);
            break;
        case "SUB3":
            byteCode.push(TASM_SUB3);
            break;
        case "MUL3":
            byteCode.push(TASM_MUL3);
            break;
        case "DIV3":
            byteCode.push(TASM_DIV3);
            break;
        case "MIN3":
            byteCode.push(TASM_MIN3);
            break;
        case "MAX3":
            byteCode.push(TASM_MAX3);
            break;

        case "USE":
        {
            const name = parts[1].toUpperCase();
            if (name === "COLOR")
            {
                byteCode.push(TASM_USE_COLOR);
            }
            else if (name === "NORMAL")
            {
                byteCode.push(TASM_USE_NORMAL);
            }
            else if (name === "ATTRIBUTES")
            {
                byteCode.push(TASM_USE_ATTRIBUTES);
            }
            break;
        }

        case "GEN":
        {
            const name = parts[1].toUpperCase();
            if (name === "LINE")
            {
                byteCode.push(TASM_GEN_LINE);
            }
            else if (name === "CHECKERBOARD")
            {
                byteCode.push(TASM_GEN_CHECKERBOARD);
            }
            else if (name === "WHITENOISE")
            {
                byteCode.push(TASM_GEN_WHITE_NOISE);
            }
            else if (name === "CELLULARNOISE2D")
            {
                byteCode.push(TASM_GEN_CELLULAR_NOISE_2D);
            }
            else if (name === "NORMAL")
            {
                byteCode.push(TASM_GEN_NORMAL);
            }
            else if (name === "MIPMAP")
            {
                byteCode.push(TASM_GEN_MIPMAP);
            }
            else if (name === "LERP")
            {
                byteCode.push(TASM_GEN_LERP);
            }
            break;
        }

        default:
            console.error("Invalid TASM intruction: " + opCode);
        }
        
        byteCode.push(instructionSize);
        byteCode.push(value1);
        byteCode.push(value2);
        programCounter += instructionSize;
    }

    // set jump labels
    //console.log(jumpLabels);
    //console.log(JSON.stringify(labelsMap));
    for (const pc of jumpLabels.keys())
    {
        const label = jumpLabels.get(pc);
        //console.log("set label: '" + label + "'");
        //console.log("which is at " + labelsMap.get(label));
        byteCode[pc * 4 + 2] = labelsMap.get(jumpLabels.get(pc));
    }

    if (registerMap.size > 16)
    {
        throw "Maximum number of TASM registers exceeded.";
    }

    //console.log(byteCode);
    return byteCode;
}
exports.compile = compile;