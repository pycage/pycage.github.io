function fract(v)
{
  return Math.floor(v * 10) / 10;
}

function random(v)
{
  return fract(Math.sin(12.9898) * 43758.5453123);
}

function cellularNoise2D(px, py, size)
{
    const cubeSize = 1.0 / size;

    // in which section am I?
    const qx = Math.floor(px / cubeSize);
    const qy = Math.floor(px / cubeSize);

    // check the surroundings
    let minDist = 9999.0;
    for (let x = -1; x < 2; ++x)
    {
        for (let y = -1; y < 2; ++y)
        {
            const sampleCubeX = qx + x;
            const sampleCubeY = qy + y;

            const moduloPointX = (sampleCubeX + size) / size;
            const moduloPointY = (sampleCubeY + size) / size;

            const randomPointX = Math.random(moduloPointX) + Math.sin(moduloPointX) * 0.1;
            const randomPointY = Math.random(moduloPointY) + Math.sin(moduloPointY) * 0.2;

            const samplePointX = (sampleCubeX + randomPointX) * cubeSize;
            const samplePointY = (sampleCubeY + randomPointY) * cubeSize;

            const dist = Math.sqrt((samplePointX - px) * (samplePointX - px) + (samplePointY - py) * (samplePointY - py));
            minDist = Math.min(minDist, dist);
        }
    }
    return minDist / cubeSize;
}

function blocks()
{
    const data = `
        ################
        #XXXXXXXXXXXXXX#
        #XXXXXXXXXXXXXX#
        ################
        #X#X#X#X#X#X#X##
        ################
        ########XXXXXXX#
        ########XXXXXXX#
        ########XXXXXXX#
        ########XXXXXXX#
        ########XXXXXXX#
        ########XXXXXXX#
        #XXXXXXXXXXXXXX#
        ################
        ################
        ################

        ################
        #             ##
        #              #
        #              #
        #            ###
        #            #  
        #  !!!       #  
        #  !!!       #  
        #  !!!       #  
        #            XX#
        #              #
        ####           #
        #              #
        #              #
        #              #
        ################

        #       ########
                      O#
                       #
                       #
                     ###
                     #  
                V    #  
                     #  
                     XX#
                       #
                       #
        #  G           #
        #              #
        # O            #
        #              #
        #!!!!!!!!!!!!!!#

        #          ###  
                        
                       #
                       #
                     ###
                     #  
                     #  
                     XX#
                       #
                       #
                       #
        #              #
        #              #
        #              #
        #              #
        #!!!!!!!!!!!!!!#

        #      #########
               #########
               #########
               #########
             ###########
             ###########
             ##      XX#
        #######        #
        #######        #
        #######        #
        ########       #
        ##########     #
        ################
        ################
        ################
        ################

        ################
        #            ###
        #            ###
        #             ##
        #              #
        #              #
        #              #
        #              #
        #              #
        #              #
        #              #
        #              #
        #              #
        #  O           #
        #              #
        ################

    `;

    let out = "";
    const lines = data.split("\n");
    lines.shift();
    for (let layer = 0; layer < 6; ++layer)
    {
        for (let row = 0; row < 16; ++row)
        {
            const line = lines.shift();
            out += line.substring(8);
        }
        lines.shift();
    }
    return { size: 16, data: out };
}
exports.blocks = blocks;

function blocks2()
{
  let data = "";

  for (let z = 0; z < 40; ++z)
  {
    for (let x = 0; x < 250; ++x)
    {
      for (let y = 0; y < 250; ++y)
      {
        //const value = cellularNoise2D(x / 250, y / 250, 3);
        const value = Math.abs(Math.sin((x * y) / 10000));
        //console.log("x " + x + " y " + y + " => " + value);

        if (z > 4 && z < 9 && y > 100 & y < 115)
        {
          data += " ";
        }
        else if (x == 0 || x == 249 || y == 0 || y == 249)
        {
          data += "X";
        }
        else if (z < value * 20)
        {
          data += "#";
        }
        else if (z < 5)
        {
          data += "!";
        }
        else
        {
          data += " ";
        }
      }
    }
  }

  return { size: 250, data: data }; 
}
exports.blocks2 = blocks2;