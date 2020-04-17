/*******************************************************************************
This file is part of the Shellfish UI toolkit examples.
Copyright (c) 2020 Martin Grimme <martin.grimme@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*******************************************************************************/

"use strict";

exports.parseIconMap = function parseIconMap(data)
{
    function expect(data, pos, s)
    {
        if (data.substr(pos[0], s.length) === s)
        {
            pos[0] += s.length;
        }
        else
        {
            throw "Parse Error at " + pos[0] + ", " + s + " expected: " + data.substr(pos[0], 16);
        }
    }

    function skipWhiteSpace(data, pos)
    {
        while (pos[0] < data.length && (data[pos[0]] === " " || data[pos[0]] === "\r" || data[pos[0]] === "\n"))
        {
            ++pos[0];
        }
    }

    function readName(data, pos)
    {
        const idx = data.slice(pos[0]).search(/[^a-z0-9_\-]/i);
        if (idx !== -1)
        {
            const name = data.substr(pos[0], idx);
            pos[0] += name.length;
            return name;
        }
        else
        {
            throw "Parse Error at " + pos[0];
        }
    }

    function readEntry(data, pos)
    {
        skipWhiteSpace(data, pos);
        expect(data, pos, ".sh-icon-");
        const name = readName(data, pos);
        expect(data, pos, ":before");
        skipWhiteSpace(data, pos);
        expect(data, pos, "{");
        skipWhiteSpace(data, pos);
        expect(data, pos, "content:");
        skipWhiteSpace(data, pos);
        expect(data, pos, "\"\\");
        const code = readName(data, pos);
        expect(data, pos, "\";");
        skipWhiteSpace(data, pos);
        expect(data, pos, "}");
        skipWhiteSpace(data, pos);

        return {
            name: name,
            code: code
        };
    }


    const map = {};
    const pos = [0];
    while (pos[0] < data.length)
    {
        const entry = readEntry(data, pos);
        map[entry.name] = entry.code;
    }
    return map;
};
