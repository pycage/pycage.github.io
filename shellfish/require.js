/*******************************************************************************
This file is part of the Shellfish UI toolkit.
Copyright (c) 2017 - 2020 Martin Grimme <martin.grimme@gmail.com>

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

function shDocLog(message)
{
    if (document.body)
    {
        const node = document.createElement("pre");
        node.appendChild(document.createTextNode(message));
        document.body.appendChild(node);
    }
    console.log(message);
}

/**
 * Loads the given modules asynchronously and invokes a callback afterwards.
 * 
 * This function takes a number of Shellfish modules and loads them in the
 * background. When all modules have been loaded, a callback function is invoked
 * with all module objects.
 * 
 * #### Example:
 * 
 *     shRequire(["a.js", "b.js", "c.js"], (a, b, c) =>
 *     {
 *         // do something with the modules
 *         a.foo();
 *     });
 * 
 * Modules cannot pollute the global namespace and the modules are only visible
 * inside the callback function.
 * 
 * A module file looks like a normal JavaScript file, but everything that shall
 * accessible from outside is assigned to the `exports` object.
 * 
 * #### Example:
 * 
 *     exports.foo = function ()
 *     {
 *         console.log("foo");
 *     };
 * 
 * Modules in turn may load other modules with `shRequire`.
 * 
 * The special member `exports.__id` may be used to register a module ID for
 * later reference.
 * 
 *     exports.__id = "mymodule";
 * 
 * For instance, it is easier to write
 * 
 *     shRequire(["shellfish/mid"], (mid) =>
 *     {
 * 
 *     });
 * 
 * than it is to write something like
 * 
 *     shRequire(["../../shellfish/core/mid.js"], (mid) =>
 *     {
 * 
 *     });
 * 
 * The special constant `__dirname` is available inside a module and points to
 * the path of the directory where the module is located. This can,
 * for instance, be used to load modules relative to the current module.
 * 
 *     shRequire([__dirname + "/otherModule.js"], (other) =>
 *     {
 * 
 *     });
 * 
 * If a transformer function is specified, the module URL and the module data is
 * passed to this function and the returned code will be executed.
 * This mechanism is used by the Feng Shui processor to directly load Shui files
 * as modules.
 * 
 * @function
 * @param {string[]} modules - The modules to load.
 * @param {function} callback - The callback to invoke after loading.
 * @param {function} [transformer = null] - An optional function for transforming module data.
 */
const shRequire = (function ()
{
    // stack of task queues
    let stackOfQueues = [];

    // modules cache
    let cache = { };

    // bundle cache
    let bundleCache = { };

    // ID to URL map
    let idsMap = { };

    let nextScheduled = false;

    /**
     * Normalizes the given URL by resolving "." and ".." and "//".
     * @private
     * 
     * @param {string} url - The URL to normalize.
     * @returns {string} The normalized URL.
     */
    function normalizeUrl(url)
    {
        const parts = url.split("/");
        let outParts = [];
        
        parts.forEach(function (p)
        {
            if (p === ".." && outParts.length > 0 && outParts[outParts.length - 1] !== "..")
            {
                outParts.pop();
            }
            else if (p === "." || p === "")
            {
                // ignore "." and ""
            }
            else
            {
                outParts.push(p);
            }
        });

        //console.log("normalized URL " + url + " -> " + outParts.join("/"));
        return outParts.join("/");
    }

    function loadBundle(url, callback)
    {
        const now = new Date().getTime();
        const xhr = new XMLHttpRequest();
        xhr.open("GET", url, true);
        xhr.onreadystatechange = function ()
        {
            if (xhr.readyState === XMLHttpRequest.DONE)
            {
                if (xhr.status === 200)
                {
                    try
                    {
                        console.log("Loaded JS bundle '" + url + "'from server in " + (new Date().getTime() - now) + " ms.");
                        let bundle = JSON.parse(xhr.responseText);
                        for (let moduleUrl in bundle)
                        {
                            bundleCache[moduleUrl] = bundle[moduleUrl];
                        }
                        callback();
                    }
                    catch (err)
                    {
                        console.error("Failed to load JS bundle '" + url + "' from server: " + err);
                    }
                }
                else
                {
                    callback();
                }
            }
        };
        xhr.send();
    }

    function loadCode(url, callback)
    {
        if (bundleCache[url])
        {
            callback(bundleCache[url]);
            //console.log("Loaded module '" + url + "' from bundle.");
            return;
        }

        const now = new Date().getTime();
        const xhr = new XMLHttpRequest();
        xhr.open("GET", url, true);
        xhr.onreadystatechange = function ()
        {
            if (xhr.readyState === XMLHttpRequest.DONE)
            {
                if (xhr.status === 200)
                {
                    console.log("Loaded module '" + url + "' from server in " + (new Date().getTime() - now) + " ms.");
                    callback(xhr.responseText);
                }
                else
                {
                    //throw "Failed to load module: status code " + xhr.status;
                    console.error("Failed to load module '" + url + "' from server, status code = " + xhr.status + ".");
                    callback("");
                }
            }
        };
        xhr.send();
    }

    function loadStyle(url, callback)
    {
        if (bundleCache[url])
        {
            url = "data:text/css;base64," + btoa(bundleCache[url]);
        }

        let link = document.createElement("link");
        link.setAttribute("type", "text/css");
        link.setAttribute("rel", "stylesheet");
        link.setAttribute("href", url);
        link.onload = function ()
        {
            callback();
        }
        document.head.appendChild(link);
    }

    function loadModule(url, processor, callback)
    {
        if (idsMap[url])
        {
            url = idsMap[url];
        }

        if (cache[url])
        {
            //console.log("loading module from cache " + url);
            callback(cache[url]);
            return;
        }
        
        loadCode(url, function (code)
        {
            if (code === "")
            {
                callback(null);
                return;
            }

            if (processor)
            {
                code = processor(url, code);
            }

            const pos = url.lastIndexOf("/");
            let dirname = url.substr(0, pos);
            if (dirname === "")
            {
                dirname = ".";
            }

            const js = `
                (function ()
                {
                    const __dirname = "${dirname}";
                    const __filename = "${url}";
                    const exports = {
                        include: (mod) => {
                            for (let key in mod)
                            {
                                if (key !== "include" && key !== "__id")
                                {
                                    exports[key] = mod[key];
                                }
                            }
                        }
                    };
                    ${code}
                    return exports;
                })();
            `;

            try
            {
                stackOfQueues.push([]);
                const jsmodule = eval(js);
                cache[url] = jsmodule;

                if (jsmodule.__id)
                {
                    console.log(`Registered module '${url}' as '${jsmodule.__id}'.`);
                    idsMap[jsmodule.__id] = url;
                }

                callback(jsmodule);
            }
            catch (err)
            {
                console.error(`Failed to initialize module '${url}': ${err}`);
                //console.error(js);
                shDocLog(`Failed to initialize module '${url}': ${err}`);
                callback(null);
            }
        });
    }

    function next()
    {
        if (nextScheduled)
        {
            return;
        }

        if (stackOfQueues.length === 0)
        {
            return;
        }

        let queue = stackOfQueues[stackOfQueues.length - 1];
        if (queue.length === 0)
        {
            stackOfQueues.pop();
            next();
            return;
        }

        let ctx = queue[0];
        if (ctx.urls.length === 0)
        {
            queue.shift();
            ctx.callback.apply(null, ctx.modules);
            next();
            return;
        }

        let url = normalizeUrl(ctx.urls[0]);
        ctx.urls.shift();

        nextScheduled = true;

        if (url.toLowerCase().endsWith(".css"))
        {
            loadStyle(url, function ()
            {
                nextScheduled = false;
                ctx.modules.push(null);
                next();
            });
        }
        else
        {
            loadModule(url, ctx.processor, function (module)
            {
                nextScheduled = false;
                ctx.modules.push(module);
                next();
            });
        }
    }

    function addTask(urls, callback, processor)
    {
        if (stackOfQueues.length === 0)
        {
            stackOfQueues.push([]);
        }
        let queue = stackOfQueues[stackOfQueues.length - 1];
        let ctx = {
            urls: urls,
            modules: [],
            callback: callback,
            processor: processor
        };

        queue.push(ctx);
    }

    const require = function (urls, callback, processor)
    {
        if (typeof urls === "string")
        {
            addTask([urls], callback, processor);
        }
        else
        {
            addTask(urls, callback, processor);
        }
        next();
    };


    const scripts = document.getElementsByTagName("script");
    for (let i = 0; i < scripts.length; ++i)
    {
        let script = scripts[i];

        // load bundles
        const bundlesString = script.getAttribute("data-bundle");
        if (bundlesString && bundlesString !== "")
        {
            const bundles = bundlesString.split(",");
            let count = bundles.length;
            bundles.forEach((bundle) =>
            {
                nextScheduled = true;
                loadBundle(bundle.trim(), () =>
                {
                    --count;
                    if (count === 0)
                    {
                        nextScheduled = false;
                        next();
                    }
                });
            });
        }

        // load main entry point
        const main = script.getAttribute("data-main");
        if (main && main !== "")
        {
            require([main], function (module) { });
        }
    }

    return require;
})();
