var KEYCODES = {
    up: 38,
    down: 40,
    left: 37,
    right: 39,
    f: 70,
    u: 85,
    d: 68,
    r: 82,
    l: 76,
    b: 66,
    shift: 16,
    two: 50
};

var AXES = [
            [0, 0, 1],
            [0, 0, -1],
            [0, 1, 0],
            [0, -1, 0],
            [1, 0, 0],
            [-1, 0, 0]
            ];

function principal_value(angle) {
    angle = ((angle + Math.PI) % (2*Math.PI) - Math.PI);
    if (feq(angle, -Math.PI))
        angle = Math.PI;
    return angle;
}

function RenderUtilities() {
    function clearAll(gl) {
        gl.clearColor(0.0, 0.0, 0.0, 1.0);  // Clear to black, fully opaque
        gl.clearDepth(1.0);                 // Clear everything
        gl.enable(gl.DEPTH_TEST);           // Enable depth testing
        gl.depthFunc(gl.LEQUAL);            // Near things obscure far things
    }

    function initShaders(data) {
        var gl = data.gl;
        var fragmentShader = getShader(gl, "shader-fs");
        var vertexShader = getShader(gl, "shader-vs");
        // Create the shader program
        data.shaderProgram = gl.createProgram();
        gl.attachShader(data.shaderProgram, vertexShader);
        gl.attachShader(data.shaderProgram, fragmentShader);
        gl.linkProgram(data.shaderProgram);
        // If creating the shader program failed, alert
        if (!gl.getProgramParameter(data.shaderProgram, gl.LINK_STATUS)) {
            alert("Unable to initialize the shader program.");
        }
        gl.useProgram(data.shaderProgram);
        data.vertexPositionAttribute = gl.getAttribLocation(data.shaderProgram,
                                                       "aVertexPosition");
        gl.enableVertexAttribArray(data.vertexPositionAttribute);
        data.textureCoordAttribute = gl.getAttribLocation(data.shaderProgram,
                                                     "aTextureCoord");
        gl.enableVertexAttribArray(data.textureCoordAttribute);
        data.vertexNormalAttribute = gl.getAttribLocation(data.shaderProgram,
                                                     "aVertexNormal");
        gl.enableVertexAttribArray(data.vertexNormalAttribute);
    }

    function getShader(gl, id) {
        var shaderScript = document.getElementById(id);
        // Didn't find an element with the specified ID; abort.
        if (!shaderScript) {
            return null;
        }
        // Walk through the source element's children, building the
        // shader source string.
        var theSource = "";
        var currentChild = shaderScript.firstChild;
        while(currentChild) {
            if (currentChild.nodeType == 3) {
                theSource += currentChild.textContent;
            }
            currentChild = currentChild.nextSibling;
        }
        // Now figure out what type of shader script we have,
        // based on its MIME type.
        var shader;
        if (shaderScript.type == "x-shader/x-fragment") {
            shader = gl.createShader(gl.FRAGMENT_SHADER);
        } else if (shaderScript.type == "x-shader/x-vertex") {
            shader = gl.createShader(gl.VERTEX_SHADER);
        } else {
            return null;  // Unknown shader type
        }
        // Send the source to the shader object
        gl.shaderSource(shader, theSource);
        // Compile the shader program
        gl.compileShader(shader);
        // See if it compiled successfully
        if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
            alert("An error occurred compiling the shaders: " +
                  gl.getShaderInfoLog(shader));
            return null;
        }
        return shader;
    };
    this.clearAll = clearAll;
    this.initShaders = initShaders;
}
var rendUtil = new RenderUtilities();

function MatrixUtilities() {
    var loadIdentity = function(data) {
        data.mvMatrix = Matrix.I(4);
    };

    var multMatrix = function(data, m) {
        data.mvMatrix = data.mvMatrix.x(m);
    };

    var mvTranslate = function(data, v) {
        multMatrix(data,
                   Matrix.Translation($V([v[0], v[1], v[2]])).ensure4x4());
    };

    var setMatrixUniforms = function(data) {
        var gl = data.gl;
        var pUniform = gl.getUniformLocation(data.shaderProgram, "uPMatrix");
        gl.uniformMatrix4fv(pUniform, false,
                            new Float32Array(data.perspectiveMatrix.flatten()));
        var mvUniform = gl.getUniformLocation(data.shaderProgram, "uMVMatrix");
        gl.uniformMatrix4fv(mvUniform, false,
                            new Float32Array(data.mvMatrix.flatten()));
        var normalMatrix = data.mvMatrix.inverse();
        normalMatrix = normalMatrix.transpose();
        var nUniform = gl.getUniformLocation(data.shaderProgram,
                                             "uNormalMatrix");
        gl.uniformMatrix4fv(nUniform, false,
                            new Float32Array(normalMatrix.flatten()));
    };

    var mvPushMatrix = function(data, m) {
        if (m) {
            data.mvMatrixStack.push(m.dup());
            data.mvMatrix = m.dup();
        } else {
            data.mvMatrixStack.push(data.mvMatrix.dup());
        }
    };
    var mvPopMatrix = function(data) {
        if (!data.mvMatrixStack.length) {
            throw("Can't pop from an empty matrix stack.");
        }
        data.mvMatrix = data.mvMatrixStack.pop();
        return data.mvMatrix;
    };
    var mvRotate = function(data, angle, v) {
        var m = Matrix.Rotation(angle, $V([v[0], v[1], v[2]])).ensure4x4();
        multMatrix(data, m);
    };

    this.loadIdentity = loadIdentity;
    this.mvTranslate = mvTranslate;
    this.mvPushMatrix = mvPushMatrix;
    this.mvPopMatrix = mvPopMatrix;
    this.mvRotate = mvRotate;
    this.setMatrixUniforms = setMatrixUniforms;
}
var matUtil = new MatrixUtilities();

function SimpleScene(canvasID) {
    'use strict';
    // `canvasID` is the id of the canvas element used for the scene.
    var data,
        renderHeap;

    data = {
        canvas: null,
        gl: null,
        rot: [0.0, 0.0, 0.0],
        mvMatrix: null,
        mvMatrixStack: [],
        shaderProgram: null,
        perspectiveMatrix: null,
        vertexPositionAttribute: null,
        textureCoordAttribute: null,
        vertexNormalAttribute: null,
        loadIdentity: function () {return matUtil.loadIdentity(data); },
        mvTranslate: function (v) {return matUtil.mvTranslate(data, v); },
        mvPushMatrix: function (m) {return matUtil.mvPushMatrix(data, m); },
        mvPopMatrix: function () {return matUtil.mvPopMatrix(data); },
        mvRotate: function (a, v) {return matUtil.mvRotate(data, a, v); },
        setMatrixUniforms: function () {
            return matUtil.setMatrixUniforms(data);
        }
    };

    renderHeap = [];

    function linkObjects(sceneObjects) {
        renderHeap.push(sceneObjects);
        sceneObjects.linkRendering(data);
    }

    function drawScene() {
        var gl,
            i,
            sceneObject;
        /*var size = getWindowSize();
        size.width *= settings.glcanvas.widthFraction;
        size.height *= settings.glcanvas.heightFraction;
        size.width = min(size.width, settings.webcam.maxWidth);
        size.height = min(size.height, settings.webcam.maxHeight);
        if (data.canva.width != size.width ||
            data.canva.height != size.height)
        {
            data.canva.width = size.width;
            data.canva.height = size.height;
        }*/

        gl = data.gl;
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
        // Establish the perspective with which we want to view the
        // scene. Our field of view is 45 degrees, with a width/height
        // ratio of 640:480, and we only want to see objects between 0.1 units
        // and 100 units away from the camera.
        data.perspectiveMatrix = makePerspective(45, 640.0 / 480.0, 0.1, 100.0);
        for (i = 0; i < renderHeap.length; i += 1) {
            sceneObject = renderHeap[i];
            sceneObject.draw();
        }
    }

    function init() {
        data.canvas = document.getElementById(canvasID);
        data.gl = data.canvas.getContext("experimental-webgl");
        rendUtil.clearAll(data.gl);
        rendUtil.initShaders(data);
        data.loadIdentity();
        data.mvPushMatrix();
        data.mvTranslate([0, 0, -15.5]);
    }

    init();

    this.draw = drawScene;
    this.linkObjects = linkObjects;
    this.data = data;
}

function transformPortion(portion, expFactor) {
    if (portion > 0.5) {
        return 0.5 + (0.5 * Math.pow(2*(portion-0.5), expFactor));
    } else {
        return 0.5 - (0.5 * Math.pow(2*(0.5-portion), expFactor));
    }
}

function Cube(sceneData, x, y, z, len, up, right, cols, set, atHome,
              xHome, yHome, zHome, lenHome, upHome, rightHome) {
    /* Cube is centered at (x, y, z)
     * Each edge has length `len`
     * "Up" indicates the normal to the face with color colors[0]
     * "Right" indicates the normal to the face with color colors[2]
     * Colors are in this order: F B U D R L
     */
    'use strict';

    if (atHome) {
        xHome = x;
        yHome = y;
        zHome = z;
        lenHome = len;
        upHome = up;
        rightHome = right;
    }

    var colors,
        scene,
        settings,
        texture,
        loc = {
            len: len,
            pos: [x, y, z],
            orientation: {up: up,
                          right: right}
        },
    /* Permanent location (updated at the end of a move) */
        ploc = {
            len: len,
            pos: [x, y, z],
            orientation: {up: copyArray(up),
                          right: copyArray(right)}
        },

    /* Buffers (used for WebGL rendering) */
        buffers = {
            vertices: null,
            normals: null,
            texture: null,
            indices: null
        },

    /* Starting position and orientation. This is where the solver
     * aims to return the Cube to.
     */
        home = {
            len: lenHome,
            pos: [xHome, yHome, zHome],
            orientation: {up: copyArray(upHome),
                          right: copyArray(rightHome)}
        },

    /* Data about the movement state. */
        moving = {
            currently: false,
            update: null,
            stop: null,
            frameStart: null,
            frameCurrent: null
        };

    /* Snap the cube into integer coordinates. Used at move completion
     * to remove rounding errors. It is important that inter-move locations
     * are on integer coordinates, because of this function!
     */


    function getState() {
        return new Cube(sceneData,
                        ploc.pos[0], ploc.pos[1], ploc.pos[2],
                        ploc.len, copyArray(ploc.orientation.up),
                        copyArray(ploc.orientation.right),
                        cols, set, false,
                        home.pos[0], home.pos[1], home.pos[2],
                        home.len, copyArray(home.orientation.up),
                        copyArray(home.orientation.right));
    }

    function snap() {
        loc.pos = vec.ints(loc.pos);
        loc.orientation = {
            up: vec.ints(loc.orientation.up),
            right: vec.ints(loc.orientation.right)
        };
        ploc.pos = copyArray(loc.pos);
        ploc.orientation = {up: copyArray(loc.orientation.up),
                            right: copyArray(loc.orientation.right)};

    }

    this.alignedWithAxis = function (axis) {
        return (afeq(vec.dot(axis, ploc.orientation.up),
                     vec.dot(axis, home.orientation.up)) &&
                afeq(vec.dot(axis, ploc.orientation.right),
                     vec.dot(axis, home.orientation.right)));
    };

    this.palignedWithAxis = function (axis) {
        return (feq(vec.dot(axis, this.ploc.orientation.up),
                    vec.dot(axis, this.home.orientation.up)) &&
                feq(vec.dot(axis, this.ploc.orientation.right),
                    vec.dot(axis, this.home.orientation.right)));
    }

    this.alignedWithAxes = function (home, current) {
        return (afeq(vec.dot(vec.unit(current),
                             vec.unit(this.ploc.orientation.up)),
                     vec.dot(vec.unit(home),
                             vec.unit(this.home.orientation.up))) &&
                afeq(vec.dot(vec.unit(current),
                             vec.unit(this.ploc.orientation.right)),
                     vec.dot(vec.unit(home),
                             vec.unit(this.home.orientation.right))));
    }

    this.isHome = function () {
        if (!(vec.isZero(this.home.pos) && vec.isZero(this.ploc.pos)) &&
            !vec.parallel(this.home.pos, this.ploc.pos))
            return false;
        if (!vec.parallel(this.home.orientation.up, this.ploc.orientation.up))
            return false;
        if (!vec.parallel(this.home.orientation.right,
                          this.ploc.orientation.right))
            return false;
        return true;
        for (var i = 0; i < 6; i += 1) {
            if (colors[i] !== -1 && !this.alignedWithAxis(AXES[i]))
                return false;
        }
        return true;
    }

    function returnHome() {
        moving.currently = false;
        loc.pos = copyArray(home.pos);
        loc.orientation.up = copyArray(home.orientation.up);
        loc.orientation.right = copyArray(home.orientation.right);
        loc.len = home.len;
        snap();
        resetBuffers();
    }

    function rotate(axis, angle, frameStart) {
        var hand = vec.without(loc.pos, axis),
            perpHand = vec.setMag(vec.mag(hand), vec.cross(axis, hand)),
            getPos = function (portion) {
                return vec.add(axis,
                               vec.add(vec.muls(Math.cos(angle * (1 - portion)),
                                                hand),
                               vec.muls(Math.sin(angle * (1 - portion)),
                                        perpHand)));
            },
            upOrig = vec.unit(loc.orientation.up),
            upPerp = vec.unit(vec.cross(axis, upOrig)),
            rightOrig = vec.unit(loc.orientation.right),
            rightPerp = vec.unit(vec.cross(axis, rightOrig)),
            getOrientation = function (portion) {
                return {up: (vec.isZero(upPerp) ? upOrig :
                             vec.add(vec.muls(Math.cos(angle * (1 - portion)),
                                              upOrig),
                                     vec.muls(Math.sin(angle * (1 - portion)),
                                              upPerp))),
                        right: (vec.isZero(rightPerp) ? rightOrig :
                                vec.add(vec.muls(Math.cos(angle *
                                                          (1 - portion)),
                                                 rightOrig),
                                        vec.muls(Math.sin(angle *
                                                          (1 - portion)),
                                                 rightPerp)))
                };
            };
        moving.update = function () {
            var portion = moving.frameCurrent / moving.frameStart;
            portion = transformPortion(portion, settings.turnAcceleration);
            loc.pos = getPos(portion);
            loc.orientation = getOrientation(portion);
            resetBuffers();
            moving.frameCurrent -= 1;
            if (moving.frameCurrent <= -1) {
                /* Stop rotating */
                moving.stop();
            }
            return (moving.frameCurrent <= -1);
        };

        moving.stop = function () {
            loc.pos = getPos(0);
            loc.orientation = getOrientation(0);
            snap();
            moving.currently = false;
            resetBuffers();
        };

        moving.currently = true;
        moving.frameStart = frameStart;
        moving.frameCurrent = frameStart;
        moving.update();
    }

    /* Used by resetBuffers */
    function getVertices() {
        var r,
            F,
            U,
            R,
            B,
            D,
            L,
            FUL,
            FUR,
            FDL,
            FDR,
            BUL,
            BUR,
            BDL,
            BDR;
        r = loc.len * 0.5;
        F = vec.unit(vec.cross(loc.orientation.right, loc.orientation.up));
        U = vec.unit(loc.orientation.up);
        R = vec.unit(loc.orientation.right);
        F = vec.muls(r, F);
        U = vec.muls(r, U);
        R = vec.muls(r, R);
        B = vec.muls(-1, F);
        D = vec.muls(-1, U);
        L = vec.muls(-1, R);
        FUL = vec.add(loc.pos, vec.add(F, vec.add(U, L)));
        FUR = vec.add(loc.pos, vec.add(F, vec.add(U, R)));
        FDL = vec.add(loc.pos, vec.add(F, vec.add(D, L)));
        FDR = vec.add(loc.pos, vec.add(F, vec.add(D, R)));
        BUL = vec.add(loc.pos, vec.add(B, vec.add(U, L)));
        BUR = vec.add(loc.pos, vec.add(B, vec.add(U, R)));
        BDL = vec.add(loc.pos, vec.add(B, vec.add(D, L)));
        BDR = vec.add(loc.pos, vec.add(B, vec.add(D, R)));

        return [
         // Front face
            FUL[0], FUL[1], FUL[2],
            FUR[0], FUR[1], FUR[2],
            FDR[0], FDR[1], FDR[2],
            FDL[0], FDL[1], FDL[2],

         // Back face
            BUL[0], BUL[1], BUL[2],
            BUR[0], BUR[1], BUR[2],
            BDR[0], BDR[1], BDR[2],
            BDL[0], BDL[1], BDL[2],

         // Top face
            FUL[0], FUL[1], FUL[2],
            BUL[0], BUL[1], BUL[2],
            BUR[0], BUR[1], BUR[2],
            FUR[0], FUR[1], FUR[2],

         // Bottom face
            FDL[0], FDL[1], FDL[2],
            BDL[0], BDL[1], BDL[2],
            BDR[0], BDR[1], BDR[2],
            FDR[0], FDR[1], FDR[2],


         // Right face
            FUR[0], FUR[1], FUR[2],
            FDR[0], FDR[1], FDR[2],
            BDR[0], BDR[1], BDR[2],
            BUR[0], BUR[1], BUR[2],

         // Left face
            FUL[0], FUL[1], FUL[2],
            FDL[0], FDL[1], FDL[2],
            BDL[0], BDL[1], BDL[2],
            BUL[0], BUL[1], BUL[2]
        ];
    }

    function getNormals() {
        var F = vec.unit(vec.cross(loc.orientation.right, loc.orientation.up)),
            U = vec.unit(loc.orientation.up),
            R = vec.unit(loc.orientation.right),
            B = vec.muls(-1, F),
            D = vec.muls(-1, U),
            L = vec.muls(-1, R);

        return [
         // Front
            F[0], F[1], F[2],
            F[0], F[1], F[2],
            F[0], F[1], F[2],
            F[0], F[1], F[2],
         // Back
            B[0], B[1], B[2],
            B[0], B[1], B[2],
            B[0], B[1], B[2],
            B[0], B[1], B[2],
         // Top
            U[0], U[1], U[2],
            U[0], U[1], U[2],
            U[0], U[1], U[2],
            U[0], U[1], U[2],
         // Bottom
            D[0], D[1], D[2],
            D[0], D[1], D[2],
            D[0], D[1], D[2],
            D[0], D[1], D[2],
         // Right
            R[0], R[1], R[2],
            R[0], R[1], R[2],
            R[0], R[1], R[2],
            R[0], R[1], R[2],
         // Left
            L[0], L[1], L[2],
            L[0], L[1], L[2],
            L[0], L[1], L[2],
            L[0], L[1], L[2]
        ];
    }

    function getTextureCoords() {
        return [
            // Front
            0.0,  1.0 - 0.125 * colors[0],
            1.0,  1.0 - 0.125 * colors[0],
            1.0,  1.0 - 0.125 * (1 + colors[0]),
            0.0,  1.0 - 0.125 * (1 + colors[0]),
            // Back
            0.0,  1.0 - 0.125 * colors[1],
            1.0,  1.0 - 0.125 * colors[1],
            1.0,  1.0 - 0.125 * (1 + colors[1]),
            0.0,  1.0 - 0.125 * (1 + colors[1]),
            // Top
            0.0,  1.0 - 0.125 * colors[2],
            1.0,  1.0 - 0.125 * colors[2],
            1.0,  1.0 - 0.125 * (1 + colors[2]),
            0.0,  1.0 - 0.125 * (1 + colors[2]),
            // Bottom
            0.0,  1.0 - 0.125 * colors[3],
            1.0,  1.0 - 0.125 * colors[3],
            1.0,  1.0 - 0.125 * (1 + colors[3]),
            0.0,  1.0 - 0.125 * (1 + colors[3]),
            // Right
            0.0,  1.0 - 0.125 * colors[4],
            1.0,  1.0 - 0.125 * colors[4],
            1.0,  1.0 - 0.125 * (1 + colors[4]),
            0.0,  1.0 - 0.125 * (1 + colors[4]),
            // Left
            0.0,  1.0 - 0.125 * colors[5],
            1.0,  1.0 - 0.125 * colors[5],
            1.0,  1.0 - 0.125 * (1 + colors[5]),
            0.0,  1.0 - 0.125 * (1 + colors[5])
        ];
    }

    function getIndices() {
        return [
            0,  1,  2,      0,  2,  3,    // front
            4,  5,  6,      4,  6,  7,    // back
            8,  9,  10,     8,  10, 11,   // top
            12, 13, 14,     12, 14, 15,   // bottom
            16, 17, 18,     16, 18, 19,   // right
            20, 21, 22,     20, 22, 23    // left
        ];
    }

    function resetBuffers() {
        var gl,
            vertices,
            vertexNormals,
            textureCoordinates,
            vertexIndices;

        gl = scene.gl;
        vertices = getVertices();
        vertexNormals = getNormals();
        textureCoordinates = getTextureCoords();
        vertexIndices = getIndices();

        buffers.vertices = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, buffers.vertices);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices),
                      gl.STATIC_DRAW);

        buffers.normals = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, buffers.normals);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertexNormals),
                      gl.STATIC_DRAW);

        buffers.texture = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, buffers.texture);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(textureCoordinates),
                      gl.STATIC_DRAW);

        buffers.indices = gl.createBuffer();
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, buffers.indices);
        gl.bufferData(gl.ELEMENT_ARRAY_BUFFER,
                      new Uint16Array(vertexIndices), gl.STATIC_DRAW);
    }

    function draw(scene) {
        var gl = scene.gl;
        scene.mvPushMatrix();
        gl.bindBuffer(gl.ARRAY_BUFFER, buffers.vertices);
        gl.vertexAttribPointer(scene.vertexPositionAttribute, 3,
                               gl.FLOAT, false, 0, 0);
        gl.bindBuffer(gl.ARRAY_BUFFER, buffers.texture);
        gl.vertexAttribPointer(scene.textureCoordAttribute, 2,
                               gl.FLOAT, false, 0, 0);
        gl.bindBuffer(gl.ARRAY_BUFFER, buffers.normals);
        gl.vertexAttribPointer(scene.vertexNormalAttribute, 3,
                               gl.FLOAT, false, 0, 0);
        // Specify the texture to map onto the faces.
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, texture);
        gl.uniform1i(gl.getUniformLocation(scene.shaderProgram, "uSampler"), 0);
        // Draw the cube.
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, buffers.indices);
        scene.setMatrixUniforms();
        gl.drawElements(gl.TRIANGLES, 36, gl.UNSIGNED_SHORT, 0);
        scene.mvPopMatrix();
    }

    function createTexture() {
        var gl,
            canvas,
            body,
            image,
            ctx,
            W,
            H,
            f,
            tex;
        gl = scene.gl;
        canvas = document.createElement('canvas');
        canvas.id     = "hiddenCanvas";
        canvas.width  = 128;
        canvas.height = 128 * 8;
        canvas.style.display   = "none";
        body = document.getElementsByTagName("body")[0];
        body.appendChild(canvas);
        // draw texture
        image = document.getElementById('hiddenCanvas');
        ctx = image.getContext('2d');
        ctx.beginPath();
        ctx.rect(0, 0, ctx.canvas.width / 2,
                 ctx.canvas.height / 2);
        ctx.fillStyle = 'white';
        ctx.fill();

        W = 128;
        H = 128;

        for (f = 0; f < settings.colors.length; f += 1) {
            ctx.beginPath();
            ctx.rect(0, H * f, W, H);
            ctx.fillStyle = settings.colors[f];
            ctx.fill();
        }
        ctx.restore();
        // create new texture
        tex = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, tex);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER,
                         gl.LINEAR_MIPMAP_NEAREST);
        gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
        gl.bindTexture(gl.TEXTURE_2D, tex);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA,
                      gl.UNSIGNED_BYTE, image);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER,
                         gl.LINEAR_MIPMAP_NEAREST);
        gl.generateMipmap(gl.TEXTURE_2D);
        gl.bindTexture(gl.TEXTURE_2D, null);
        return tex;
    }

    function tostring() {
        var outwardColors = [];
        for (var i = 0; i < colors.length; i++) {
            var color = colors[i];
            if (color !== -1) {
                outwardColors.push(settings.colors[color]);
            }
        }
        switch (outwardColors.length) {
        case 0:
            return "inner piece";
        case 1:
            return outwardColors[0] + " center piece";
        case 2:
            return outwardColors[0] + " and " +
                outwardColors[1] + " edge piece";
        case 3:
            return outwardColors[0] + ", " +
                outwardColors[1] + ", and " +
                outwardColors[2] + " corner piece";
        default:
            return "unknown piece";
        }
    }

    function init(sce, set, cols) {
        scene = sce;
        settings = set;
        colors = cols;
        texture = createTexture();
        resetBuffers();
    }
    init(sceneData, set, cols);

    this.draw = draw;
    this.moving = moving;
    this.rotate = rotate;
    this.loc = loc;
    this.ploc = ploc;
    this.home = home;
    this.returnHome = returnHome;
    this.getState = getState;
    this.str = tostring;
}

function Cubelets(set) {
    'use strict';
    var cubes,
        scene,
        settings;

    function linkRendering(d) {
        scene = d;
    }

    function draw() {
        var i;
        for (i = 0; i < cubes.length; i += 1) {
            cubes[i].draw(scene);
        }
    }

    function removeAll() {
        cubes = [];
    }

    function add(x, y, z, len, up, right, colors) {
        cubes.push(new Cube(scene, x, y, z, len, up, right,
                            colors, settings, true));
    }

    function makeMove(move, moveFrameStart) {
        var i,
            cube;
        if (move.hasOwnProperty("action")) {
            move.action();
            return true;
        }
        for (i = 0; i < cubes.length; i += 1) {
            cube = cubes[i];
            if (move.applies(cube)) {
                if (cube.moving.currently) {
                    return false;
                }
            }
        }
        for (i = 0; i < cubes.length; i += 1) {
            cube = cubes[i];
            if (move.applies(cube)) {
                cube.rotate(move.axis, move.angle, moveFrameStart);
            }
        }
        return true;
    }

    function makeMoveOnState(cubes, move) {
        var i,
            cube;
        for (i = 0; i < cubes.length; i += 1) {
            cube = cubes[i];
            if (move.applies(cube)) {
                cube.rotate(move.axis, move.angle, 5);
                cube.moving.stop();
            }
        }
        return cubes;
    }

    function getMove(moves, moveIdx) {
        return moves[moveIdx];
    }

    function getIdx(moves, axis, angle) {
        var moveIdx,
            move;
        for (moveIdx in moves) {
            if (moves.hasOwnProperty(moveIdx)) {
                move = moves[moveIdx];
                if (vec.parallel(move.axis, axis) &&
                    feq(move.angle, angle)) {
                    return moveIdx;
                }
            }
        }
        throw ("Move not found: (" + vec.str(axis) + ", " +
               angle.toString() + ")");
    }

    function setState(state) {
        var i,
            cube;
        if (state === "solved") {
            for (i = 0; i < cubes.length; i += 1) {
                cube = cubes[i];
                cube.returnHome();
            }
        }
    }

    function opposite(move, state) {
        if (move.hasOwnProperty("action")) {
            return move;
        }
        else {
            var angle = move.angle;
            var axis = move.axis;
            angle *= -1;
            angle = principal_value(angle);
            return state.getMoveFromIdx(state.getIdxFromMove(axis, angle));
        }
    }

    function getState(moves) {
        var i,
            cube,
            state = {};
        state.cubes = [];
        for (i = 0; i < cubes.length; i += 1) {
            cube = cubes[i];
            state.cubes.push(cube.getState());
        }
        state.makeMove = function (move) {
            return makeMoveOnState(state.cubes, move);
        };
        state.unmakeMove = function (move) {
            return makeMoveOnState(state.cubes, opposite(move, state));
        };
        state.getMoveFromIdx = function(moveIdx) {
            return getMove(moves, moveIdx);
        };
        state.moves = moves;
        state.getIdxFromMove = function (axis, angle) {
            return getIdx(moves, axis, angle);
        };
        state.isSolved = function () {
            for (var k = 0; k < cubes.length; k += 1) {
                var cube = cubes[k];
                if (!cube.isHome())
                    return false;
            }
            return true;
        };
        return state;
    }

    function updateRotation() {
        var i,
            cube;
        for (i = 0; i < cubes.length; i += 1) {
            cube = cubes[i];
            if (cube.moving.currently) {
                cube.moving.update()
            }
        }
    }

    function init(set) {
        cubes = [];
        settings = set;
    }
    init(set);

    this.linkRendering = linkRendering;
    this.draw = draw;
    this.removeAll = removeAll;
    this.updateRotation = updateRotation;
    this.makeMove = makeMove;
    this.add = add;
    this.setState = setState;
    this.getState = getState;
}

function RubiksCube(sce, set) {
    'use strict';
    var cubelets,
        scene,
        moveQueue,
        moves,
        settings;

    function setVersion(versionID) {
        cubelets.removeAll();
        var sl,
            min,
            max,
            z,
            y,
            x,
            colors;
        sl = versionID;
        min = -sl + 1;
        max = sl - 1;
        for (z = min; z <= max; z += 2) {
            for (y = min; y <= max; y += 2) {
                for (x = min; x <= max; x += 2) {
                    colors = [ (z === max) ? 0 : -1,
                               (z === min) ? 1 : -1,
                               (y === max) ? 2 : -1,
                               (y === min) ? 3 : -1,
                               (x === max) ? 4 : -1,
                               (x === min) ? 5 : -1
                             ];
                    cubelets.add(x, y, z, 1.95,
                                 [0, 1, 0], [1, 0, 0], colors);
                }
            }
        }
    }

    function checkForMoves(key, keys) {
        var i,
            move;
        for (i in moves) {
            if (moves.hasOwnProperty(i)) {
                move = moves[i];
                if (!keys[key] && (move.key === key) &&
                    keys[KEYCODES.shift] === move.shiftReq &&
                    keys[KEYCODES.two] === move.tReq) {
                    moveQueue.push(move);
                    break;
                }
            }
        }
    }

    function randomMove() {
        var result,
            key,
            count;
        count = 0;
        for (key in moves) {
            if (moves.hasOwnProperty(key)) {
                count += 1;
                if (Math.random() < 1 / count) {
                    result = moves[key];
                }
            }
        }
        return result;
    }

    function makeMoves(moveList) {
        enqueueMoves(moveList, false);
    };

    function enqueueMoves(moves, clearOthers) {
        if (clearOthers) {
            moveQueue = [];
        }
        moveQueue.push.apply(moveQueue, moves);
    }

    function shuffle(startAction, endAction, numMoves) {
        var i,
            randomMoves,
            info;
        randomMoves = [];
        info = {moves: randomMoves,
                newMove: function (i) {
                info.moves.push(randomMove());
                info.moves.push({action: function () {
                            cubr.updateProgressBar((1.0* i) / numMoves);
                        }});
            }
        };
        randomMoves.push({"action": startAction});
        for (i = 0; i < numMoves; i += 1) {
            info.newMove(i);
        }
        randomMoves.push({"action": function () {
                    cubr.updateProgressBar(0);
                }});
        randomMoves.push({"action": endAction});
        enqueueMoves(randomMoves, false);
    }

    function makeMove(move) {
        return cubelets.makeMove(move, toInt(settings.speed *
                                             Math.sqrt(Math.abs(move.angle))));
    }

    function cycleMoves() {
        var soFar = 0;
        var move;
        while (moveQueue.length > 0 &&
               soFar < settings.movesPerFrame &&
               !settings.paused) {
            move = moveQueue[0];
            if (!makeMove(move)) {
                break;
            } else {
                moveQueue.shift();
            }
            soFar++;
        }
    }

    function rotate(dx, dy) {
        var right,
            up,
            inv,
            newRight,
            newUp;
        right = $V([1.0, 0.0, 0.0, 0.0]);
        up = $V([0.0, 1.0, 0.0, 0.0]);

        inv = scene.data.mvMatrix.inverse();
        newRight = inv.x(right);
        if (dy !== 0) {
            scene.data.mvRotate(dy, [newRight.elements[0],
                                     newRight.elements[1],
                                     newRight.elements[2]]);
        }

        newUp = inv.x(up);
        if (dx !== 0) {
            scene.data.mvRotate(dx, [newUp.elements[0],
                                     newUp.elements[1],
                                     newUp.elements[2]]);
        }

    }


    function update(keys, momentum) {
        var r,
            dx,
            dy;
        cycleMoves();
        cubelets.updateRotation();
        r = settings.rotateSpeed;
        dx = 0;
        dy = 0;
        if (keys[KEYCODES.up]) {
            dy -= r;
        }
        if (keys[KEYCODES.down]) {
            dy += r;
        }
        if (keys[KEYCODES.left]) {
            dx -= r;
        }
        if (keys[KEYCODES.right]) {
            dx += r;
        }
        rotate(dx + momentum.x, dy + momentum.y);
    }

    function initMoves() {
        moveQueue = [];
        moves = {};
        /* Front */
        moves["f"] = {
            axis: [0.0, 0.0, 2.0],
            key: KEYCODES.f,
            shiftReq: false,
            tReq: false,
            angle: -Math.PI / 2,
            applies: function (c) {return feq(c.ploc.pos[2], 2); }
        };
        moves["f'"] = {
            axis: [0.0, 0.0, 2.0],
            key: KEYCODES.f,
            shiftReq: true,
            tReq: false,
            angle: Math.PI / 2,
            applies: function (c) {return feq(c.ploc.pos[2], 2); }
        };
        moves["f2"] = {
            axis: [0.0, 0.0, 2.0],
            key: KEYCODES.f,
            shiftReq: false,
            tReq: true,
            angle: Math.PI,
            applies: function (c) {return feq(c.ploc.pos[2], 2); }
        }
        /* Back */
        moves["b"] = {
            axis: [0.0, 0.0, -2.0],
            key: KEYCODES.b,
            shiftReq: false,
            tReq: false,
            angle: -Math.PI / 2,
            applies: function (c) {return feq(c.ploc.pos[2], -2); }
        };
        moves["b'"] = {
            axis: [0.0, 0.0, -2.0],
            key: KEYCODES.b,
            shiftReq: true,
            tReq: false,
            angle: Math.PI / 2,
            applies: function (c) {return feq(c.ploc.pos[2], -2); }
        };
        moves["b2"] = {
            axis: [0.0, 0.0, -2.0],
            key: KEYCODES.b,
            shiftReq: false,
            tReq: true,
            angle: Math.PI,
            applies: function (c) {return feq(c.ploc.pos[2], -2); }
        };
        /* Right */
        moves["r"] = {
            axis: [2.0, 0.0, 0.0],
            key: KEYCODES.r,
            shiftReq: false,
            tReq: false,
            angle: -Math.PI / 2,
            applies: function (c) {return feq(c.ploc.pos[0], 2); }
        };
        moves["r'"] = {
            axis: [2.0, 0.0, 0.0],
            key: KEYCODES.r,
            shiftReq: true,
            tReq: false,
            angle: Math.PI / 2,
            applies: function (c) {return feq(c.ploc.pos[0], 2); }
        };
        moves["r2"] = {
            axis: [2.0, 0.0, 0.0],
            key: KEYCODES.r,
            shiftReq: false,
            tReq: true,
            angle: Math.PI,
            applies: function (c) {return feq(c.ploc.pos[0], 2); }
        };
        /* Left */
        moves["l"] = {
            axis : [-2.0, 0.0, 0.0],
            angle : -Math.PI / 2,
            key: KEYCODES.l,
            shiftReq: false,
            tReq: false,
            applies : function (c) {return feq(c.ploc.pos[0], -2); }
        };
        moves["l'"] = {
            axis : [-2.0, 0.0, 0.0],
            angle : Math.PI / 2,
            key: KEYCODES.l,
            shiftReq: true,
            tReq: false,
            applies : function (c) {return feq(c.ploc.pos[0], -2); }
        };
        moves["l2"] = {
            axis : [-2.0, 0.0, 0.0],
            angle : Math.PI,
            key: KEYCODES.l,
            shiftReq: false,
            tReq: true,
            applies : function (c) {return feq(c.ploc.pos[0], -2); }
        };
        /* Up */
        moves["u"] = {
            axis : [0.0, 2.0, 0.0],
            key: KEYCODES.u,
            shiftReq: false,
            tReq: false,
            angle : -Math.PI / 2,
            applies : function (c) {return feq(c.ploc.pos[1], 2); }
        };
        moves["u'"] = {
            axis : [0.0, 2.0, 0.0],
            key: KEYCODES.u,
            shiftReq: true,
            tReq: false,
            angle : Math.PI / 2,
            applies : function (c) {return feq(c.ploc.pos[1], 2); }
        };
        moves["u2"] = {
            axis : [0.0, 2.0, 0.0],
            key: KEYCODES.u,
            shiftReq: false,
            tReq: true,
            angle : Math.PI,
            applies : function (c) {return feq(c.ploc.pos[1], 2); }
        };
        /* Down */
        moves["d"] = {
            axis : [0.0, -2.0, 0.0],
            key: KEYCODES.d,
            shiftReq: false,
            tReq: false,
            angle : -Math.PI / 2,
            applies : function (c) {return feq(c.ploc.pos[1], -2); }
        }
        moves["d'"] = {
            axis : [0.0, -2.0, 0.0],
            key: KEYCODES.d,
            shiftReq: true,
            tReq: false,
            angle : Math.PI / 2,
            applies : function (c) {return feq(c.ploc.pos[1], -2); }
        }
        moves["d2"] = {
            axis : [0.0, -2.0, 0.0],
            key: KEYCODES.d,
            shiftReq: false,
            tReq: true,
            angle : Math.PI,
            applies : function (c) {return feq(c.ploc.pos[1], -2); }
        }
    }

    function setState(state, abort) {
        var i,
            move;
        if (abort) {
            for (i = 0; i < moveQueue.length; i += 1) {
                move = moveQueue[i];
                if (move.hasOwnProperty("action")) {
                    move.action();
                }
            }
            moveQueue = [];
        }
        cubelets.setState(state);
    }

    function getState() {
        return cubelets.getState(moves);
    };

    function init(sce, set) {
        scene = sce;
        settings = set;
        cubelets = new Cubelets(settings);
        initMoves();
        scene.linkObjects(cubelets);
    }
    init(sce, set);

    this.setVersion = setVersion;
    this.setState = setState;
    this.getState = getState;
    this.update = update;
    this.checkForMoves = checkForMoves;
    this.rotate = rotate;
    this.shuffle = shuffle;
    this.makeMoves = makeMoves;
}

function getShuffleLength() {
    var len = document.getElementById("shuffleLength");
    return parseInt(len.options[len.selectedIndex].value);
}

function getTutorial() {
    var tut = document.getElementById("tutorSelect");
    return tut.options[tut.selectedIndex].value === "on";
}


function getWindowSize() {
    return {"width": window.innerWidth,
            "height": window.innerHeight};
}

function WebcamInterface(canvasID, settings) {
    var settings = settings;
    var canvas = document.getElementById(canvasID);
    var active = false;
    var cameraUp = false;
    var hovering = false;
    var size;

    function analyzeFrame(b64) {
        var img = new Image();
        var context = canvas.getContext('2d');
        img.onload = function () {
            context.drawImage(this, 0, 0, canvas.width, canvas.height);
        }
        img.src = "data:image/jpeg;base64," + b64;
    }

    function onError(errorId,errorMsg) {
        alert(errorMsg);
    }
    function changeCamera() {
        $.scriptcam.changeCamera($('#cameraNames').val());
    }
    function onWebcamReady(cameraNames,camera,microphoneNames,
                           microphone,volume) {
        cameraUp = true;
        $.each(cameraNames, function(index, text) {
                $('#cameraNames')
                    .append( $('<option></option>').val(index).html(text) )
                    });
        $('#cameraNames').val(camera);
    }
    function activate() {
        active = true;
        $("#webcam").scriptcam({
                cornerRadius:0,
                    width: 640,
                    height: 480,
                    useMicrophone: false,
                    onError:onError,
                    onWebcamReady:onWebcamReady
                    });
        document.getElementById("webcam").height = 0;
    }

    function getCanvasSize(canvas, settings) {
       var size = getWindowSize();
       size.width *= settings.webcam.widthFraction;
       size.height *= settings.webcam.heightFraction;
       size.width = min(size.width, settings.webcam.maxWidth);
       size.height = min(size.height, settings.webcam.maxHeight);
       return size;
    }

    function drawIdle() {
        var context = canvas.getContext('2d');
        size = getCanvasSize(canvas, settings);
        if (context.canvas.width != size.width ||
            context.canvas.height != size.height) {
            context.canvas.width = size.width;
            context.canvas.height = size.height;
        }
        context.clearRect(0, 0, size.width, size.height);
        context.fillStyle = "black";
        context.fillRect(0, 0, size.width, size.height);
        if (hovering) {
            context.fillStyle = "#333333";
            context.fillRect(size.width/2-100,size.height/2-30,200,50);
        }
        context.shadowColor = "transparent";
        context.fillStyle = "white";
        context.font = "bold 16px Arial";
        context.textAlign = "center";
        context.fillText("Click to open webcam", size.width/2, size.height/2);
    }

    function drawLoading() {
        var context = canvas.getContext('2d');
        size = getCanvasSize(canvas, settings);
        if (context.canvas.width != size.width ||
            context.canvas.height != size.height) {
            context.canvas.width = size.width;
            context.canvas.height = size.height;
        }
        context.clearRect(0, 0, size.width, size.height);
        context.fillStyle = "black";
        context.fillRect(0, 0, size.width, size.height);
        context.shadowColor = "transparent";
        context.fillStyle = "white";
        context.font = "bold 16px Arial";
        context.textAlign = "center";
        context.fillText("Waiting for webcam...", size.width/2, size.height/2);
    }

    function update() {
        if (active) {
            if (cameraUp) {
                /* Only do this every 200 ms or so, to prevent lag. */
                var raw = $.scriptcam.getFrameAsBase64();
                analyzeFrame(raw);
            } else {
                drawLoading();
            }
        } else {
            drawIdle();
        }
    }

    function onMouseMove(e) {
        var x = e.layerX;
        var y = e.layerY;
        hovering = (x >= size.width/2-100 &&
                    x <= size.width/2+100 &&
                    y >= size.height/2-30 &&
                    y <= size.height/2+60);
    }
    function onMouseDown(e) {
        if (!active)
            activate();
    }

    this.onMouseMove = onMouseMove;
    this.onMouseDown = onMouseDown;
    this.update = update;
}

function Cubr() {
    'use strict';
    var scene,
        cube,
        keys,
        cam,
        mouse = {down: false, last: [0, 0]},
        momentum = {x: 0, y: 0},
        settings = {
            timerInterval: 20,
            rotateSpeed: Math.PI / 48,
            defaultSpeed: 12,
            speed: 12,
            movesPerFrame: 3,
            defaultMPF: 3,
            dragSensitivity: 0.003,
            inertia: 0.75,
            colors: ["green", "blue", "white", "yellow",
                     "red", "orange", "pink", "#303030"],
            startMomentum: {
                x: 0.55,
                y: 1.65
            },
            webcam: {
                maxWidth: 320,
                maxHeight: 240,
                widthFraction: 0.5,
                heightFraction: 1.0
            },
            glcanvas: {
                maxWidth: 320,
                widthFraction: 0.5,
                heightFraction: 1.0
            },
            shuffleLength: getShuffleLength,
            tutorial: getTutorial,
            progBar: {
                queueMin: 3,
                queueMax: 10,
                color: "green",
                margin: 10,
                thickness: 20
            },
            turnAcceleration: 2/3,
            paused: false
        };

    function resetKeys() {
        var k;
        keys = [];
        for (k = 0; k < 256; k += 1) {
            keys.push(false);
        }
    }

    function timerFired() {
        cam.update();
        cube.update(keys, momentum);
        momentum.x *= settings.inertia;
        momentum.y *= settings.inertia;
        scene.draw();
    }

    function onKeyDown(e) {
        var keyCode = e.keyCode;
        if (cube) {
            cube.checkForMoves(keyCode, keys);
        }
        if (33 <= keyCode && keyCode <= 40) {
            e.preventDefault();
        }
        keys[keyCode] = true;
    }

    function onKeyUp(e) {
        var keyCode = e.keyCode;
        keys[keyCode] = false;
    }

    function onMouseDown(e) {
        if (e.toElement.id === "glcanvas") {
            e.preventDefault();
            mouse.down = true;
            mouse.last = [e.x, e.y];
        } else if (e.toElement.id === "webcamcanvas") {
            cam.onMouseDown();
        }
    }

    function onMouseUp() {
        mouse.down = false;
    }

    function onMouseMove(e) {
        if (mouse.down) {
            e.preventDefault();
            if (e.toElement.id === "glcanvas") {
                momentum.x += settings.dragSensitivity * (e.x - mouse.last[0]);
                momentum.y += settings.dragSensitivity * (e.y - mouse.last[1]);
                mouse.last = [e.x, e.y];
            }
        }
        if (e.toElement.id === "webcamcanvas") {
            cam.onMouseMove(e);
        }
    }

    function bindEventListeners() {
        resetKeys();
        document.addEventListener("keydown", onKeyDown, false);
        document.addEventListener("keyup", onKeyUp, false);
        document.addEventListener("mousedown", onMouseDown, false);
        document.addEventListener("mouseup", onMouseUp, false);
        document.addEventListener("mousemove", onMouseMove, false);
        // Timer:
        setInterval(timerFired, settings.timerInterval);
    }

    function reset() {
        cube.setState("solved", true);
        momentum.x = settings.startMomentum.x;
        momentum.y = settings.startMomentum.y;
    }

    function run() {
        scene = new SimpleScene("glcanvas");
        cube = new RubiksCube(scene, settings);
        cam = new WebcamInterface("webcamcanvas", settings);
        cube.setVersion(3);
        bindEventListeners();
        reset();
    }

    function shuffle() {
        var numMoves = settings.shuffleLength();
        cube.shuffle(function () {
                settings.speed = (numMoves < 100) ? 3: 0;
                settings.movesPerFrame = (numMoves > 500) ? 20: 3;
            },
            function () {
                settings.speed = settings.defaultSpeed;
                settings.movesPerFrame = settings.defaultMPF;
            }, numMoves);
    }

    function solve() {
        var moves = getSolution(cube.getState());
        cube.makeMoves(moves);
    }

    function updateStatus(s) {
        var box = document.getElementById("statusBox");
        box.innerHTML = s;
    }

    function updateProgressBar(percent) {
        var bar = document.getElementById("progressBar");
        var cont = document.getElementById("progressBarContainer");
        var maxWidth = cont.width.baseVal.value;
        bar.setAttribute("width", (~~(percent * maxWidth)).toString());
    }

    function updatePauseButton() {
        var text = settings.paused ? "Unpause" : "Pause";
        document.getElementById("pause").value = text;
    };

    this.run = run;
    this.reset = reset;
    this.shuffle = shuffle;
    this.solve = solve;
    this.updateStatus = updateStatus;
    this.updateProgressBar = updateProgressBar;
    this.pause = function () {
        settings.paused = true;
        updatePauseButton();
    };
    this.unpause = function () {
        settings.paused = false;
        updatePauseButton();
    };
    this.togglePause = function () {
        settings.paused = !(settings.paused);
        updatePauseButton();
    };
    this.pauseOnTutorial = function () {
        if (settings.tutorial()) {
            settings.paused = true;
            updatePauseButton();
        }
    };
    this.slowdown = function () {
        settings.speed = min(settings.speed + 2, 30);
    };
    this.speedup = function() {
        settings.speed = max(settings.speed - 2, 1);
    };
}
var cubr = new Cubr();
