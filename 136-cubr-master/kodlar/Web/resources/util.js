/* Utility library by Chris Barker
 * originally written for Grid
 * http://cbarker.net/grid
 */

/* Miscellaneous */
function toInt(num) {
    // http://stackoverflow.com/questions/596467
    return ~~num;
}

function half(num) {
    return toInt(num/2);
}

function zfill(number, size) {
    /* convert int to string with leading zeros
     * python equivalent: (("%%0%dd" % size) % number)
     * https://gist.github.com/superjoe30/4382935
     */
    number = number.toString();
    while (number.length < size) number = "0" + number;
    return number;
}

function max(a, b) {
    /* Return the larger of two values. */
    return (a > b) ? a : b;
}

function min(a, b) {
    /* Return the smaller of two values. */
    return (a > b) ? b : a;
}

/* Copy */
function copyArray(A) {
    var arr = [];
    for (var i = 0; i < A.length; i++) {
        arr.push(A[i]);
    }
    return arr;
}

function copy2DArray(B) {
    var arr = [];
    for (var i = 0; i < B.length; i++) {
        arr.push(copyArray(B[i]));
    }
    return arr;
}

function copyObject(D) {
    var res = {};
    for (var key in D) {
        res[key] = D[key];
    }
    return res;
}

/* Arrays */
function all_array(A) {
    for (var i = 0; i < A.length; i++) {
        if (!(A[i]))
            return false;
    }
    return true;
}

function any_array(A) {
    for (var i = 0; i < A.length; i++) {
        if (A[i])
            return true;
    }
    return false;
}

/* Priority Queue */
function PriorityQueue() {
    this.Q = [0];
    this.size = 0;
    this.isEmpty = PQ_isEmpty;
    this.insert = PQ_insert;
    this.pop = PQ_pop;
}

function PQ_isEmpty() {
    return this.size == 0;
}

function PQ_insert(elem, priority) {
    this.Q.push([elem, priority]);
    this.size++;
    var idx = this.size; // Starting index of new element.
    while (idx > 1 && this.Q[half(idx)][1] > this.Q[idx][1]) {
        // The parent node is "higher" priority. Swap them
        var parent = this.Q[half(idx)];
        this.Q[half(idx)] = this.Q[idx];
        this.Q[idx] = parent;
        idx = half(idx);
    }
}

function PQ_pop() {
    var item,
        idx,
        child;
    if (this.isEmpty()) return;
    this.size--;
    item = this.Q[1][0];
    this.Q[1] = this.Q.pop();
    idx = 1;
    while (idx*2 < this.size && this.Q[idx*2][1] < this.Q[idx][1]) {
        child = this.Q[idx*2];
        this.Q[idx*2] = this.Q[idx];
        this.Q[idx] = child;
        idx *= 2;
    }
    return item;
}

/* vec is a limited vector library for strictly 3-dimensional vectors */
var vec = {
    add: function(A, B) {
        var C = [];
        for (var i = 0; i < A.length; i++) {
            C.push(A[i] + B[i]);
        }
        return C;
    },
    cross: function(A, B) {
        var C = [];
        C.push(A[1] * B[2] - A[2] * B[1]);
        C.push(A[2] * B[0] - A[0] * B[2]);
        C.push(A[0] * B[1] - A[1] * B[0]);
        return C;
    },
    muls: function(s, A) {
        var B = [];
        for (var i = 0; i < A.length; i++) {
            B.push(s*A[i]);
        }
        return B;
    },
    sub: function(A, B) {
        return vec.add(A, vec.muls(-1, B));
    },
    proj: function(A, B) {
        return vec.muls(vec.dot(A,B)/(vec.mag2(B)), B);
    },
    dot: function(A, B) {
        return A[0]*B[0] + A[1]*B[1] + A[2]*B[2];
    },
    mag2: function(A) {
        return A[0]*A[0] + A[1]*A[1] + A[2]*A[2];
    },
    without: function(A, B) {
        return vec.sub(A, vec.proj(A, B));
    },
    isZero: function(A) {
        return (A[0]==0 && A[1]==0 && A[2]==0);
    },
    ints: function(A) {
        return [Math.round(A[0]),
                Math.round(A[1]),
                Math.round(A[2])];
    },
    mag: function(A) {
        return Math.sqrt(vec.mag2(A));
    },
    zero: function() {
        return [0, 0, 0];
    },
    unit: function(A) {
        if (vec.isZero(A))
            return vec.zero();
        return vec.muls(1.0/vec.mag(A), A);
    },
    setMag: function(m, A) {
        if (m==0)
            return vec.zero();
        return vec.muls(m, vec.unit(A));
    },
    eq: function(A, B) {
        return (feq(A[0], B[0]) &&
                feq(A[1], B[1]) &&
                feq(A[2], B[2]));
    },
    parallel: function(A, B) {
        return (feq(0, vec.mag(vec.cross(A, B))) &&
                vec.dot(A, B) > 0);
    },
    parallels: function(A, B) {
        return (feq(0, vec.mag(vec.cross(A, B))) &&
                vec.dot(A, B) != 0);
    },
    str: function(A) {
        return ("<" + A[0].toString() + ", " +
                A[1].toString() + ", " +
                A[2].toString() + ">");
    },
    getPerpendicular: function (A) {
        var crossed = vec.cross(A, [0, 1, 0]);
        if (vec.isZero(crossed))
            return [1, 0, 0];
        else
            return crossed;
    },
    angleBetween: function (A, B) {
        /* (A dot B) = |A||B|cos(theta) */
        if (vec.isZero(A) || vec.isZero(B)) {
            return 0;
        } else {
            return Math.acos(vec.dot(A,B)/(vec.mag(A)*vec.mag(B)));
        }
    }
};

var feq = function(a, b) {
    return Math.abs(a-b) < 0.0001;
};

function afeq(a, b) {
    return feq(Math.abs(a), Math.abs(b));
};
