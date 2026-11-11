/*
 * ARL differential packing: the C extension arlmet._pack.
 *
 * pack_core is HYSPLIT's PAKOUT (library/hysplit/pakout.f) and unpack_core
 * its PAKINP (pakinp.f), with each cell's arithmetic in single precision and
 * in HYSPLIT's order, so they give the same bytes and values bit for bit.
 * PAKOUT writes each cell's byte as its difference from the previous value
 * *as a reader will rebuild it* (the feedback loop), which keeps the error of
 * every cell within half a packing step.
 *
 * In both, only the first column chains from row to row; past it each row
 * chains only from its own first cell, so the first column is done first.
 * Every product is by a power of 2, so it is exact, and a compiler that
 * fuses multiply and add (clang on arm64) changes no result.
 *
 * Python signatures:
 *   pack_core(unpacked: np.ndarray[float32, 2-D (ny, nx)])
 *       -> (packed: np.ndarray[uint8, 2-D], precision: float,
 *           exponent: int, initial_value: float)
 *   unpack_core(packed: bytes-like, nx, ny, x_start, x_stop, y_start, y_stop,
 *               precision: float, exponent: int, initial_value: float)
 *       -> np.ndarray[float32, 2-D (y_stop - y_start, x_stop - x_start)]
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>
#include <stdint.h>

/* Rows pack_core packs together, past the first column. */
#define BLOCK_ROWS 8

static PyObject *
pack_core(PyObject *self, PyObject *args)
{
    PyArrayObject *unpacked_in;

    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &unpacked_in))
        return NULL;

    /* Ensure contiguous C-order float32. */
    PyArrayObject *arr = (PyArrayObject *)PyArray_FROM_OTF(
        (PyObject *)unpacked_in, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    if (arr == NULL)
        return NULL;

    if (PyArray_NDIM(arr) != 2) {
        PyErr_SetString(PyExc_ValueError, "unpacked must be a 2-D array");
        Py_DECREF(arr);
        return NULL;
    }

    npy_intp ny = PyArray_DIM(arr, 0);
    npy_intp nx = PyArray_DIM(arr, 1);
    if (nx == 0 || ny == 0) {
        PyErr_SetString(PyExc_ValueError, "unpacked must not be empty");
        Py_DECREF(arr);
        return NULL;
    }

    npy_intp dims[2] = {ny, nx};
    PyArrayObject *packed = (PyArrayObject *)PyArray_SimpleNew(
        2, dims, NPY_UINT8);
    if (packed == NULL) {
        Py_DECREF(arr);
        return NULL;
    }

    const float   *rvar = (const float   *)PyArray_DATA(arr);
          uint8_t *cvar = (      uint8_t *)PyArray_DATA(packed);

    float var1 = rvar[0];

    /* The largest difference between neighbors along each row; the first
     * element of a row is differenced from the first element of the row
     * before. */
    float rold = var1;
    float rmax = 0.0f;
    for (npy_intp j = 0; j < ny; j++) {
        for (npy_intp i = 0; i < nx; i++) {
            float d = fabsf(rvar[j * nx + i] - rold);
            if (d > rmax)
                rmax = d;
            rold = rvar[j * nx + i];
        }
        rold = rvar[j * nx];
    }

    /* The scaling exponent: positive or whole-number logs round up. */
    float sexp = 0.0f;
    if (rmax != 0.0f)
        sexp = logf(rmax) / logf(2.0f);
    int nexp = (int)sexp;
    if (sexp >= 0.0f || fmodf(sexp, 1.0f) == 0.0f)
        nexp += 1;
    /* The differences span -127 to 127, 254 steps. */
    float prec = ldexpf(1.0f, nexp) / 254.0f;
    float scexp = ldexpf(1.0f, 7 - nexp);
    /* PAKOUT divides by SCEXP.  SCEXP is a power of 2, so its reciprocal is
     * exact (0 where SCEXP overflows to inf, as x / inf is 0) and multiplying
     * by it gives the same bits, faster. */
    float rscexp = 1.0f / scexp;

    /* Each difference is from the previous value as it will be unpacked.
     * Column 0 comes first: each row's first cell is differenced from the
     * first cell of the row before, one chain down the rows. */
    float *rcol = PyMem_Malloc((size_t)ny * sizeof(float));
    if (rcol == NULL) {
        Py_DECREF(arr);
        Py_DECREF(packed);
        return PyErr_NoMemory();
    }
    rold = var1;
    for (npy_intp j = 0; j < ny; j++) {
        npy_intp k = j * nx;
        int icval = (int)((rvar[k] - rold) * scexp + 127.5f);
        rold = (float)(icval - 127) * rscexp + rold;
        rcol[j] = rold;
        cvar[k] = (uint8_t)icval;
    }

    /* The rest of each row, BLOCK_ROWS rows together: the processor overlaps
     * their chains, about 5 times faster than one row at a time. */
    for (npy_intp j0 = 0; j0 < ny; j0 += BLOCK_ROWS) {
        npy_intp nb = ny - j0 < BLOCK_ROWS ? ny - j0 : BLOCK_ROWS;
        float rows[BLOCK_ROWS];
        for (npy_intp b = 0; b < nb; b++)
            rows[b] = rcol[j0 + b];
        for (npy_intp i = 1; i < nx; i++) {
            for (npy_intp b = 0; b < nb; b++) {
                npy_intp k = (j0 + b) * nx + i;
                int icval = (int)((rvar[k] - rows[b]) * scexp + 127.5f);
                rows[b] = (float)(icval - 127) * rscexp + rows[b];
                cvar[k] = (uint8_t)icval;
            }
        }
    }
    PyMem_Free(rcol);

    Py_DECREF(arr);
    return Py_BuildValue("Ndid", (PyObject *)packed, (double)prec, nexp,
                         (double)var1);
}

static PyObject *
unpack_core(PyObject *self, PyObject *args)
{
    Py_buffer buf;
    Py_ssize_t nx, ny, x0, x1, y0, y1;
    double precision, initial_value;
    int exponent;

    if (!PyArg_ParseTuple(args, "y*nnnnnndid", &buf, &nx, &ny, &x0, &x1, &y0,
                          &y1, &precision, &exponent, &initial_value))
        return NULL;
    if (nx < 1 || ny < 1 || buf.len != nx * ny) {
        PyErr_SetString(PyExc_ValueError,
                        "packed must hold nx * ny bytes, with nx, ny >= 1");
        PyBuffer_Release(&buf);
        return NULL;
    }
    if (x0 < 0 || x0 > x1 || x1 > nx || y0 < 0 || y0 > y1 || y1 > ny) {
        PyErr_SetString(PyExc_ValueError, "window outside the packed grid");
        PyBuffer_Release(&buf);
        return NULL;
    }

    npy_intp w = x1 - x0;
    npy_intp dims[2] = {y1 - y0, w};
    PyArrayObject *out = (PyArrayObject *)PyArray_SimpleNew(2, dims, NPY_FLOAT32);
    float *rcol = PyMem_Malloc((size_t)(y1 - y0 + 1) * sizeof(float));
    if (out == NULL || rcol == NULL) {
        Py_XDECREF(out);
        PyMem_Free(rcol);
        PyBuffer_Release(&buf);
        return PyErr_NoMemory();
    }

    const uint8_t *cvar = (const uint8_t *)buf.buf;
    float *rvar = (float *)PyArray_DATA(out);
    float scexp = ldexpf(1.0f, exponent - 7);
    float prec = (float)precision;

    Py_BEGIN_ALLOW_THREADS

    /* The first column, one chain down the rows from the initial value, with
     * no precision test. */
    float rold = (float)initial_value;
    for (npy_intp j = 0; j < y1; j++) {
        rold = (float)((int)cvar[j * nx] - 127) * scexp + rold;
        if (j >= y0)
            rcol[j - y0] = rold;
    }

    /* The rest of each row, from its first cell.  PAKINP sets a value
     * smaller than PREC to zero in every column but the first; the running
     * value keeps it.  (Rows in blocks, as in pack_core, gain little here:
     * the chain is one add per cell.) */
    for (npy_intp j = y0; j < y1; j++) {
        const uint8_t *crow = cvar + j * nx;
        float *orow = rvar + (j - y0) * w;
        rold = rcol[j - y0];
        if (x0 == 0 && w > 0)
            orow[0] = rold;
        for (npy_intp i = 1; i < x1; i++) {
            rold = (float)((int)crow[i] - 127) * scexp + rold;
            if (i >= x0)
                orow[i - x0] = fabsf(rold) < prec ? 0.0f : rold;
        }
    }

    Py_END_ALLOW_THREADS

    PyMem_Free(rcol);
    PyBuffer_Release(&buf);
    return (PyObject *)out;
}

/* ---------- module plumbing -------------------------------------------- */

static PyMethodDef PackMethods[] = {
    {"pack_core", pack_core, METH_VARARGS,
     "HYSPLIT's PAKOUT: pack a 2-D float32 field as ARL byte differences."},
    {"unpack_core", unpack_core, METH_VARARGS,
     "HYSPLIT's PAKINP: unpack a window of an ARL byte-difference field."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef packmodule = {
    PyModuleDef_HEAD_INIT, "_pack", NULL, -1, PackMethods
};

PyMODINIT_FUNC
PyInit__pack(void)
{
    import_array();  /* initialize NumPy's C API; returns NULL on failure */
    return PyModule_Create(&packmodule);
}
