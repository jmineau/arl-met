/*
 * ARL differential packing: the C extension arlmet._pack.
 *
 * A line-by-line translation of HYSPLIT's PAKOUT (library/hysplit/pakout.f),
 * in single precision as PAKOUT is, so it writes the same bytes, exponent,
 * precision and initial value.  Each cell's byte is its difference from the
 * previous value *as a reader will rebuild it* (the feedback loop), which
 * keeps the error of every cell within half a packing step.
 *
 * Python signature:
 *   pack_core(unpacked: np.ndarray[float32, 2-D (ny, nx)])
 *       -> (packed: np.ndarray[uint8, 2-D], precision: float,
 *           exponent: int, initial_value: float)
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>
#include <stdint.h>

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

    /* Pack from low to high, each difference from the previous value as it
     * will be unpacked. */
    float rcol = var1;
    npy_intp k = 0;
    for (npy_intp j = 0; j < ny; j++) {
        rold = rcol;
        for (npy_intp i = 0; i < nx; i++, k++) {
            int icval = (int)((rvar[k] - rold) * scexp + 127.5f);
            rold = (float)(icval - 127) / scexp + rold;
            if (i == 0)
                rcol = rold;
            cvar[k] = (uint8_t)icval;
        }
    }

    Py_DECREF(arr);
    return Py_BuildValue("Ndid", (PyObject *)packed, (double)prec, nexp,
                         (double)var1);
}

/* ---------- module plumbing -------------------------------------------- */

static PyMethodDef PackMethods[] = {
    {"pack_core", pack_core, METH_VARARGS,
     "HYSPLIT's PAKOUT: pack a 2-D float32 field as ARL byte differences."},
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
