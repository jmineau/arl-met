/*
 * ARL differential pack core — C extension for arlmet.
 *
 * Implements the feedback-loop encoder that is the write-side inverse of
 * HYSPLIT's PAKINP cumsum decoder.  Each cell's delta is computed against the
 * *reconstructed* running value (i.e. what unpack's cumsum will reproduce),
 * not the original float.  This keeps per-cell quantisation error bounded to
 * 0.5 * inv_scale regardless of grid width.
 *
 * Python signature:
 *   pack_core(unpacked: np.ndarray[float32, 2-D],
 *             scale: float,
 *             inv_scale: float,
 *             initial_value: float) -> np.ndarray[uint8, 2-D]
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdint.h>

static PyObject *
pack_core(PyObject *self, PyObject *args)
{
    PyArrayObject *unpacked_in;
    double scale, inv_scale, initial_value;

    if (!PyArg_ParseTuple(args, "O!ddd",
                          &PyArray_Type, &unpacked_in,
                          &scale, &inv_scale, &initial_value))
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

    npy_intp dims[2] = {ny, nx};
    PyArrayObject *packed = (PyArrayObject *)PyArray_SimpleNew(
        2, dims, NPY_UINT8);
    if (packed == NULL) {
        Py_DECREF(arr);
        return NULL;
    }

    const float   *in  = (const float   *)PyArray_DATA(arr);
          uint8_t *out = (      uint8_t *)PyArray_DATA(packed);

    /* First cell encodes delta from initial_value, which is unpacked[0,0]
     * itself, so the delta is 0 and the byte is always 127. */
    out[0] = 127;

    double prev_row0 = initial_value;

    for (npy_intp y = 0; y < ny; y++) {
        if (y > 0) {
            int icval = (int)((in[y * nx] - prev_row0) * scale + 127.5);
            out[y * nx] = (uint8_t)icval;
            prev_row0 += (icval - 127) * inv_scale;
        }
        double prev = prev_row0;
        for (npy_intp x = 1; x < nx; x++) {
            int icval = (int)((in[y * nx + x] - prev) * scale + 127.5);
            out[y * nx + x] = (uint8_t)icval;
            prev += (icval - 127) * inv_scale;
        }
    }

    Py_DECREF(arr);
    return (PyObject *)packed;
}

/* ---------- module plumbing -------------------------------------------- */

static PyMethodDef PackMethods[] = {
    {"pack_core", pack_core, METH_VARARGS,
     "ARL differential feedback-loop encoder (C extension)."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef packmodule = {
    PyModuleDef_HEAD_INIT, "_pack", NULL, -1, PackMethods
};

PyMODINIT_FUNC
PyInit__pack(void)
{
    import_array();  /* initialise NumPy C-API; returns NULL on failure */
    return PyModule_Create(&packmodule);
}
