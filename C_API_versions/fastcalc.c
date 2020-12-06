#include <Python.h>
#include <arrayobject.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

#define PI 3.14159265359


double getitem2d(PyArrayObject* array2d, int x, int y) {
    npy_intp ptr[2] = { y, x };
    PyObject* dummy = PyArray_GETITEM(array2d, PyArray_GetPtr(array2d, ptr));
    double result = PyFloat_AsDouble(dummy);
    Py_DECREF(dummy);  // necessary for garbage collector (Python)
    return result;
}

void setitem2d(PyArrayObject* array2d, int x, int y, double value) {
    npy_intp ptr[2] = { y, x };
    PyObject* dummy = PyFloat_FromDouble(value);
    PyArray_SETITEM(array2d, PyArray_GetPtr(array2d, ptr), dummy);
    Py_DECREF(dummy);  // necessary for garbage collector (Python)
}

double getitem3d(PyArrayObject* array3d, int x, int y, int z) {
    npy_intp ptr[3] = { y, x, z };
    PyObject* dummy = PyArray_GETITEM(array3d, PyArray_GetPtr(array3d, ptr));
    double result = PyFloat_AsDouble(dummy);
    Py_DECREF(dummy);  // necessary for garbage collector (Python)
    return result;
}

void setitem3d(PyArrayObject* array3d, int x, int y, int z, double value) {
    npy_intp ptr[3] = { y, x, z };
    PyObject* dummy = PyFloat_FromDouble(value);
    PyArray_SETITEM(array3d, PyArray_GetPtr(array3d, ptr), dummy);
    Py_DECREF(dummy);  // necessary for garbage collector (Python)
}


PyDoc_STRVAR(get_coupling_doc, "get_coupling(N, T, M, L) -> 3d-array[N, N, 3] (numpy):\
\n\
Return coupling constants of a field with size NxN\n\
:param N: int = Lattice Side length\n\
:param T: int = Temperature of the Blackhole\n\
:param M: int = Mass of the Blackhole\n\
:param L: int = Curvature of the AdS\n\
:return: 3d-array[N, N, 3] of coupling constants for neighbours (horizontal | up | down)");

PyObject* get_coupling(PyObject* self, PyObject* args, PyObject* kwargs) {

    int N, M, L;
    double T;
    PyObject* js;

    // init random generator with seed for update later
    srand(time(NULL));

    /* Parse positional and keyword arguments */
    static char* keywords[] = { "N", "T", "M", "L", NULL };
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "idii", keywords, &N, &T, &M, &L)) {
        return NULL;
    }

    // create empty 3d array
    npy_intp dims[4] = { N, N, 3, NULL };
    js = PyArray_ZEROS(3, &dims, NPY_FLOAT, 0);
    Py_INCREF(js);

    // calculate prefactors and constants
    double pref_jh, pref_jv, pi_2n;
    pref_jh = PI * L / (2 * (double)N);
    pref_jv = 2 * PI * T / (double)N;
    pi_2n = PI / (2 * (double)N);

    double j_hor, j_ver_up, j_ver_down, sum_;

    // calculate couplings
    for (int y = 0; y < N; y++) {
        j_hor = pref_jh / cos(pi_2n * (y + 0.5));
        j_ver_up = pref_jv / cos(pi_2n * y + pi_2n);
        j_ver_down = pref_jv / cos(pi_2n * y);
        sum_ = 2 * (j_hor + j_ver_up + j_ver_down);
        j_hor /= sum_;
        j_ver_up /= sum_;
        j_ver_down /= sum_;
        for (int x = 0; x < N; x++) {
            setitem3d(js, x, y, 0, j_hor);
            setitem3d(js, x, y, 1, j_ver_up);
            setitem3d(js, x, y, 2, j_ver_down);
        }
    }
    return js;
}


PyDoc_STRVAR(update_doc, "update(field, coupling, N, loops, beta) -> void:\
\n\
Update the given field (Ising model)\n\
:param field: 2d-array (numpy) = statefield of the Ising model\n\
:param coupling: 3d-array (numpy) = couplings to the neighbours\n\
:param N: int = Lattice Side length\n\
:param loops: int = number of updates\n\
:param beta: int? = inverse temperature\n\
:return: void");

PyObject* update(PyObject* self, PyObject* args, PyObject* kwargs) {

    PyArrayObject* field[2], * js[2];
    int loops, beta, N;

    /* Parse positional and keyword arguments */
    static char* keywords[] = { "field", "coupling", "N", "loops", "beta", NULL };
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!O!iii", keywords, &PyArray_Type, &field[0], &PyArray_Type, &js[0], &N, &loops, &beta)) {
        return NULL;
    }
    field[1] = js[1] = NULL;
    int x, y, s_y, x_add, x_sub;
    double h;


    for (int i = 0; i < loops; i++) {
        y = rand() % (N - 2) + 1;
        s_y = y + 2;
        x = rand() % (s_y);

        x_add = (x + 1) % s_y;
        x_sub = (x - 1 + s_y) % s_y;
        h = getitem3d(js[0], x, y, 0) * (getitem2d(field[0], x_add, y) + getitem2d(field[0], x_sub, y)) +
            getitem3d(js[0], x, y, 1) * (getitem2d(field[0], x, y + 1) + getitem2d(field[0], x_add, y + 1)) +
            getitem3d(js[0], x, y, 2) * (getitem2d(field[0], x, y - 1) + getitem2d(field[0], x_sub, y - 1));

        if ((double)rand() / (double)RAND_MAX < 1 / (1 + exp(-2 * (double)beta * h))) {
            setitem2d(field[0], x, y, 1);
        }
        else {
            setitem2d(field[0], x, y, -1);
        }

    }
    Py_RETURN_NONE;
}


/*
 * List of functions to add to fastcalc in exec_fastcalc().
 */
static PyMethodDef fastcalc_functions[] = {
    { "get_coupling", (PyCFunction)get_coupling, METH_VARARGS | METH_KEYWORDS, get_coupling_doc },
    { "update", (PyCFunction)update, METH_VARARGS | METH_KEYWORDS, update_doc },
    { NULL, NULL, 0, NULL } /* marks end of array */
};

/*
 * Initialize fastcalc. May be called multiple times, so avoid
 * using static state.
 */
int exec_fastcalc(PyObject* module) {
    PyModule_AddFunctions(module, fastcalc_functions);

    PyModule_AddStringConstant(module, "__author__", "Yanick Thurn");
    PyModule_AddStringConstant(module, "__version__", "1.0.0");
    PyModule_AddIntConstant(module, "year", 2020);

    return 0; /* success */
}

/*
 * Documentation for fastcalc.
 */
PyDoc_STRVAR(fastcalc_doc, "The fastcalc module");


static PyModuleDef_Slot fastcalc_slots[] = {
    { Py_mod_exec, exec_fastcalc },
    { 0, NULL }
};

static PyModuleDef fastcalc_def = {
    PyModuleDef_HEAD_INIT,
    "fastcalc",
    fastcalc_doc,
    0,              /* m_size */
    NULL,           /* m_methods */
    fastcalc_slots,
    NULL,           /* m_traverse */
    NULL,           /* m_clear */
    NULL,           /* m_free */
};

PyMODINIT_FUNC PyInit_fastcalc() {
    import_array();
    return PyModuleDef_Init(&fastcalc_def);
}
