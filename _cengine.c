#include <Python.h>
#include <structmember.h>
#include "rin_api.h"
#include "rin/backends/rin_backend.h"

/* ------------------------------------------------------------------ */
/*  Exception                                                        */
/* ------------------------------------------------------------------ */
static PyObject *RinException;

/* ------------------------------------------------------------------ */
/*  RinContext : Python object wrapping RinContext*                 */
/* ------------------------------------------------------------------ */
typedef struct {
    PyObject_HEAD
    ThorContext *ctx;
    int          closed;
} PyRinContext;

static int
PyRinContext_init(PyRinContext *self, PyObject *args, PyObject *kwds)
{
    self->ctx = thor_create();
    if (!self->ctx) {
        PyErr_SetString(RinException, "thor_create() returned NULL");
        return -1;
    }
    self->closed = 0;
    return 0;
}

static void
PyRinContext_dealloc(PyRinContext *self)
{
    if (self->ctx && !self->closed) {
        thor_destroy(self->ctx);
    }
    self->ctx = NULL;
    Py_TYPE(self)->tp_free((PyObject *)self);
}

/* load_model(self, path: str) -> None */
static PyObject *
PyRinContext_load_model(PyRinContext *self, PyObject *arg)
{
    const char *path;

    if (self->closed || !self->ctx) {
        PyErr_SetString(RinException, "context is closed");
        return NULL;
    }
    if (!PyArg_Parse(arg, "s", &path))
        return NULL;

    ThorStatus st = thor_load_model(self->ctx, path);
    if (st != THOR_OK) {
        PyErr_Format(RinException, "thor_load_model(%s) failed: %d", path, st);
        return NULL;
    }
    Py_RETURN_NONE;
}

/* get_model_info(self) -> dict */
static PyObject *
PyRinContext_get_model_info(PyRinContext *self, PyObject *Py_UNUSED(ignored))
{
    ThorModelInfo info;

    if (self->closed || !self->ctx) {
        PyErr_SetString(RinException, "context is closed");
        return NULL;
    }
    ThorStatus st = thor_get_model_info(self->ctx, &info);
    if (st != THOR_OK) {
        PyErr_Format(RinException, "thor_get_model_info failed: %d", st);
        return NULL;
    }
    return Py_BuildValue("{s:I,s:I,s:I,s:I,s:I,s:I,s:I,s:I,s:f}",
        "num_layers",      info.num_layers,
        "model_dim",       info.model_dim,
        "vocab_size",      info.vocab_size,
        "num_heads",       info.num_heads,
        "max_seq_len",     info.max_seq_len,
        "ffn_dim",         info.ffn_dim,
        "num_parameters",  info.num_parameters,
        "architecture",    info.architecture,
        "size_mb",         info.size_mb);
}

/* mode property getter/setter */
static PyObject *
PyRinContext_get_mode(PyRinContext *self, void *closure)
{
    (void)closure;
    if (self->closed || !self->ctx) {
        Py_RETURN_NONE;
    }
    return PyLong_FromLong((long)thor_get_mode(self->ctx));
}

static int
PyRinContext_set_mode(PyRinContext *self, PyObject *value, void *closure)
{
    (void)closure;
    if (self->closed || !self->ctx) {
        PyErr_SetString(RinException, "context is closed");
        return -1;
    }
    if (!PyLong_Check(value)) {
        PyErr_SetString(PyExc_TypeError, "mode must be an integer");
        return -1;
    }
    thor_set_mode(self->ctx, (ThorMode)PyLong_AsLong(value));
    return 0;
}

/* sampling params */
static PyObject *
PyRinContext_set_temperature(PyRinContext *self, PyObject *arg)
{
    if (self->closed || !self->ctx) { Py_RETURN_NONE; }
    float v = (float)PyFloat_AsDouble(arg);
    if (PyErr_Occurred()) return NULL;
    thor_set_temperature(self->ctx, v);
    Py_RETURN_NONE;
}

static PyObject *
PyRinContext_set_top_k(PyRinContext *self, PyObject *arg)
{
    if (self->closed || !self->ctx) { Py_RETURN_NONE; }
    uint32_t v = (uint32_t)PyLong_AsLong(arg);
    if (PyErr_Occurred()) return NULL;
    thor_set_top_k(self->ctx, v);
    Py_RETURN_NONE;
}

static PyObject *
PyRinContext_set_top_p(PyRinContext *self, PyObject *arg)
{
    if (self->closed || !self->ctx) { Py_RETURN_NONE; }
    float v = (float)PyFloat_AsDouble(arg);
    if (PyErr_Occurred()) return NULL;
    thor_set_top_p(self->ctx, v);
    Py_RETURN_NONE;
}

static PyObject *
PyRinContext_set_power_budget(PyRinContext *self, PyObject *arg)
{
    if (self->closed || !self->ctx) { Py_RETURN_NONE; }
    float v = (float)PyFloat_AsDouble(arg);
    if (PyErr_Occurred()) return NULL;
    thor_set_power_budget(self->ctx, v);
    Py_RETURN_NONE;
}

/* infer(self, input_ids: list, max_output: int = 1) -> dict */
static PyObject *
PyRinContext_infer(PyRinContext *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"input_ids", "max_output", NULL};
    PyObject    *input_list;
    uint32_t     max_output = 1;

    if (self->closed || !self->ctx) {
        PyErr_SetString(RinException, "context is closed");
        return NULL;
    }
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!|I", kwlist,
                                     &PyList_Type, &input_list, &max_output))
        return NULL;

    Py_ssize_t n = PyList_Size(input_list);
    uint32_t  *ids = (uint32_t *)PyMem_Malloc((size_t)n * sizeof(uint32_t));
    if (!ids) return PyErr_NoMemory();

    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject *item = PyList_GetItem(input_list, i);
        if (!PyLong_Check(item)) {
            PyMem_Free(ids);
            PyErr_SetString(PyExc_TypeError, "input_ids must be integers");
            return NULL;
        }
        ids[i] = (uint32_t)PyLong_AsLong(item);
    }

    ThorResult result;
    ThorStatus st = thor_infer(self->ctx, ids, (uint32_t)n, max_output, &result);
    PyMem_Free(ids);

    if (st != THOR_OK) {
        PyErr_Format(RinException, "thor_infer failed: %d", st);
        return NULL;
    }

    /* build token list */
    PyObject *token_list = PyList_New((Py_ssize_t)result.num_tokens);
    if (!token_list) {
        thor_free_result(&result);
        return PyErr_NoMemory();
    }
    for (uint32_t i = 0; i < result.num_tokens; i++) {
        PyList_SET_ITEM(token_list, (Py_ssize_t)i,
                        PyLong_FromUnsignedLong(result.tokens[i]));
    }
    thor_free_result(&result);

    return Py_BuildValue("{s:O,s:I,s:d,s:f,s:K}",
        "tokens",            token_list,
        "num_tokens",        result.num_tokens,
        "energy_joules",     result.energy_joules,
        "tokens_per_second", result.tokens_per_second,
        "latency_ns",        result.latency_ns);
}

/* encode(self, text: str) -> list */
static PyObject *
PyRinContext_encode(PyRinContext *self, PyObject *arg)
{
    const char *text;

    if (self->closed || !self->ctx) {
        PyErr_SetString(RinException, "context is closed");
        return NULL;
    }
    if (!PyArg_Parse(arg, "s", &text))
        return NULL;

    int max_ids = (int)strlen(text) * 4 + 256;
    uint32_t *ids = (uint32_t *)PyMem_Malloc((size_t)max_ids * sizeof(uint32_t));
    if (!ids) return PyErr_NoMemory();

    int n = thor_encode(self->ctx, text, ids, max_ids);
    if (n < 0) {
        PyMem_Free(ids);
        PyErr_Format(RinException, "thor_encode failed: %d", n);
        return NULL;
    }

    PyObject *lst = PyList_New((Py_ssize_t)n);
    if (!lst) { PyMem_Free(ids); return PyErr_NoMemory(); }
    for (int i = 0; i < n; i++)
        PyList_SET_ITEM(lst, (Py_ssize_t)i, PyLong_FromUnsignedLong(ids[i]));

    PyMem_Free(ids);
    return lst;
}

/* decode(self, ids: list) -> str */
static PyObject *
PyRinContext_decode(PyRinContext *self, PyObject *arg)
{
    if (self->closed || !self->ctx) {
        PyErr_SetString(RinException, "context is closed");
        return NULL;
    }

    PyObject *seq = PySequence_Fast(arg, "decode() argument must be iterable");
    if (!seq) return NULL;

    Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
    uint32_t *ids = (uint32_t *)PyMem_Malloc((size_t)n * sizeof(uint32_t));
    if (!ids) { Py_DECREF(seq); return PyErr_NoMemory(); }

    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
        if (!PyLong_Check(item)) {
            PyMem_Free(ids); Py_DECREF(seq);
            PyErr_SetString(PyExc_TypeError, "ids must be integers");
            return NULL;
        }
        ids[i] = (uint32_t)PyLong_AsLong(item);
    }
    Py_DECREF(seq);

    int max_text = (int)n * 16 + 256;
    char *buf = (char *)PyMem_Malloc((size_t)max_text);
    if (!buf) { PyMem_Free(ids); return PyErr_NoMemory(); }

    thor_decode(self->ctx, ids, (int)n, buf, max_text);
    PyMem_Free(ids);

    PyObject *result = PyUnicode_FromString(buf);
    PyMem_Free(buf);
    return result;
}

/* get_charset(self) -> str */
static PyObject *
PyRinContext_get_charset(PyRinContext *self, PyObject *Py_UNUSED(ignored))
{
    if (self->closed || !self->ctx) {
        PyErr_SetString(RinException, "context is closed");
        return NULL;
    }
    int vocab_size = 0;
    const char *charset = thor_get_charset(self->ctx, &vocab_size);
    if (!charset) {
        PyErr_SetString(RinException, "thor_get_charset returned NULL");
        return NULL;
    }
    return PyUnicode_FromString(charset);
}

/* energy properties */
static PyObject *
PyRinContext_get_energy_joules(PyRinContext *self, void *closure)
{
    (void)closure;
    if (self->closed || !self->ctx) { Py_RETURN_NONE; }
    return PyFloat_FromDouble(thor_get_energy_joules(self->ctx));
}

static PyObject *
PyRinContext_get_energy_millijoules(PyRinContext *self, void *closure)
{
    (void)closure;
    if (self->closed || !self->ctx) { Py_RETURN_NONE; }
    return PyFloat_FromDouble(thor_get_energy_millijoules(self->ctx));
}

static PyObject *
PyRinContext_get_inference_count(PyRinContext *self, void *closure)
{
    (void)closure;
    if (self->closed || !self->ctx) { Py_RETURN_NONE; }
    return PyLong_FromUnsignedLongLong(thor_get_inference_count(self->ctx));
}

static PyObject *
PyRinContext_get_total_tokens(PyRinContext *self, void *closure)
{
    (void)closure;
    if (self->closed || !self->ctx) { Py_RETURN_NONE; }
    return PyLong_FromUnsignedLongLong(thor_get_total_tokens(self->ctx));
}

/* profile(self, mode, warmup, iterations) -> (ms/tok, tok/s) */
static PyObject *
PyRinContext_profile(PyRinContext *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"mode", "warmup", "iterations", NULL};
    int mode = 0;
    uint32_t warmup = 10, iterations = 100;

    if (self->closed || !self->ctx) {
        PyErr_SetString(RinException, "context is closed");
        return NULL;
    }
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "I|II", kwlist,
                                     &mode, &warmup, &iterations))
        return NULL;

    double ms_per_tok, toks_per_sec;
    ThorStatus st = thor_profile(
        self->ctx, (ThorMode)mode, warmup, iterations,
        &ms_per_tok, &toks_per_sec);
    if (st != THOR_OK) {
        PyErr_Format(RinException, "thor_profile failed: %d", st);
        return NULL;
    }
    return Py_BuildValue("(dd)", ms_per_tok, toks_per_sec);
}

/* close(self) -> None  (explicit cleanup) */
static PyObject *
PyRinContext_close(PyRinContext *self, PyObject *Py_UNUSED(ignored))
{
    if (self->ctx && !self->closed) {
        thor_destroy(self->ctx);
    }
    self->ctx = NULL;
    self->closed = 1;
    Py_RETURN_NONE;
}

/* Methods table */
static PyMethodDef PyRinContext_methods[] = {
    {"close",
     (PyCFunction)PyRinContext_close, METH_NOARGS,
     "close() -> None\nExplicitly release the C context."},
    {"load_model",
     (PyCFunction)PyRinContext_load_model, METH_O,
     "load_model(path) -> None\nLoad a model file."},
    {"get_model_info",
     (PyCFunction)PyRinContext_get_model_info, METH_NOARGS,
     "get_model_info() -> dict\nReturn model metadata."},
    {"infer",
     (PyCFunction)PyRinContext_infer, METH_VARARGS | METH_KEYWORDS,
     "infer(input_ids, max_output=1) -> dict\nRun inference."},
    {"encode",
     (PyCFunction)PyRinContext_encode, METH_O,
     "encode(text) -> list\nTokenize text."},
    {"decode",
     (PyCFunction)PyRinContext_decode, METH_O,
     "decode(ids) -> str\nDecode token IDs to text."},
    {"get_charset",
     (PyCFunction)PyRinContext_get_charset, METH_NOARGS,
     "get_charset() -> str\nReturn the model's character set."},
    {"set_temperature",
     (PyCFunction)PyRinContext_set_temperature, METH_O,
     "set_temperature(temp) -> None"},
    {"set_top_k",
     (PyCFunction)PyRinContext_set_top_k, METH_O,
     "set_top_k(k) -> None"},
    {"set_top_p",
     (PyCFunction)PyRinContext_set_top_p, METH_O,
     "set_top_p(p) -> None"},
    {"set_power_budget",
     (PyCFunction)PyRinContext_set_power_budget, METH_O,
     "set_power_budget(watts) -> None"},
    {"profile",
     (PyCFunction)PyRinContext_profile, METH_VARARGS | METH_KEYWORDS,
     "profile(mode, warmup=10, iterations=100) -> (ms_per_tok, tok_per_sec)"},
    {NULL, NULL, 0, NULL}
};

/* Getters / setters */
static PyGetSetDef PyRinContext_getset[] = {
    {"mode",
     (getter)PyRinContext_get_mode, (setter)PyRinContext_set_mode,
     "Inference mode (integer).", NULL},
    {"energy_joules",
     (getter)PyRinContext_get_energy_joules, NULL,
     "Accumulated energy in joules.", NULL},
    {"energy_millijoules",
     (getter)PyRinContext_get_energy_millijoules, NULL,
     "Accumulated energy in millijoules.", NULL},
    {"inference_count",
     (getter)PyRinContext_get_inference_count, NULL,
     "Total inference calls.", NULL},
    {"total_tokens",
     (getter)PyRinContext_get_total_tokens, NULL,
     "Total tokens processed.", NULL},
    {NULL, NULL, NULL, NULL, NULL}
};

static PyTypeObject PyRinContextType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name      = "_cengine.RinContext",
    .tp_basicsize = sizeof(PyRinContext),
    .tp_itemsize  = 0,
    .tp_dealloc   = (destructor)PyRinContext_dealloc,
    .tp_flags     = Py_TPFLAGS_DEFAULT,
    .tp_doc       = "RIN Engine Context",
    .tp_methods   = PyRinContext_methods,
    .tp_getset    = PyRinContext_getset,
    .tp_init      = (initproc)PyRinContext_init,
    .tp_new       = PyType_GenericNew,
};

/* ------------------------------------------------------------------ */
/*  Module-level helpers                                              */
/* ------------------------------------------------------------------ */

static PyObject *
module_version(PyObject *self, PyObject *Py_UNUSED(ignored))
{
    (void)self;
    return PyUnicode_FromString(thor_version());
}

static PyObject *
module_version_numbers(PyObject *self, PyObject *Py_UNUSED(ignored))
{
    (void)self;
    uint32_t major, minor, patch;
    thor_version_numbers(&major, &minor, &patch);
    return Py_BuildValue("(III)", major, minor, patch);
}

static PyObject *
module_best_backend(PyObject *self, PyObject *Py_UNUSED(ignored))
{
    (void)self;
    return PyUnicode_FromString(rin_best_backend());
}

static PyObject *
module_neon_available(PyObject *self, PyObject *Py_UNUSED(ignored))
{
    (void)self;
    return PyBool_FromLong((long)thor_neon_available());
}

static PyObject *
module_wasm_available(PyObject *self, PyObject *Py_UNUSED(ignored))
{
    (void)self;
    return PyBool_FromLong((long)thor_wasm_available());
}

static PyMethodDef module_methods[] = {
    {"version",           module_version,           METH_NOARGS,
     "Return the RIN C library version string."},
    {"version_numbers",   module_version_numbers,   METH_NOARGS,
     "Return (major, minor, patch)."},
    {"best_backend",      module_best_backend,      METH_NOARGS,
     "Return the name of the best available SIMD backend."},
    {"neon_available",    module_neon_available,    METH_NOARGS,
     "Return True if ARM NEON kernels are compiled in."},
    {"wasm_available",    module_wasm_available,    METH_NOARGS,
     "Return True if WASM SIMD kernels are compiled in."},
    {NULL, NULL, 0, NULL}
};

/* ------------------------------------------------------------------ */
/*  Module definition                                                 */
/* ------------------------------------------------------------------ */

static struct PyModuleDef _cengine_module = {
    PyModuleDef_HEAD_INIT,
    .m_name     = "_cengine",
    .m_doc      = "Native CPython extension for the RIN Engine runtime.",
    .m_size     = -1,
    .m_methods  = module_methods,
};

PyMODINIT_FUNC
PyInit__cengine(void)
{
    if (PyType_Ready(&PyRinContextType) < 0)
        return NULL;

    PyObject *m = PyModule_Create(&_cengine_module);
    if (!m) return NULL;

    Py_INCREF(&PyRinContextType);
    if (PyModule_AddObject(m, "RinContext",
                           (PyObject *)&PyRinContextType) < 0) {
        Py_DECREF(&PyRinContextType);
        Py_DECREF(m);
        return NULL;
    }

    /* Exception */
    RinException = PyErr_NewException("_cengine.RinException",
                                       PyExc_RuntimeError, NULL);
    if (PyModule_AddObject(m, "RinException", RinException) < 0) {
        Py_DECREF(RinException);
        Py_DECREF(m);
        return NULL;
    }

    /* Mode constants */
#define ADD_INT(m, name, val) \
    do { PyObject *v = PyLong_FromLong(val); \
         PyModule_AddObject(m, name, v); } while(0)

    ADD_INT(m, "MODE_MLP",          0);
    ADD_INT(m, "MODE_SNN",          1);
    ADD_INT(m, "MODE_ATTN",         2);
    ADD_INT(m, "MODE_THOR",         3);
    ADD_INT(m, "MODE_TRANSFORMER",  4);

    ADD_INT(m, "STATUS_OK",               0);
    ADD_INT(m, "STATUS_ERR_INIT",        -1);
    ADD_INT(m, "STATUS_ERR_MEMORY",      -2);
    ADD_INT(m, "STATUS_ERR_WEIGHTS",     -3);
    ADD_INT(m, "STATUS_ERR_INFERENCE",   -4);
    ADD_INT(m, "STATUS_ERR_NOT_INITED",  -5);
    ADD_INT(m, "STATUS_ERR_INVALID",     -6);
    ADD_INT(m, "STATUS_ERR_UNSUPPORTED", -7);

    return m;
}
