#define PY_ARRAY_UNIQUE_SYMBOL QCM_ARRAY_API

#include "qcm_ED_wrap.hpp"
#include "qcm_wrap.hpp"
#include <nanobind/nanobind.h>
#ifdef _OPENMP
#include <omp.h>
#endif

PyObject *qcm_Error = nullptr;

//==============================================================================
// doc string
const char *qcm_help =
    R"{(
qcm performs many tasks associated with quantum cluster methods. The following
functions are available (type help(<function> or <function>? in ipython) for help on any of them):
){";

// numpy's import_array() expands to a `return` on failure, which is illegal in
// the void-returning NB_MODULE body; wrap it in a bool-returning helper.
static bool qcm_import_numpy() {
  import_array1(false);
  return true;
}

NB_MODULE(qcm, m) {
  if (!qcm_import_numpy())
    throw std::runtime_error("qcm: failed to import the NumPy C API");

  m.doc() = qcm_help;

  QCM::qcm_init();

  // All functions are registered through nanobind (converted from the former
  // hand-written CPython C-API wrappers).
  register_qcm_ED(m);
  register_qcm(m);

  qcm_Error = PyErr_NewException("qcm.error", NULL, NULL);
  Py_INCREF(qcm_Error);
  if (PyModule_AddObject(m.ptr(), "error", qcm_Error) < 0) {
    Py_XDECREF(qcm_Error);
    Py_CLEAR(qcm_Error);
    throw std::runtime_error("qcm: failed to register error object");
  }

  qcm_ED_Error = PyErr_NewException("qcm_ED.error", NULL, NULL);
  Py_INCREF(qcm_ED_Error);
  if (PyModule_AddObject(m.ptr(), "error", qcm_ED_Error) < 0) {
    Py_XDECREF(qcm_ED_Error);
    Py_CLEAR(qcm_ED_Error);
    throw std::runtime_error("qcm: failed to register ED error object");
  }
}

/**
  initialization of the library. Must be called before the first call to the
  library.
  */
void QCM::qcm_init() {
  qcm_model = make_shared<lattice_model>();
  QCM::global_parameter_init();

#ifdef _OPENMP
  omp_set_max_active_levels(2);
  cout << "Number of OpenMP threads = " << omp_get_max_threads() << endl;
#endif
}

void qcm_catch(const std::exception &e) { PyErr_SetString(qcm_Error, e.what()); }
void qcm_ED_catch(const std::exception &e) {
  PyErr_SetString(qcm_Error, e.what());
}
