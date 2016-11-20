#include <Python.h>

static PyObject *exmodError;

static PyObject* exmod_say_hello(PyObject* self, PyObject *args){
	const char* msg;
	int sts=0;

	if(!PyArg_ParseTuple(args,"s",&msg)){
		return NULL;
	}
	return NULL;
}

static PyObject* exmod_add(PyObject* self, PyObject *args){

	double a,b;
	double sts = 0;

	if(!PyArg_ParseTuple(args,"dd",&a,&b)){
		return NULL;
	}

	sts = a + b;

	return Py_BuildValue("d", sts);

}

static PyObject* exmod_attr(PyObject* self, PyObject *args){

	double strength,production;
	double attr = 0;

	if(!PyArg_ParseTuple(args,"dd",&strength,&production)){
		return NULL;
	}

	attr = (255.-strength)/255. + production/30.;

	return Py_BuildValue("d", attr);

}


static PyMethodDef exmod_methods[] = {
	//PythonName, C-function name, argument presentation, description
	{"say_hello", exmod_say_hello, METH_VARARGS, "Say Hello"},
	{"add", exmod_add, METH_VARARGS, "Add two numbers in C"},
	{"attr", exmod_attr, METH_VARARGS, "Computes attractiveness"},
	{NULL, NULL, 0, NULL} /* Sentinel */
};

PyMODINIT_FUNC initexmod(void){
	PyObject *m;
	m = Py_InitModule("exmod", exmod_methods);
	if( m == NULL) return;

	exmodError = PyErr_NewException("exmod.error", NULL, NULL);

	Py_INCREF(exmodError);

	PyModule_AddObject(m, "error", exmodError);
}