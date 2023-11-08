#property copyright "Aryn Ducharme"
#property link      "https://www.facebook.com/Original_Synthetics"
#include <Python.h>
#include <Windows.h>
#include <string>

void OnTick()
{
   // Set the path to your Python script
   std::string script_path = "C:\\path\\to\\your\\python_script.py";

   // Add the Python directory to your system path
   std::string python_path = "C:\\path\\to\\python\\directory";
   SetEnvironmentVariable("PATH", (LPCSTR)python_path.c_str());

   // Initialize the Python interpreter
   Py_Initialize();

   // Import your Python script as a module
   PyObject *pModule = PyImport_ImportModule("python_script");

   // Call your Python function
   PyObject *pFunc = PyObject_GetAttrString(pModule, "OnTick");
   PyObject *pResult = PyObject_CallObject(pFunc, NULL);

   // Do something with the result (e.g. print it)
   double result = PyFloat_AsDouble(pResult);
   printf("Result: %f", result);

   // Clean up the Python objects and finalize the interpreter
   Py_DECREF(pModule);
   Py_DECREF(pFunc);
   Py_DECREF(pResult);
   Py_Finalize();
}


    // Do something with the result (e.g. print it)
    double result = PyFloat_AsDouble(pResult);
    std::cout << "Result: " << result << std::endl;

    // Clean up the Python objects and finalize the interpreter
    Py_DECREF(pModule);
    Py_DECREF(pFunc);
    Py_DECREF(pResult);
    Py_Finalize();

    return 0;
}
