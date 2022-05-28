perl -pi.bak -e 's%, CompareUFunc%, (PyUFuncGenericFunction) CompareUFunc%g' tensorflow/python/lib/core/bfloat16.cc
