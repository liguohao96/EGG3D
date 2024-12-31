#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "extension.h"

namespace cpp{

    PYBIND11_MODULE(EXTENSION_NAME, m){
        m.def(
          "save_ply_file",      // name in Python
          &save_ply_file,       // function pointer in Cpp
          "save as ply format"  // description and others
        ); 
    }
} // namespace cpp