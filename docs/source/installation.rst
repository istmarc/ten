Installation
============

Install from source
-------------------
The ten library can be installed from its source code available at https://github.com/istmarc/ten.

The following instructions describe how to clone the repository and install the library on Linux and MacOS:

.. code-block:: bash

   $ git clone https://github.com/istmarc/ten.git
   $ cd ten
   $ mkdir build
   $ cd build
   $ cmake ..
   $ make install

Set the install prefix:

.. code-block:: bash

   $ cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local
   $ make install

